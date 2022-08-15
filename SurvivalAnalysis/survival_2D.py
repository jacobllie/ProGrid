from survival_analysis_4 import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats import f, ttest_ind, chi2, f_oneway
import seaborn as sb
from scipy import stats, optimize
from scipy.interpolate import interp1d
from utils import poisson_regression, data_stacking_2, mean_survival ,corrfunc
from plotting_functions_survival import survival_histogram
import cv2
from sklearn.model_selection import train_test_split
import pickle
import string
from datetime import date

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021"
time = ["18112019", "20112019"]
mode = ["Control", "Open", "GRID Stripes"]
dose = ["02", "05", "10"]
ctrl_dose = ["00"]
template_file_control =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\Control\\A549-1811-K1-TemplateMask.csv"
template_file_open = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\Open\\A549-1811-02-open-A-TemplateMask.csv"
template_file_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\GRID Stripes\\A549-1811-02-gridS-A-TemplateMask.csv"
dose_path_open = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open_test.npy"
dose_path_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_grid_test.npy"
save_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\310821\\ColonyData"
position = ["A","B","C","D"]
colors = ["b","g","r","grey","m","y","black","saddlebrown"]

kernel_size_mm = [1]#[0.5,1,2,3,4]
kernel_size_p = [int(i*47) for i in kernel_size_mm]

cropping_limits_2D = [200,2000,350,1850]

peak_dist_reg = False
num_regressors = 4

plt.style.use("seaborn")
"""
18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
We therefore combine these data to find alpha beta for open field irradiation.
"""

plt.imshow(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\Control\\A549-1811-K1-SegMask.csv"))
plt.close()

header = pd.MultiIndex.from_product([['2 Gy','5 Gy', '10 Gy'],
                                    ['Peak','Valley'], ["RPD","p-value"]])

df_pois = pd.DataFrame(columns = header)

dose_var = np.zeros(len(kernel_size_mm))
excessive_zeros = np.zeros(len(kernel_size_mm))
rel_diff2 = np.zeros((len(kernel_size_mm),2,2))
rel_diff5 = np.zeros((len(kernel_size_mm), 2,2))
# rel_diff = np.zeros((len(kernel_size_mm), 3))
rel_diff10 = np.zeros((len(kernel_size_mm), 2,2))
poisson_test = False
error_df = {}


"""
Finding the number of counted colonies for control (0Gy) and open field
experiments (2Gy and 5Gy)
"""
for i in range(len(kernel_size_mm)):
    print("-------------------------")
    print("running for kernel " + str(kernel_size_mm[i]) + "mm")
    print("-------------------------")
    control  = True
    if control == True:
        flask_template = np.asarray(pd.read_csv(template_file_control))[cropping_limits_2D[0]:cropping_limits_2D[1],cropping_limits_2D[2]:cropping_limits_2D[3]]

        #plt.imshow(flask_template)
        #plt.show()

        survival_control = survival_analysis(folder, time, mode[0], position,
                           kernel_size_p[i], dose_map_path = None,
                           template_file = template_file_control,
                           save_path = save_path, dose = ctrl_dose, cropping_limits = cropping_limits_2D,
                           data_extract = False)
        ColonyData_control, data_control = survival_control.data_acquisition()

        colony_map_ctrl = survival_control.Colonymap()
        pooled_SC_ctrl = survival_control.Quadrat()

        print("Control survival standard deviation")
        print(np.std(pooled_SC_ctrl))


        #checking for excessive zeros
        excessive_zeros[i] = len(pooled_SC_ctrl[pooled_SC_ctrl < 1])
        print(excessive_zeros[i])
        # survival_histogram(None, None, pooled_SC_ctrl, 0,kernel_size_mm[i],mode = "Control")



    open_= True

    if open_ == True:
        #removing edges
        #initializing class
        survival_open = survival_analysis(folder, time, mode[1], position,
                                          kernel_size_p[i], dose_path_open,
                                          template_file_open,
                                          save_path = save_path, dose = dose,
                                          cropping_limits = cropping_limits_2D, data_extract = False)
        #gathering colonydata, and count data
        ColonyData_open, data_open  = survival_open.data_acquisition()
        #placing colonies in their respective coordinates
        colony_map_open = survival_open.Colonymap()

        survival_open.registration()



        survival_open.Quadrat() #2D analysis

        dose2Gy_open, SC_open_2Gy = survival_open.SC(2)
        dose5Gy_open, SC_open_5Gy = survival_open.SC(5)
        print("Standard deviation survival OPEN 2 Gy")
        print(np.std(SC_open_2Gy))
        print("Standard deviation survival OPEN 5 Gy")
        print(np.std(SC_open_5Gy))


        df = pd.DataFrame({"Ctrl":pooled_SC_ctrl.ravel(), "OPEN 2 Gy":SC_open_2Gy.ravel(), "OPEN 5 GY":SC_open_5Gy.ravel()})
        plt.ylabel("SC",fontsize = 15)
        sb.boxplot(data = df)
        plt.tight_layout()
        # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\CTRL_OPEN_boxplot.png", dpi = 300)

        print(df)


    test_image = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))
    Grid = True
    if Grid == True:

        survival_grid = survival_analysis(folder, time, mode[2], position,
                                          kernel_size_p[i], dose_path_grid,
                                          template_file_grid, save_path = save_path,
                                          dose = dose, cropping_limits =  cropping_limits_2D, data_extract = False)
        ColonyData_grid, data_grid  = survival_grid.data_acquisition()
        colony_map_grid = survival_grid.Colonymap()


        #remember to unhash this when performing 2D analysis
        survival_grid.registration()

        _,dose_map = survival_grid.Quadrat("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\grid_survival10Gy_300821_stripes_{}mm.png".format(kernel_size_mm[i]))


        dose2Gy_grid, SC_grid_2Gy = survival_grid.SC(2)
        dose5Gy_grid, SC_grid_5Gy = survival_grid.SC(5)
        dose10Gy_grid, SC_grid_10Gy = survival_grid.SC(10)


        SC_grid_df = pd.DataFrame({"GRID Stripes 2 Gy":SC_grid_2Gy.ravel(), "GRID Stripes 5 Gy":SC_grid_2Gy.ravel(), "GRID Stripes 10 Gy": SC_grid_10Gy.ravel()})
        sb.boxplot(data = SC_grid_df)
        plt.ylabel("SC",fontsize = 15)
        plt.tight_layout()
        # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\GRID_Stripes_boxplot.png", dpi = 300)

        plt.close()

        dose_var[i] = np.var(dose5Gy_grid)

        """
        Plot histogram of survival
        Separate into dose categories
        """

        if poisson_test:
            """
            Testing if peak and valley survival data is Poisson distributed
            Also testing other performance criteria as dose variance and excessive zeros.
            """
            print(survival_histogram(dose_map*2/5,dose2Gy_grid, SC_grid_2Gy, 2, kernel_size_mm[i]).shape)


            rel_diff2[i] = survival_histogram(dose_map*2/5,dose2Gy_grid, SC_grid_2Gy, 2, kernel_size_mm[i])
            rel_diff5[i] = survival_histogram(dose_map, dose5Gy_grid, SC_grid_5Gy, 5, kernel_size_mm[i])
            rel_diff10[i] = survival_histogram(dose_map*10/5, dose10Gy_grid, SC_grid_10Gy, 10, kernel_size_mm[i])



            # x = np.zeros((6,6))
            x = np.zeros((12,12))

            x[0] = np.ravel([rel_diff2[i], rel_diff5[i], rel_diff10[i]])
            tmp_df = pd.DataFrame(x, columns = header) #with three headers and two subheader = 6 , we need 6 rows

            for j in range(1,len(x)):
                tmp_df = tmp_df.drop([len(tmp_df)-1])
            #tmp_df = tmp_df.drop([len(tmp_df)-1, len(tmp_df)-2,len(tmp_df)-3,len(tmp_df)-4,len(tmp_df)-5])

            df_pois = pd.concat([df_pois,tmp_df])


            if i == len(kernel_size_mm) - 1:

                """
                Plotting rel diff for 5 Gy
                """
                #df_pois = df_pois.set_index(pd.Index(kernel_size_mm, name = "Kernel Size [mm]"))
                # df_pois.to_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\poisson_test.csv",)

                fig, ax = plt.subplots(nrows = 1,ncols = 2)
                plt.suptitle("Quadrat Size estimation for 5 Gy")
                ax1 = ax[0]

                color = 'tab:red'
                ax1.set_xlabel('Quadrat size [mm]', fontsize = 14)
                ax1.set_ylabel(string.capwords("# 0's"), color=color)
                ax1.plot(kernel_size_mm, excessive_zeros, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax2.set_ylabel('Variance between quadrat doses', color=color, fontsize = 14)  # we already handled the x-label with ax1
                ax2.plot(kernel_size_mm, dose_var, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                ax3 = ax[1]
                color = 'tab:red'
                ax3.set_xlabel('Quadrat size [mm]',fontsize = 14)
                ax3.set_ylabel("Peak RPD", color=color, fontsize = 14)  #string.capwords("Peak RPD")
                ax3.plot(kernel_size_mm, rel_diff5[:,0,0], color=color)
                ax3.tick_params(axis='y', labelcolor=color)

                ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax4.set_ylabel('Valley RPD', color=color, fontsize = 14)  # we already handled the x-label with ax1
                ax4.plot(kernel_size_mm, rel_diff5[:,1,0], color=color)
                ax4.tick_params(axis='y', labelcolor=color)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                # fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\kernel_size_eval_5Gy.png", pad_inches = 0.2, bbox_inches = "tight", dpi = 300)
                plt.show()

        #[210,2240,405,1900]
        if peak_dist_reg or num_regressors == 4:  #for 4 regressors we need peak dist anyways
            peak_dist = survival_grid.nearest_peak(cropping_limits_2D)/(47*10) #distance from pooled dose quadrat centers to peak dose in cm
            #same peak dist for all doses
            peak_dist = np.tile(np.ravel(peak_dist),len(dose)*pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1]*pooled_SC_ctrl.shape[2])

            print(peak_dist.shape)

        #print(pooled_SC_ctrl.shape)




    """
    Poisson regression predicting number of survivors
    """

    method3 = True
    """
    Final method
    """
    if method3 == True:
        """
        Introduce distance to peak as a regressor
        """
        tot_irradiated_area = 25*100  #mm^2 #24.505*100 #mm^2
        peak_area = 3*215+170.75 #3 full peaks, 1 trapezoidal peak mm^2
        peak_area_ratio = peak_area/tot_irradiated_area
        valley_area_ratio = 1-peak_area_ratio  #(tot_irradiatet_area-peak_area)/tot_irradiatet_area

        print(peak_area_ratio,valley_area_ratio)
        SC_open, tot_dose_axis_open = data_stacking_2(False, SC_open_2Gy,
                                                    SC_open_5Gy, dose2Gy_open,
                                                    dose5Gy_open)
        SC_grid, tot_dose_axis_grid = data_stacking_2(True, SC_grid_2Gy,
                                                      SC_grid_5Gy, SC_grid_10Gy,
                                                      dose2Gy_grid, dose5Gy_grid, dose10Gy_grid)
        print(SC_open.shape,SC_grid.shape,tot_dose_axis_open.shape,tot_dose_axis_grid.shape)

        grid_dots = True
        if grid_dots == True:
            if peak_dist_reg and num_regressors == 3:
                with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_X_{}regressors_peak_dist.pickle".format(num_regressors), 'rb') as handle:
                    tmp_data = pickle.load(handle)
                    X_grid_dots = tmp_data[kernel_size_mm[i]]
                    print(X_grid_dots.shape)
                with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_SC_{}regressors_peak_dist.pickle".format(num_regressors), 'rb') as handle:
                    tmp_data = pickle.load(handle)
                    SC_grid_dots = tmp_data[kernel_size_mm[i]]
                    print(SC_grid_dots.shape)
            elif not peak_dist_reg and num_regressors == 3:
                with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_X_{}regressors_peak_area.pickle".format(num_regressors), 'rb') as handle:
                    tmp_data = pickle.load(handle)
                    X_grid_dots = tmp_data[kernel_size_mm[i]]
                    print(X_grid_dots.shape)
                with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_SC_{}regressors_peak_area.pickle".format(num_regressors), 'rb') as handle:
                    tmp_data = pickle.load(handle)
                    SC_grid_dots = tmp_data[kernel_size_mm[i]]
                    print(SC_grid_dots.shape)
            else:
                with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_X_{}regressors.pickle".format(num_regressors), 'rb') as handle:
                    tmp_data = pickle.load(handle)
                    X_grid_dots = tmp_data[kernel_size_mm[i]]
                    print(X_grid_dots.shape)
                with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_SC_{}regressors.pickle".format(num_regressors), 'rb') as handle:
                    tmp_data = pickle.load(handle)
                    SC_grid_dots = tmp_data[kernel_size_mm[i]]
                    print(SC_grid_dots.shape)



            """fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111, projection='3d')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.scatter(X_grid_dots[:,1],X_grid_dots[:,-1] , SC_grid_dots)  #dose distance survival
            #ax.invert_yaxis()
            plt.close()"""
            # X_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_X_w.g_factor.npy")
            # SC_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_SC_w.g_factor.npy")
            X_grid_dots_train, X_grid_dots_test, SC_grid_dots_train, SC_grid_dots_test = train_test_split(X_grid_dots,SC_grid_dots, test_size = 0.2)

            SC_grid_dots_len = len(SC_grid_dots_train)

        print(SC_grid.shape)
        """
        Adding the control survival and doses (0Gy) to open. Then make individual design matrix with different G factors
        """
        #peak_dist = np.tile(np.ravel(peak_dist),len(dose)*pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1])
        #peak_dist = [i**2 if i > 0 else 0 for i in stacked_dist]
        # tmp = np.repeat(dose2Gy_grid*0,pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1]*pooled_SC_ctrl.shape[2])
        if num_regressors == 1:
            tot_len = len(np.ravel(pooled_SC_ctrl))
            X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len)]).T#design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, num_regressors, 0,0,0)

            tot_len = len(tot_dose_axis_open)
            X_open =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis_open]).T#design_matrix(len(SC_open),tot_dose_axis_open,num_regressors,1,0,0)
            tot_len = len(tot_dose_axis_grid)
            X_grid =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis_grid]).T
        elif num_regressors == 2:
            tot_len = len(np.ravel(pooled_SC_ctrl))
            X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len),np.repeat(0,tot_len)]).T#design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, num_regressors, 0,0,0)

            tot_len = len(tot_dose_axis_open)
            X_open =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis_open, tot_dose_axis_open**2]).T#design_matrix(len(SC_open),tot_dose_axis_open,num_regressors,1,0,0)
            tot_len = len(tot_dose_axis_grid)
            X_grid =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis_grid, tot_dose_axis_grid**2]).T

        elif num_regressors == 3:
            tot_len = len(np.ravel(pooled_SC_ctrl))
            X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len),np.repeat(0,tot_len),
                               np.repeat(0,tot_len)]).T#design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, num_regressors, 0,0,0)

            tot_len = len(tot_dose_axis_open)
            if peak_dist_reg == False:
                X_open =  np.array([np.repeat(1,tot_len),
                                    tot_dose_axis_open, tot_dose_axis_open**2,
                                    np.repeat(1,tot_len)]).T
                # X_open =  np.array([np.repeat(1,tot_len),
                #                     tot_dose_axis_open, tot_dose_axis_open**2,
                #                     np.repeat(0,tot_len)]).T
                tot_len = len(tot_dose_axis_grid)
                X_grid =  np.array([np.repeat(1,tot_len),
                                    tot_dose_axis_grid, tot_dose_axis_grid**2,
                                    np.repeat(peak_area_ratio,tot_len)]).T
            else:
                X_open =  np.array([np.repeat(1,tot_len),
                                    tot_dose_axis_open, tot_dose_axis_open**2,
                                    np.repeat(0,tot_len)]).T#design_matrix(len(SC_open),tot_dose_axis_open,num_regressors,1,0,0)
                tot_len = len(tot_dose_axis_grid)
                X_grid =  np.array([np.repeat(1,tot_len),
                                    tot_dose_axis_grid, tot_dose_axis_grid**2,
                                    peak_dist]).T
        elif num_regressors == 4:
            tot_len = len(np.ravel(pooled_SC_ctrl))
            X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len),np.repeat(0,tot_len),
                               np.repeat(0,tot_len),np.repeat(0,tot_len)]).T#design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, num_regressors, 0,0,0)

            tot_len = len(tot_dose_axis_open)
            X_open =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis_open, tot_dose_axis_open**2,
                                np.repeat(1,tot_len),np.repeat(0,tot_len)]).T

            tot_len = len(tot_dose_axis_grid)
            X_grid =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis_grid, tot_dose_axis_grid**2,
                                np.repeat(peak_area_ratio,tot_len),peak_dist]).T                                                           #design_matrix(len(SC_grid),tot_dose_axis_grid, num_regressors, peak_area_ratio, valley_area_ratio, peak_dist)

        print("design matrix shape")
        print(X_ctrl.shape,X_open.shape,X_grid.shape)

        #print(X_grid[:10])
        #print(np.tile(np.ravel(peak_dist),len(dose)*pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1]))


        #X_open = np.vstack((X_ctrl,X_open))
        #X_grid = np.vstack((X_ctrl,X_grid))
        #SC_open = np.concatenate((np.ravel(pooled_SC_ctrl), SC_open))
        #SC_grid = np.concatenate((np.ravel(pooled_SC_ctrl), SC_grid))
        X_ctrl_train,X_ctrl_test,SC_ctrl_train,SC_ctrl_test = train_test_split(X_ctrl,np.ravel(pooled_SC_ctrl),test_size = 0.2)
        X_open_train,X_open_test,SC_open_train,SC_open_test = train_test_split(X_open,SC_open,test_size = 0.2)
        X_grid_train,X_grid_test, SC_grid_train,SC_grid_test = train_test_split(X_grid,SC_grid, test_size = 0.2)

        SC_ctrl_len = len(SC_ctrl_train)
        SC_open_len = len(SC_open_train)
        SC_grid_len = len(SC_grid_train)


        if grid_dots == True:

            SC_train = np.concatenate((SC_ctrl_train, SC_open_train, SC_grid_train, SC_grid_dots_train))
            X_train = np.vstack((X_ctrl_train,X_open_train,X_grid_train, X_grid_dots_train))

            SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test, SC_grid_dots_test))
            X_test = np.vstack((X_ctrl_test,X_open_test,X_grid_test, X_grid_dots_test))
            SC_len = [0,
                     SC_ctrl_len,
                     SC_ctrl_len +  SC_open_len,
                     SC_ctrl_len + SC_open_len + SC_grid_len,
                     SC_ctrl_len + SC_open_len + SC_grid_len + SC_grid_dots_len]
            legend = ["Ctrl", "Open", "GRID Stripes", "GRID Dots"]


        else:

            SC_train = np.concatenate((SC_ctrl_train,SC_open_train,SC_grid_train))
            X_train = np.vstack((X_ctrl_train, X_open_train, X_grid_train))
            # SC_train = np.concatenate((SC_ctrl_train,SC_open_train))
            # X_train = np.vstack((X_ctrl_train, X_open_train))

            SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test))
            X_test = np.vstack((X_ctrl_test, X_open_test, X_grid_test))
            SC_len = [0,
                     SC_ctrl_len,
                     SC_ctrl_len + SC_open_len,
                     SC_ctrl_len + SC_open_len + SC_grid_len]
            # SC_len = [0,
            #          SC_ctrl_len,
            #          SC_ctrl_len + SC_open_len]
            legend = ["Ctrl", "Open", "GRID Stripes"]
            #legend = ["Ctrl", "Open"]


        # mean_obs_SC_train = mean_survival(X_train,SC_train)

        """Regressor correlation"""

        print(X_train.shape)

        """X_train_df = pd.DataFrame(X_train[:,1:], columns = ["D", r"$D^2$", "PA","PD"])

        axes = scatter_matrix(X_train_df, alpha=0.5, diagonal='kde')
        corr = X_train_df.corr().to_numpy()

        corr_df = pd.DataFrame(corr, index = ["D","D2","PA","PD"],columns = ["D","D2","PA","PD"])
        corr_df.to_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\correlation_matrix_{}mmkernel.csv".format(kernel_size_mm[i]))
        print(corr)
        for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
           axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')

        g = sb.PairGrid(X_train_df)
        g.map_diag(sb.kdeplot, fill = True, alpha = 0.7, color = "tab:blue")#, label = "KDE")
        g.map_offdiag(plt.scatter, s = 5,color = "tab:orange")#, label = "Scatter")
        g.map_offdiag(corrfunc)
        g.fig.suptitle("Diag: KDE   off diag: Scatter")
        g.fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\correlation_plot_{}regressors_{}kernel.png".format(num_regressors, kernel_size_mm[i]), dpi = 300)
        g.map_offdiag(sb.scatterplot)
        plt.close()"""

        """Running the Poisson regression"""

        if peak_dist_reg and num_regressors == 3:
            if grid_dots:
                poisson_results = 'GLM_results_{}mm_OPEN&STRIPES&DOTS_{}regressors_peak_dist.tex'.format(kernel_size_mm[i], num_regressors)
            else:
                poisson_results = 'GLM_results_{}mm_OPEN&STRIPES_{}regressors_peak_dist.tex'.format(kernel_size_mm[i], num_regressors)
        elif not peak_dist_reg and num_regressors ==3:
            if grid_dots:
                poisson_results = 'GLM_results_{}mm_OPEN&STRIPES&DOTS_{}regressors_peak_area.tex'.format(kernel_size_mm[i], num_regressors)
            else:
                poisson_results = 'GLM_results_{}mm_OPEN&STRIPES_{}regressors_peak_area.tex'.format(kernel_size_mm[i], num_regressors)
        else:
            if grid_dots:
                poisson_results = 'GLM_results_{}mm_OPEN&STRIPES&DOTS_{}regressors.tex'.format(kernel_size_mm[i], num_regressors)
            else:
                poisson_results = 'GLM_results_{}mm_OPEN&STRIPES_{}regressors.tex'.format(kernel_size_mm[i], num_regressors)


        #poisson_results = 'GLM_results_{}mm_OPEN_{}regressors_peak_area.tex'.format(kernel_size_mm[i], num_regressors)
        model, mean_pred_SC_train, summary = poisson_regression(SC_train,X_train,num_regressors,
                                  r"GRID: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size_mm[i], kernel_size_mm[i]),
                                  'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\regression results\\' + poisson_results,
                                  legend, SC_len, kernel_size_mm[i], False)

        # if peak_dist_reg and num_regressors == 3:
        #     plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_OPEN&STRIPES&DOTS_{}mm_{}regressors_peak_dist.png".format(kernel_size_mm[i], num_regressors), dpi = 300)
        # elif not peak_dist_reg and num_regressors ==3:
        #     plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_OPEN&STRIPES&DOTS_{}mm_{}regressors_peak_area.png".format(kernel_size_mm[i], num_regressors), dpi = 300)
        # else:
        #     plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_OPEN&STRIPES&DOTS_{}mm_{}regressors.png".format(kernel_size_mm[i], num_regressors), dpi = 300)

        # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_OPEN_{}mm_{}regressors_peak_area.png".format(kernel_size_mm[i], num_regressors), dpi = 300)
        plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Foredrag\\quadrat_raw_data_2mm.png",dpi = 300)
        plt.show()

        """final error evaluation"""
        error_eval3 = True
        if error_eval3:
            """
            Comparing observed survival from striped GRID irradiated cells with survival
            predicted by fitting the Poisson regression with dose and dose squared
            as explanatory variables.
            """
            if num_regressors == 2:
                """
                dose5Gy_grid
                SC_grid_5Gy

                dose5Gy_open
                SC_open_5Gy
                Comparing OPEN field 2 regressors to striped GRID observed
                This comparison can only be done with 2 regressors
                """
                d70 = np.max(dose5Gy_grid)*0.7  #dividing by 0.8 because OD is opposite to dose
                # d85_idx  = np.abs(dose_map-d85).argmin()
                d30 = np.min(dose5Gy_grid)*1.3

                plt.title("Distribution of doses in 5 Gy GRID irradiated dosemap")
                n, bins, patches = plt.hist(dose5Gy_grid.ravel(), density = True,
                                            facecolor='magenta', alpha = 0.8, edgecolor = "black", align = "left")
                peak = dose5Gy_grid[dose5Gy_grid > 2.5]
                valley = dose5Gy_grid[dose5Gy_grid < 2.5]

                print(np.std(peak),np.std(valley))
                plt.show()

                peak_idx = dose5Gy_grid > d70
                valley_idx = dose5Gy_grid < d30

                print(np.std(dose5Gy_grid[peak_idx]), np.std(dose5Gy_grid[valley_idx]))


                peak_survival =SC_grid_5Gy[:,:,peak_idx]
                valley_survival = SC_grid_5Gy[:,:,valley_idx]
                peak_dose = np.repeat(dose5Gy_grid[peak_idx], peak_survival.shape[0]*peak_survival.shape[1])
                valley_dose = np.repeat(dose5Gy_grid[valley_idx], peak_survival.shape[0]*peak_survival.shape[1])

                peak_survival = peak_survival.flatten()
                valley_survival = valley_survival.flatten()

                np.unique(np.round(peak_dose*2)/2)

                new_dose_axis = np.linspace(0,np.max(X_open[:,1]),10000)
                X_interp = np.array([np.repeat(1,len(new_dose_axis)),
                                    new_dose_axis, new_dose_axis**2]).T

                fit_results = model.get_prediction(X_interp) #need this for confidence interval
                predictions = model.predict(X_interp)

                frame = fit_results.summary_frame(alpha=0.05)
                #print(frame)
                #predicted = model.get_prediction(X_interp).summary_frame()["mean"]

                # print(predictions[np.argmin(new_dose_axis - np.mean(peak_dose))])

                RPD_peak = 2*np.abs(np.mean(peak_survival) - predictions[np.argmin(new_dose_axis - np.mean(peak_dose))])/(np.mean(peak_survival) + predictions[np.argmin(new_dose_axis - np.mean(peak_dose))])
                RPD_valley = 2*np.abs(np.mean(valley_survival) - predictions[np.argmin(new_dose_axis - np.mean(valley_dose))])/(np.mean(valley_survival) + predictions[np.argmin(new_dose_axis - np.mean(valley_dose))])

                fig,ax = plt.subplots(figsize = (9,5))

                ax.plot(new_dose_axis,predictions, "--", color = "navy",label = "Predicted OPEN")
                ax.errorbar(np.mean(peak_dose), np.mean(peak_survival),color = "r",xerr = np.std(peak_dose), yerr = np.std(peak_survival)/np.sqrt(len(peak_survival)), label = "Observed Peak")
                ax.errorbar(np.mean(valley_dose), np.mean(valley_survival),color = "g",xerr = np.std(valley_dose), yerr = np.std(valley_survival)/np.sqrt(len(valley_survival)), label = "Observed Valley")
                ax.fill_between(new_dose_axis, frame.mean_ci_lower,frame.mean_ci_upper, alpha = 0.7, color = "cyan", label = "95% C.I.")
                ax.set_xlabel("Dose [Gy]")
                ax.set_ylabel("SC")
                ax.set_title(f"Predicted from OPEN VS Observed Peak and Valley \n {kernel_size_mm[i]} mm quadrat")
                ax.legend()

                #ax2 = ax.twinx()
                #ax2.plot(np.mean(peak_dose), RPD_peak, "o", color = "r", label = "RPD Peak")
                #ax2.plot(np.mean(valley_dose), RPD_valley, "o", color = "g", label = "RPD Valley")
                #ax2.set_ylabel("RPD")
                #ax2.legend(loc = "lower right")
                #ax2.grid(False)
                fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\peak_valley_vs_predicted_2.png", dpi = 300)
                plt.show()

            # predicted = model.get_prediction(X_open[1]).summary_frame()["mean"]

            # pred_isodose = predicted[]




            method_labels = ["Ctrl", "OPEN", "GRID Stripes", "GRID Dots"]
            test_data = {"Ctrl":{"SC":SC_ctrl_test,"X":X_ctrl_test}, "OPEN":{"SC":SC_open_test, "X":X_open_test},
                       "GRID Stripes":{"SC":SC_grid_test,"X":X_grid_test}, "GRID Dots":{"SC":SC_grid_dots_test, "X":X_grid_dots_test}}
            train_data = {"Ctrl":{"SC":SC_ctrl_train,"X":X_ctrl_train}, "OPEN":{"SC":SC_open_train, "X":X_open_train},
                       "GRID Stripes":{"SC":SC_grid_train,"X":X_grid_train}, "GRID Dots":{"SC":SC_grid_dots_train, "X":X_grid_dots_train}}


            """Finding mean survival dose"""
            fig, ax = plt.subplots(ncols = 3,figsize = (10,5),sharex = True, sharey = True)
            plt.suptitle("Mean SC  {} regressors {} mm Quadrats".format(num_regressors, kernel_size_mm[i]))
            ax = ax.flatten()
            for idx, method in enumerate(test_data):
                print("Irradiation Configuration")
                print(method)
                X_train = train_data[method]["X"]
                # X_test = test_data[method]["X"]
                SC_train = train_data[method]["SC"]
                # SC_test = test_data[method]["SC"]
                # print(model.get_prediction(test_data[method]["X"]).summary_frame())
                # print(model.get_prediction(train_data[method]["X"]).summary_frame())

                #predicted_test = model.get_prediction(X_test).summary_frame()["mean"]
                predicted_train = model.get_prediction(X_train).summary_frame()["mean"]

                """
                mean survival bins the dose into dose categories. Then it finds the mean survival within
                these categories
                """

                mean_obs_SC_train, std_obs_SC_train, dose_categories_obs_train, dose_std_obs_train = mean_survival(X_train,SC_train, 1, method) #mean observed surviving colonies train
                # mean_obs_SC_test, std_obs_SC_test,dose_categories_obs_test, dose_std_obs_test = mean_survival(X_test,SC_test,1) #mean observed surviving colonies test
                mean_pred_SC_train, std_pred_SC_train,dose_categories_pred_train, dose_std_pred_train = mean_survival(X_train,predicted_train,1, method)
                # mean_pred_SC_test, std_pred_SC_test,dose_categories_pred_test, dose_std_pred_test = mean_survival(X_test, predicted_test,1) #mean predicted surviving colonies train


                # dose_axis_test = np.linspace(np.min(X_test[:,1]),np.max(X_test[:,1]),len(mean_obs_SC_test))
                # dose_axis_train = np.linspace(np.min(X_test[:,1]),np.max(X_train[:,1]), len(mean_obs_SC_train)) #mean observed surviving colonies train was found in the Poisson regression

                if method == "Ctrl":
                    ctrl_obs_SC_train = mean_obs_SC_train
                    ctrl_pred_SC_train = mean_pred_SC_train
                    ctrl_obs_std = std_obs_SC_train
                    ctrl_pred_std = std_pred_SC_train
                    ctrl_dose_std = dose_std_obs_train

                    ctrl_dose = [0]

                else:
                    obs_axis = np.append(ctrl_obs_SC_train,mean_obs_SC_train)
                    pred_axis = np.append(ctrl_pred_SC_train,mean_pred_SC_train)
                    std_obs_axis = np.append(ctrl_obs_std, std_obs_SC_train)
                    #std_pred_axis = np.append(ctrl_pred_std, std_pred_SC_train)
                    dose_axis = np.append(ctrl_dose, dose_categories_obs_train)
                    std_axis = np.append(ctrl_dose_std,dose_std_obs_train)

                    #print(ctrl_obs_SC_train)
                    #print(dose_axis.shape, obs_axis.shape)
                    ax[idx-1].set_xlabel("Dose [Gy]")
                    ax[idx-1].set_ylabel("SC")
                    ax[idx-1].errorbar(dose_axis, obs_axis,xerr = std_axis,yerr = std_obs_axis, fmt = "o", label = "Observed SC {}".format(method), color = colors[idx-1], markersize = 5)
                    ax[idx-1].plot(dose_axis, pred_axis,"--", label = "Predicted SC {}".format(method), color = colors[idx-1])
                    ax[idx-1].legend()


            plt.tight_layout()
            # fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\ObsVSPred_{}mm_survival_{}regressors_new.png".format(kernel_size_mm[i], num_regressors), dpi = 300)
            plt.show()

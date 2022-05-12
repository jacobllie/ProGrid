from survival_analysis4 import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f, ttest_ind, chi2
from kernel_density_estimation import kde
import seaborn as sb
from scipy import stats, optimize
import seaborn as sb
from poisson import poisson
from scipy.interpolate import interp1d
from utils import K_means, logLQ, fit, poisson_regression, data_stacking,design_matrix, data_stacking_2, mean_survival, logLQ, LQres, dose_profile2
import sys
from plotting_functions_survival import survival_histogram, survival_curve_grid, survival_curve_open, pred_vs_true_SC
import cv2
from playsound import playsound
from sklearn.model_selection import train_test_split
import pickle
import skimage.transform as tf
import string




sound_path = "C:\\Users\\jacob\\OneDrive\\Documents\\livet\\veldig viktig\\"
sounds = ["Ah Shit Here We Go Again - GTA Sound Effect (HD).mp3",
         "Anakin Skywalker - Are You An Angel.mp3","Get-in-there-Lewis-F1-Mercedes-AMG-Sound-Effect.wav",
          "he-need-some-milk-sound-effect.wav",
         "MOM GET THE CAMERA Sound Effect.mp3","My-Name-is-Jeff-Sound-Effect-_HD_.wav",
         "Nice (HD) Sound effects.mp3","Number-15_-Burger-king-foot-lettuce-Sound-Effect.wav",
         "oh_my_god_he_on_x_games_mode_sound_effect_hd_4036319132337496168.mp3",
         "Ok-Sound-Effect.wav","PIZZA TIME! Sound Effect (Peter Parker).mp3","WHY-ARE-YOU-GAY-SOUND-EFFECT.wav", "OKLetsGo.mp3",
         "Adam vine.wav","Fresh Avocado Vine.wav", "I Can't Believe You've Done This.wav",
         "I Don't Have Friends, I Got Family.wav","Just Do It - Sound Effect [Perfect Cut].wav","Wait A Minute, Who Are You Meme.wav",
         "WTF Richard.wav", "you almost made me drop my croissant vine.wav","Martin, Thea og Nikolai, hele klippet.wav"]
weights = np.zeros(len(sounds))
weights[-1] = 1
# playsound(sound_path + np.random.choice(sounds,p = weights))

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
# kernel_size = 3.9 #mm
# kernel_size = int(kernel_size*47) #pixels/mm
#cropping_limits = [250,2200,300,1750]

#cropping_limits = [225,2200,300,1750] #this will not affect 1D analysis
#cropping_limits = [225,2200,350,1950]
# cropping_limits = [210,2100,405,1900]

kernel_size_mm = [0.5,1,2,3]
# kernel_size_mm = [0.5,1,2,3,4]
kernel_size_p = [int(i*47) for i in kernel_size_mm]
# cropping_limits_2D = [100,2000,100,2050] #absolute max area
#cropping_limits_2D = [100,2000,350,1850]
cropping_limits_2D = [200,2000,350,1850]

peak_dist_reg = False
num_regressors = 2

plt.style.use("seaborn")
"""
18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
We therefore combine these data to find alpha beta for open field irradiation.
"""

plt.imshow(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Control\\A549-1811-K1-SegMask.csv"))
plt.close()

header = pd.MultiIndex.from_product([['2 Gy','5 Gy', '10 Gy'],
                                    ['Peak','Valley']])

df_pois = pd.DataFrame(columns = header)
print(df_pois)

dose_var = np.zeros(len(kernel_size_mm))
excessive_zeros = np.zeros(len(kernel_size_mm))
rel_diff = np.zeros((len(kernel_size_mm), 2))
poisson_test = False

"""
Finding the number of counted colonies for control (0Gy) and open field
experiments (2Gy and 5Gy)
"""
for i in range(len(kernel_size_mm)):
    print("{}/{}".format(i+1,len(kernel_size_mm)))
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


        #checking for excessive zeros
        excessive_zeros[i] = len(pooled_SC_ctrl[pooled_SC_ctrl < 1])
        print(excessive_zeros[i])
        # survival_histogram(None, None, pooled_SC_ctrl, mode = "Control")

        #pooled_SC_ctrl = np.reshape(pooled_SC_ctrl[:,0,:], (pooled_SC_ctrl.shape[0],
                              #pooled_SC_ctrl.shape[2], pooled_SC_ctrl.shape[3],pooled_SC_ctrl.shape[4]))
        #mean_SC_ctrl = np.mean(pooled_SC_ctrl)
        #extracting flask template and cropping away edges



        #GREY_chan_cropped = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes\\EBT3_Stripes_310821_Xray220kV_5Gy1_001.tif",-1)[10:-10,10:-10,0]
        #GREY_chan_cropped = tf.rescale(GREY_chan_cropped,4)



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


        """
        we are here
        """

        _,dose_map = survival_grid.Quadrat("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\2D survival\\grid_survival5Gy_300821_stripes_{}mm.png".format(kernel_size_mm[i]))


        dose2Gy_grid, SC_grid_2Gy = survival_grid.SC(2)
        dose5Gy_grid, SC_grid_5Gy = survival_grid.SC(5)
        dose10Gy_grid, SC_grid_10Gy = survival_grid.SC(10)



        print(np.var(dose5Gy_grid))
        dose_var[i] = np.var(dose5Gy_grid)
        print("variance in pooled dose")
        print(dose_var[i])
        print(dose5Gy_grid.shape)
        print(SC_grid_5Gy.shape)
        print(dose_map.shape)

        """
        Plot histogram of survival
        Separate into dose categories
        """

        if poisson_test:


            rel_diff2 = survival_histogram(dose_map*2/5,dose2Gy_grid, SC_grid_2Gy, 2, kernel_size_mm[i])
            rel_diff[i] = survival_histogram(dose_map, dose5Gy_grid, SC_grid_5Gy, 5, kernel_size_mm[i])
            rel_diff10 = survival_histogram(dose_map*10/5, dose10Gy_grid, SC_grid_10Gy, 10, kernel_size_mm[i])

            print(rel_diff10)

            x = np.zeros((6,6))
            x[0] = np.ravel([rel_diff2, rel_diff[i], rel_diff10])
            tmp_df = pd.DataFrame(x, columns = header) #with three headers and two subheader = 6 , we need 6 rows
            tmp_df = tmp_df.drop([len(tmp_df)-1, len(tmp_df)-2,len(tmp_df)-3,len(tmp_df)-4,len(tmp_df)-5])
            df_pois = pd.concat([df_pois,tmp_df])


            if i == len(kernel_size_mm) - 1:
                df_pois = df_pois.set_index(pd.Index(kernel_size_mm, name = "Kernel Size [mm]"))
                #df_pois.to_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\poisson_test.csv",)

                fig, ax = plt.subplots(nrows = 1,ncols = 2)
                ax1 = ax[0]
                color = 'tab:red'
                ax1.set_xlabel('kernel size [mm]')
                ax1.set_ylabel(string.capwords("# 0's"), color=color)
                ax1.plot(kernel_size_mm, excessive_zeros, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax2.set_ylabel('variance', color=color)  # we already handled the x-label with ax1
                ax2.plot(kernel_size_mm, dose_var, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                ax3 = ax[1]
                color = 'tab:red'
                ax3.set_xlabel('kernel size [mm]')
                ax3.set_ylabel(string.capwords("Peak relative difference mean vs variance"), color=color)
                ax3.plot(kernel_size_mm, rel_diff[:,0], color=color)
                ax3.tick_params(axis='y', labelcolor=color)

                ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax4.set_ylabel('Valley relative difference mean vs variance', color=color)  # we already handled the x-label with ax1
                ax4.plot(kernel_size_mm, rel_diff[:,1], color=color)
                ax4.tick_params(axis='y', labelcolor=color)
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                #fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\kernel_size_eval.png", pad_inches = 0.2, bbox_inches = "tight", dpi = 1200)
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

    """
    tmp_dose2Gy, tmp_dose5Gy, tmp_dose10Gy, SC_grid_2Gy, SC_grid_5Gy, SC_grid_10Gy, pooled_SC_ctrl, SC_open_2Gy, SC_open_5Gy
    """

    """
    First analysis method. Calculating peak and valley ratio from cell flask. Not including
    square size.
    """

    method1 = False
    if method1 == True:
        print("Running method 1")
        """tot_irradiatet_area = 24.505*100 #mm^2
        peak_area = 3*215+170.75 #3 full peaks, 1 trapezoidal peak
        valley_area_ratio = (tot_irradiatet_area-peak_area)/tot_irradiatet_area
        peak_area_ratio = peak_area/tot_irradiatet_area"""

        tot_analysis_area = (cropping_limits[1] - cropping_limits[0])*(cropping_limits[3] - cropping_limits[2])/47**2 #mm^2 47pixels per mm for 1200 dpi
        peak_area = 393.32 #mm^2
        valley_area_ratio = (tot_analysis_area-peak_area)/tot_analysis_area
        peak_area_ratio = peak_area/tot_analysis_area



        """
        Now we stack all data togheter and perform poisson regression.
        We split the data into training and testing
        """



        SC_open, tot_dose_axis_open = data_stacking_2(False, SC_open_2Gy,
                                                    SC_open_5Gy, dose2Gy_open,
                                                    dose5Gy_open)
        SC_grid, tot_dose_axis_grid = data_stacking_2(True, SC_grid_2Gy,
                                                      SC_grid_5Gy, SC_grid_10Gy,
                                                      dose2Gy_grid, dose5Gy_grid, dose10Gy_grid)
        """
        Adding the control survival and doses (0Gy) to open. Then make individual design matrix with different G factors
        """
        tmp = np.repeat(dose2Gy_grid*0,pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1])
        X_ctrl = design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, 4, 0, 0)
        X_open = design_matrix(len(SC_open),tot_dose_axis_open,4, 1, 0)
        X_grid = design_matrix(len(SC_grid),tot_dose_axis_grid, 4, peak_area_ratio, valley_area_ratio)


        grid_dots = True
        if grid_dots == True:
            X_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\GRID_Dots_X_{}regressors.npy".format(num_regressors))
            SC_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\GRID_Dots_SC_{}regressors.npy".format(num_regressors))
            X_grid_dots_train, X_grid_dots_test, SC_grid_dots_train, SC_grid_dots_test = train_test_split(X_grid_dots,SC_grid_dots, test_size = 0.2)



        #X_open = np.vstack((X_ctrl,X_open))
        #X_grid = np.vstack((X_ctrl,X_grid))
        #SC_open = np.concatenate((np.ravel(pooled_SC_ctrl), SC_open))
        #SC_grid = np.concatenate((np.ravel(pooled_SC_ctrl), SC_grid))
        X_ctrl_train,X_ctrl_test,SC_ctrl_train,SC_ctrl_test = train_test_split(X_ctrl,np.ravel(pooled_SC_ctrl),test_size = 0.2)
        X_open_train,X_open_test,SC_open_train,SC_open_test = train_test_split(X_open,SC_open,test_size = 0.2)
        X_grid_train,X_grid_test, SC_grid_train,SC_grid_test = train_test_split(X_grid,SC_grid, test_size = 0.2)

        if grid_dots == True:
            SC_train = np.concatenate((SC_ctrl_train, SC_open_train, SC_grid_train, SC_grid_dots_train))
            X_train = np.vstack((X_ctrl_train,X_open_train,X_grid_train, X_grid_dots_train))

            SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test, SC_grid_dots_test))
            X_test = np.vstack((X_ctrl_test,X_open_test,X_grid_test, X_grid_dots_test))

        else:
            SC_train = np.concatenate((SC_ctrl_train, SC_open_train, SC_grid_train))
            X_train = np.vstack((X_ctrl_train,X_open_train,X_grid_train))

            SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test))
            X_test = np.vstack((X_ctrl_test,X_open_test,X_grid_test))

        mean_obs_SC_train = mean_survival(X_train,SC_train)


        model, mean_pred_SC_train, summary = poisson_regression(SC_train,X_train,5,
                                  r"GRID&OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                                  'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                                  True)

        plt.show()

        mean_obs_SC_test = mean_survival(X_test,SC_test)

        print(mean_obs_SC_test.shape)
        predicted_tmp = model.get_prediction(X_test).summary_frame()["mean"]
        mean_pred_SC_test = mean_survival(X_test, predicted_tmp)


        dose_axis_train = np.linspace(0,np.max(X_train[:,1]), len(mean_obs_SC_train))
        dose_axis_test = np.linspace(0,np.max(X_test[:,1]),len(mean_obs_SC_test))

        plt.suptitle("Mean SC observed vs predicted OPEN&GRID")
        plt.subplot(121)
        pred_vs_true_SC(mean_obs_SC_train, mean_pred_SC_train, dose_axis_train, "Train")
        plt.subplot(122)
        pred_vs_true_SC(mean_obs_SC_test, mean_pred_SC_test, dose_axis_test, "Test")
        # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_train_test_39mm_survival.png", dpi = 1200)

        plt.show()

        mean_obs_SC_open = mean_survival(X_open_test, SC_open_test)
        dose_axis_open = np.linspace(0,np.max(X_open_test[:,1]),len(mean_obs_SC_open))
        mean_obs_SC_open = mean_survival(X_open_test,SC_open_test)
        predicted_tmp = model.get_prediction(X_open_test).summary_frame()["mean"]
        mean_pred_SC_open = mean_survival(X_open_test,predicted_tmp)

        mean_obs_SC_grid = mean_survival(X_grid_test, SC_grid_test)
        dose_axis_grid = np.linspace(0,np.max(X_grid_test[:,1]),len(mean_obs_SC_grid))
        mean_obs_SC_grid = mean_survival(X_grid_test,SC_grid_test)
        predicted_tmp = model.get_prediction(X_grid_test).summary_frame()["mean"]
        mean_pred_SC_grid = mean_survival(X_grid_test,predicted_tmp)



        plt.suptitle("Mean SC observed vs predicted")
        plt.subplot(121)
        pred_vs_true_SC(mean_obs_SC_open, mean_pred_SC_open, dose_axis_open, "OPEN")
        plt.subplot(122)
        pred_vs_true_SC(mean_obs_SC_grid, mean_pred_SC_grid, dose_axis_grid, "GRID")
        # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_open_grid_39mm_survival.png", dpi = 1200)
        plt.show()


        """
        No train test split
        """

        SC = np.concatenate((np.ravel(pooled_SC_ctrl), SC_open, SC_grid))
        X = np.vstack((X_ctrl,X_open,X_grid))


        true_SC = mean_survival(X,SC)
        print("True average survival")
        model, predicted_SC, summary = poisson_regression(SC,X,4,
                                  r"GRID&OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                                  'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                                  False)
        plt.close()

        """
        Finding MSE between predicted mean survival vs true mean survival for grid and open stacked
        """

        print(np.shape(true_SC), np.shape(predicted_SC))
        MSE = 1/len(true_SC)*np.sum(np.subtract(true_SC,predicted_SC)**2)
        print(MSE)

        """
        Finding MSE between predicted mean survival  vs true mean survival for grid
        """


        true_SC_grid = mean_survival(X_grid,SC_grid)
        predicted_tmp = model.get_prediction(X_grid).summary_frame()["mean"]

        predicted_SC_grid = mean_survival(X_grid, predicted_tmp)
        MSE = 1/len(true_SC_grid)*np.sum(np.subtract(true_SC_grid,predicted_SC_grid)**2)

        print(MSE)

        """
        Finding MSE between predicted mean survival vs true mean survival for open
        """
        true_SC_open = mean_survival(X_open,SC_open)
        predicted_tmp = model.get_prediction(X_open).summary_frame()["mean"]

        predicted_SC_open = mean_survival(X_open, predicted_tmp)
        MSE = 1/len(true_SC_open)*np.sum(np.subtract(true_SC_open,predicted_SC_open)**2)

        print(MSE)


        pred_vs_true_SC(true_SC,predicted_SC,true_SC_grid,predicted_SC_grid,true_SC_open,predicted_SC_open)
    method2 = False
    if method2 == True:
        """tot_analysis_area = (cropping_limits[1] - cropping_limits[0])*(cropping_limits[3] - cropping_limits[2])/47**2 #mm^2 47pixels per mm for 1200 dpi
        peak_area = 393.32 #mm^2
        valley_area_ratio = (tot_analysis_area-peak_area)/tot_analysis_area
        peak_area_ratio = peak_area/tot_analysis_area"""

        # print(tot_analysis_area)
        # print(valley_area_ratio, peak_area_ratio)

        """
        Now we stack all data togheter and perform poisson regression.
        We split the data into training and testing
        """



        SC_open, tot_dose_axis_open = data_stacking_2(False, SC_open_2Gy,
                                                    SC_open_5Gy, dose2Gy_open,
                                                    dose5Gy_open)
        SC_grid, tot_dose_axis_grid = data_stacking_2(True, SC_grid_2Gy,
                                                      SC_grid_5Gy, SC_grid_10Gy,
                                                      dose2Gy_grid, dose5Gy_grid, dose10Gy_grid)
        """
        Adding the control survival and doses (0Gy) to open. Then make individual design matrix with different G factors
        """
        tmp = np.repeat(dose2Gy_grid*0,pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1])
        X_ctrl = design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, 5, 0, 0, kernel_size)
        X_open = design_matrix(len(SC_open),tot_dose_axis_open,5, 1, 0, kernel_size)
        X_grid = design_matrix(len(SC_grid),tot_dose_axis_grid, 5, peak_area_ratio, valley_area_ratio, kernel_size)

        X_ctrl_train,X_ctrl_test,SC_ctrl_train,SC_ctrl_test = train_test_split(X_ctrl,np.ravel(pooled_SC_ctrl),test_size = 0.2)
        X_open_train,X_open_test,SC_open_train,SC_open_test = train_test_split(X_open,SC_open,test_size = 0.2)
        X_grid_train,X_grid_test, SC_grid_train,SC_grid_test = train_test_split(X_grid,SC_grid, test_size = 0.2)
        SC_train = np.concatenate((SC_ctrl_train, SC_open_train, SC_grid_train))
        X_train = np.vstack((X_ctrl_train,X_open_train,X_grid_train))

        SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test))
        X_test = np.vstack((X_ctrl_test,X_open_test,X_grid_test))
        mean_obs_SC_train = mean_survival(X_train,SC_train)
        model, mean_pred_SC_train, summary = poisson_regression(SC_train,X_train,5,
                                                       r"GRID&OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                                                       'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                                                       False)

        plt.show()
    method3 = True
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
            pass
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
                                    np.repeat(1,tot_len)]).T#design_matrix(len(SC_open),tot_dose_axis_open,num_regressors,1,0,0)
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
                                np.repeat(1,tot_len),np.repeat(0,tot_len)]).T#design_matrix(len(SC_open),tot_dose_axis_open,num_regressors,1,0,0)
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

            #SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test, SC_grid_dots_test))
            #X_test = np.vstack((X_ctrl_test,X_open_test,X_grid_test, X_grid_dots_test))
            SC_len = [0,
                     SC_ctrl_len,
                     SC_ctrl_len +  SC_open_len,
                     SC_ctrl_len + SC_open_len + SC_grid_len,
                     SC_ctrl_len + SC_open_len + SC_grid_len + SC_grid_dots_len]
            legend = ["Ctrl", "Open", "GRID Stripes", "GRID Dots"]


        else:

            SC_train = np.concatenate((SC_ctrl_train,SC_open_train,SC_grid_train))
            X_train = np.vstack((X_ctrl_train, X_open_train, X_grid_train))
            # SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test))
            # X_test = np.vstack((X_ctrl_test, X_open_test, X_grid_test))
            SC_len = [0,
                     SC_ctrl_len,
                     SC_ctrl_len + SC_open_len,
                     SC_ctrl_len + SC_open_len + SC_grid_len]
            legend = ["Ctrl", "Open", "GRID Stripes"]


        mean_obs_SC_train = mean_survival(X_train,SC_train)

        poisson_results = 'GLM_results_{}mm_OPEN&STRIPES&DOTS_{}regressors.tex'.format(kernel_size_mm[i], num_regressors)
        model, mean_pred_SC_train, summary = poisson_regression(SC_train,X_train,num_regressors,
                                  r"GRID: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size_mm[i], kernel_size_mm[i]),
                                  'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\regression results\\' + poisson_results,
                                  legend, SC_len, kernel_size_mm[i], True)
        # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_OPEN&STRIPES&DOTS_{}mm_{}regressors.png".format(kernel_size_mm[i], num_regressors), dpi = 1200)
        plt.close()



        error_eval2 = True

        if error_eval2:
            """
            We test how well the model estimates the mean
            SC_ctrl_test, SC_open_test, SC_grid_test, SC_grid_dots_test
            X_ctrl_test,X_open_test,X_grid_test, X_grid_dots_test
            """
            method_labels = ["Ctrl", "OPEN", "GRID Stripes", "GRID Dots"]
            test_data = {"Ctrl":{"SC":SC_ctrl_test,"X":X_ctrl_test}, "OPEN":{"SC":SC_open_test, "X":X_open_test},
                       "GRID Stripes":{"SC":SC_grid_test,"X":X_grid_test}, "GRID Dots":{"SC":SC_grid_dots_test, "X":X_grid_dots_test}}
            MSE = np.zeros((2,len(method_labels)))
            fig,ax = plt.subplots(ncols = 2, figsize = (20,20))

            for idx, method in enumerate(test_data):
                predicted = model.get_prediction(test_data[method]["X"]).summary_frame()["mean"]
                true = test_data[method]["SC"]
                err = (predicted-true)**2

                MSE[0,idx] = np.mean(err)
                MSE[1,idx] = np.std(err)/np.sqrt(len(err))
                ax[0].set_ylabel(r"$ (pred_i - true_i)^2 $", fontsize = 15, rotation = 90)
                ax[0].plot(test_data[method]["X"][:,1], err, "o", markersize = 5, label = method_labels[idx]) #plotting dose vs quadrat of error
                ax[0].legend()
                #mean_pred_SC_test = mean_survival(X_test, predicted_tmp)
            ax[0].set_title("MSE w. Standard error")
            ax[1].set_title("Squared error vs dose")
            ax[1].set_xlabel("Dose [Gy]", fontsize = 10)
            ax[1].set_ylabel(r"$\frac{1}{n} \cdot \sum_{i = 0}^{n} (pred_i - true_i)^2 $", fontsize = 12, rotation = 60)
            ax[1].bar(method_labels, MSE[0], yerr = MSE[1] , color = colors[:4], alpha = 0.6)
            # fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_OPEN&GRID&STRIPES&DOTS_{}mm_{}regressors_MSE.png".format(kernel_size_mm[i], num_regressors), dpi = 1200)

            plt.close()





        error_eval = False
        if error_eval:
            mean_obs_SC_test = mean_survival(X_test,SC_test)

            print(mean_obs_SC_test.shape)
            predicted_tmp = model.get_prediction(X_test).summary_frame()["mean"]
            mean_pred_SC_test = mean_survival(X_test, predicted_tmp)


            dose_axis_train = np.linspace(0,np.max(X_train[:,1]), len(mean_obs_SC_train))
            dose_axis_test = np.linspace(0,np.max(X_test[:,1]),len(mean_obs_SC_test))

            plt.suptitle("Mean SC observed vs predicted GRID")
            plt.subplot(121)
            pred_vs_true_SC(mean_obs_SC_train, mean_pred_SC_train, dose_axis_train, "Train")
            plt.subplot(122)
            pred_vs_true_SC(mean_obs_SC_test, mean_pred_SC_test, dose_axis_test, "Test")
            #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_train_test_{}mm_survival_{}regressors_distance.png".format(kernel_size_mm[i], num_regressors), dpi = 1200)

            plt.close()

            mean_obs_SC_open = mean_survival(X_open_test, SC_open_test)
            dose_axis_open = np.linspace(0,np.max(X_open_test[:,1]),len(mean_obs_SC_open))
            mean_obs_SC_open = mean_survival(X_open_test,SC_open_test)
            predicted_tmp = model.get_prediction(X_open_test).summary_frame()["mean"]
            mean_pred_SC_open = mean_survival(X_open_test,predicted_tmp)

            #adding dots
            if grid_dots == True:
                X_grid_test = np.vstack((X_grid_test, X_grid_dots_test))
                SC_grid_test = np.concatenate((SC_grid_test,SC_grid_dots_test))

            mean_obs_SC_grid = mean_survival(X_grid_test, SC_grid_test)
            dose_axis_grid = np.linspace(0,np.max(X_grid_test[:,1]),len(mean_obs_SC_grid))
            mean_obs_SC_grid = mean_survival(X_grid_test,SC_grid_test)
            predicted_tmp = model.get_prediction(X_grid_test).summary_frame()["mean"]
            mean_pred_SC_grid = mean_survival(X_grid_test,predicted_tmp)



            plt.suptitle("Mean SC observed vs predicted")
            plt.subplot(121)
            pred_vs_true_SC(mean_obs_SC_open, mean_pred_SC_open, dose_axis_open, "OPEN")
            plt.subplot(122)
            pred_vs_true_SC(mean_obs_SC_grid, mean_pred_SC_grid, dose_axis_grid, "GRID")
            #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_open_grid_{}mm_survival_{}regressors_distance.png".format(kernel_size_mm[i], num_regressors), dpi = 1200)
            plt.close()

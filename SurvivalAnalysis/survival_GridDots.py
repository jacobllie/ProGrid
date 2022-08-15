from survival_analysis_4 import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from utils import poisson_regression, data_stacking_2
import sys
import cv2
import pickle
from sklearn.model_selection import train_test_split



folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021"
time = ["20112019"]
mode = ["Control", "GRID Dots"]
dose = ["02", "05", "10"]
ctrl_dose = ["00"]
template_file_control =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\20112019\\Control\\A549-2011-K1-TemplateMask.csv"
template_file_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\20112019\\GRID Dots\\A549-2011-02-gridC-A-TemplateMask.csv"
dose_path_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\mean_film_dose_map\\mean_dose_map_grid_.npy"
save_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\131021\\ColonyData"

position = ["A","B","C","D"]
# kernel_size = 3.9 #mm
# kernel_size = int(kernel_size*47) #pixels/mm
#cropping_limits = [250,2200,300,1750]

#cropping_limits = [225,2200,300,1750]


# cropping_limits = [225,2200,350,1950]
#cropping_limits = [210,2100,405,1900]

# cropping_limits_2D = [100,2000,350,1850] #absolute max area
cropping_limits_2D = [200,2000,350,1850]
num_regressors = 2
peak_dist_reg = False

kernel_size_mm = [1]#[0.5,1,2,3,4]
kernel_size_p = [int(i*47) for i in kernel_size_mm]
SC = {}
X_grid = {}
peak_dist = {}

plt.style.use("seaborn")
"""
18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
We therefore combine these data to find alpha beta for open field irradiation.
"""

plt.imshow(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\20112019\\Control\\A549-2011-K1-SegMask.csv"))
plt.close()

"""
Finding the number of counted colonies for control (0Gy) and open field
experiments (2Gy and 5Gy)
"""
#time, mode, position, kernel_size, dose_map_path, template_file, save_path, dose, cropping_limits, data_extract = True
for i in range(len(kernel_size_mm)):
    print("-------------------------")
    print("running for kernel " + str(kernel_size_mm[i]) + "mm")
    print("-------------------------")
    control  = True
    if control == True:
        survival_control = survival_analysis(folder, time, mode[0], position,
                                             kernel_size_p[i], dose_map_path = None,
                                             template_file = template_file_control,
                                             save_path = save_path, dose = ctrl_dose,
                                             cropping_limits = cropping_limits_2D,
                                             data_extract = False)
        ColonyData_control, data_control = survival_control.data_acquisition()
        survival_control.Colonymap()
        pooled_SC_ctrl = survival_control.Quadrat()
        print(pooled_SC_ctrl.shape)

        #mean_SC_ctrl = np.mean(pooled_SC_ctrl)



    grid = True
    if grid == True:
        survival_grid = survival_analysis(folder, time, mode[1], position, kernel_size_p[i],
                                          dose_path_grid, template_file_grid, save_path = save_path,
                                          dose = dose, cropping_limits = cropping_limits_2D,
                                          data_extract = False)
        ColonyData_grid, data_grid  = survival_grid.data_acquisition()
        survival_grid.Colonymap()
        survival_grid.registration()





        _,dose_map = survival_grid.Quadrat("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\grid_survival10Gy_130821_dots_{}mm.png".format(kernel_size_mm[i]))
        dose2Gy_grid, SC_grid_2Gy = survival_grid.SC(2)
        dose5Gy_grid, SC_grid_5Gy = survival_grid.SC(5)
        dose10Gy_grid, SC_grid_10Gy = survival_grid.SC(10)

        # [225,2100,405,1900]
        if peak_dist_reg == True or num_regressors == 4:
            peak_dist = survival_grid.nearest_peak(cropping_limits_2D)/(47*10) #cm
            """
            All cell flask quadrat centers are assumed to have the same distance to nearest peak.
            """

            peak_dist = np.tile(np.ravel(peak_dist),len(dose)*pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1]*pooled_SC_ctrl.shape[2])

    tot_irradiatet_area = 25*100 #mm^2  #24.505*100 #mm^2
    hole_diameter = 5 #mm
    peak_area = 7 * np.pi* (hole_diameter/2)**2 #7 grid holes with 5 mm diameter
    peak_area_ratio = peak_area/tot_irradiatet_area
    valley_area_ratio = 1-peak_area_ratio#(tot_irradiatet_area-peak_area)/tot_irradiatet_area

    print(peak_area_ratio, valley_area_ratio)

    print(SC_grid_2Gy.shape)
    SC_grid_df = pd.DataFrame({"GRID dots 2 Gy":SC_grid_2Gy.ravel(), "GRID dots 5 Gy":SC_grid_5Gy.ravel(), "GRID dots 10 Gy":SC_grid_10Gy.ravel()})
    print(SC_grid_df)
    sb.boxplot(data =SC_grid_df)
    plt.ylabel("SC", fontsize = 15)
    plt.tight_layout()
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\GRID_dots_boxplot.png", dpi = 300)
    plt.show()


    SC[kernel_size_mm[i]], tot_dose_axis = data_stacking_2(True, SC_grid_2Gy,
                                        SC_grid_5Gy, SC_grid_10Gy,
                                        dose2Gy_grid, dose5Gy_grid, dose10Gy_grid)

    if num_regressors == 1:
        tot_len = len(np.ravel(pooled_SC_ctrl))
        X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len)]).T#design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, num_regressors, 0,0,0)
        tot_len = len(tot_dose_axis)
        X_grid[kernel_size_mm[i]] =  np.array([np.repeat(1,tot_len),tot_dose_axis]).T

    elif num_regressors == 2:
        tot_len = len(np.ravel(pooled_SC_ctrl))
        X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len),np.repeat(0,tot_len)]).T#design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, num_regressors, 0,0,0)
        tot_len = len(tot_dose_axis)
        X_grid[kernel_size_mm[i]] =  np.array([np.repeat(1,tot_len),tot_dose_axis, tot_dose_axis**2]).T

    elif num_regressors == 3:
        tot_len = len(np.ravel(pooled_SC_ctrl))
        X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len),np.repeat(0,tot_len),
                           np.repeat(0,tot_len)]).T #peak_area_ratio and peak dist is 0 for ctrl
        tot_len = len(tot_dose_axis)
        if peak_dist_reg == False:
            X_grid[kernel_size_mm[i]] =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis, tot_dose_axis**2,
                                np.repeat(peak_area_ratio,tot_len)]).T
        else:
            X_grid[kernel_size_mm[i]] =  np.array([np.repeat(1,tot_len),
                                tot_dose_axis, tot_dose_axis**2,
                                peak_dist]).T

    elif num_regressors == 4:
        tot_len = len(np.ravel(pooled_SC_ctrl))
        X_ctrl = np.array([np.repeat(1,tot_len),np.repeat(0,tot_len),np.repeat(0,tot_len),
                           np.repeat(0,tot_len),np.repeat(0,tot_len)]).T#design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, num_regressors, 0,0,0)

        tot_len = len(tot_dose_axis)
        X_grid[kernel_size_mm[i]] =  np.array([np.repeat(1,tot_len),
                            tot_dose_axis, tot_dose_axis**2,
                            np.repeat(peak_area_ratio,tot_len),peak_dist]).T



    """X_grid[kernel_size_mm[i]] = np.array([np.repeat(1,len(tot_dose_axis)),
                                          tot_dose_axis, tot_dose_axis**2,
                                          np.repeat(peak_area_ratio, len(tot_dose_axis)),
                                          np.repeat(valley_area_ratio, len(tot_dose_axis)),
                                          peak_dist]).T"""
    # X_grid[kernel_size_mm[i]] = design_matrix(len(SC[kernel_size_mm[i]]),tot_dose_axis, num_regressors,peak_area_ratio,valley_area_ratio, peak_dist)
    tot_len = len(np.ravel(pooled_SC_ctrl))

    X_ctrl_train,X_ctrl_test,SC_ctrl_train,SC_ctrl_test = train_test_split(X_ctrl,np.ravel(pooled_SC_ctrl),test_size = 0.2)
    X_grid_dots_train, X_grid_dots_test, SC_grid_dots_train, SC_grid_dots_test = train_test_split(X_grid[kernel_size_mm[i]],SC[kernel_size_mm[i]], test_size = 0.2)

    SC_train = np.concatenate((SC_ctrl_train, SC_grid_dots_train))
    X_train = np.vstack((X_ctrl_train, X_grid_dots_train))

    SC_test = np.concatenate((SC_ctrl_test, SC_grid_dots_test))
    X_test = np.vstack((X_ctrl_test, X_grid_dots_test))

    print(X_ctrl.shape, len(tot_dose_axis), np.ravel(pooled_SC_ctrl).shape)



    SC_len = [0,
             len(SC_ctrl_train),           #need to account for splitting into train and test
             len(SC_ctrl_train) +  len(SC_grid_dots_train)]
    legend = ["Control","GRID Dots"]

    poisson_results = 'GLM_results_{}mm_DOTS_{}regressors.tex'.format(kernel_size_mm[i], num_regressors)
    model, mean_pred_SC_train, summary = poisson_regression(SC_train,X_train,num_regressors,
                              r"GRID: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size_mm[i], kernel_size_mm[i]),
                              'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\regression results\\' + poisson_results,
                              legend, SC_len, kernel_size_mm[i], False)
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_DOTS_{}mm_{}regressors.png".format(kernel_size_mm[i], num_regressors), dpi = 1200)
    plt.close()

# if peak_dist_reg == True and num_regressors == 3:
#     with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_X_{}regressors_peak_dist.pickle".format(num_regressors), 'wb') as handle:
#         pickle.dump(X_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_SC_{}regressors_peak_dist.pickle".format(num_regressors), 'wb') as handle:
#         pickle.dump(SC, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# elif peak_dist_reg == False and num_regressors == 3:
#     with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_X_{}regressors_peak_area.pickle".format(num_regressors), 'wb') as handle:
#         pickle.dump(X_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_SC_{}regressors_peak_area.pickle".format(num_regressors), 'wb') as handle:
#         pickle.dump(SC, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# else:
#     with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_X_{}regressors.pickle".format(num_regressors), 'wb') as handle:
#         pickle.dump(X_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Dotted GRID\\GRID_Dots_SC_{}regressors.pickle".format(num_regressors), 'wb') as handle:
#         pickle.dump(SC, handle, protocol=pickle.HIGHEST_PROTOCOL)


# np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_X_w.g_factor.npy", X_grid)
# np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_SC_w.g_factor.npy", SC)

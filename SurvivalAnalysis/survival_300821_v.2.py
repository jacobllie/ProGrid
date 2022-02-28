from survival_analysis3 import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f, ttest_ind
from kernel_density_estimation import kde
import seaborn as sb
from scipy import stats, optimize
import seaborn as sb
from poisson import poisson
from scipy.interpolate import interp1d
from utils import K_means, logLQ, fit, poisson_regression, data_stacking,design_matrix, data_stacking_2, mean_survival
import sys
from plotting_functions_survival import pooled_colony_hist, survival_curve_grid, survival_curve_open, pred_vs_true_SC
import cv2
from playsound import playsound
from sklearn.model_selection import train_test_split

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

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021"
time = ["18112019", "20112019"]
mode = ["Control", "Open", "GRID Stripes"]
dose = ["02", "05", "10"]
ctrl_dose = ["00"]
template_file_control =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Control\\A549-1811-K1-TemplateMask.csv"
template_file_open = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Open\\A549-1811-02-open-A-TemplateMask.csv"
template_file_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-02-gridS-A-TemplateMask.csv"
dose_path_open = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open.npy"
dose_path_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_grid.npy"
position = ["A","B","C","D"]
kernel_size = 3.9 #mm
kernel_size = int(kernel_size*47) #pixels/mm
#cropping_limits = [250,2200,300,1750]

#cropping_limits = [225,2200,300,1750]
#cropping_limits = [225,2200,350,1950]
cropping_limits = [225,2100,405,1950]

plt.style.use("seaborn")
"""
18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
We therefore combine these data to find alpha beta for open field irradiation.
"""

plt.imshow(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Control\\A549-1811-K1-SegMask.csv"))
plt.close()

"""
Finding the number of counted colonies for control (0Gy) and open field
experiments (2Gy and 5Gy)
"""

control  = True
if control == True:
    survival_control = survival_analysis(folder, time, mode[0], position, kernel_size, dose_map_path = None, template_file = template_file_control, dose = ctrl_dose, cropping_limits = cropping_limits)
    ColonyData_control, data_control = survival_control.data_acquisition()
    survival_control.Colonymap()
    pooled_SC_ctrl = survival_control.Quadrat()

    pooled_SC_ctrl = np.reshape(pooled_SC_ctrl[:,0,:], (pooled_SC_ctrl.shape[0],
                          pooled_SC_ctrl.shape[2], pooled_SC_ctrl.shape[3],pooled_SC_ctrl.shape[4]))
    mean_SC_ctrl = np.mean(pooled_SC_ctrl)

open = True

if open == True:
    print("yes im open")
    survival_open = survival_analysis(folder, time, mode[1], position, kernel_size, dose_path_open, template_file_open, dose = dose, cropping_limits = cropping_limits)
    ColonyData_open, data_open  = survival_open.data_acquisition()
    survival_open.Colonymap()
    survival_open.registration()
    survival_open.Quadrat()

    dose2Gy_open, SC_open_2Gy = survival_open.SC(2)
    dose5Gy_open, SC_open_5Gy = survival_open.SC(5)




test_image = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))
Grid = True
if Grid == True:

    survival_grid = survival_analysis(folder, time, mode[2], position, kernel_size, dose_path_grid, template_file_grid, dose = dose, cropping_limits = cropping_limits)
    ColonyData_grid, data_grid  = survival_grid.data_acquisition()

    survival_grid.Colonymap()
    survival_grid.registration()

    _,dose_map = survival_grid.Quadrat()

    dose2Gy_grid, SC_grid_2Gy = survival_grid.SC(2)
    dose5Gy_grid, SC_grid_5Gy = survival_grid.SC(5)
    dose10Gy_grid, SC_grid_10Gy = survival_grid.SC(10)

    peak_dist = survival_grid.nearest_peak()/47 #distance from pooled dose quadrat centers to peak dose in mm
    print(peak_dist.shape)
    print(np.ravel(peak_dist))
    #pooled_colony_hist(pooled_SC_ctrl,SC_grid_2Gy,SC_grid_5Gy,SC_grid_10Gy, kernel_size)

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


    grid_dots = False
    if grid_dots == True:
        X_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_X_w.g_factor.npy")
        SC_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_SC_w.g_factor.npy")
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


    model, mean_pred_SC_train = poisson_regression(SC_train,X_train,4,
                              r"GRID&OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                              'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                              False)

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
    plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_train_test_39mm_survival.png", dpi = 1200)

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
    plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_open_grid_39mm_survival.png", dpi = 1200)
    plt.show()

    sys.exit()


    """
    No train test split
    """

    SC = np.concatenate((np.ravel(pooled_SC_ctrl), SC_open, SC_grid))
    X = np.vstack((X_ctrl,X_open,X_grid))


    true_SC = mean_survival(X,SC)
    print("True average survival")
    model, predicted_SC = poisson_regression(SC,X,4,
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
    model, mean_pred_SC_train = poisson_regression(SC_train,X_train,5,
                                                   r"GRID&OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                                                   'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                                                   False)

    plt.show()
method3 = True
if method3 == True:
    """
    Introduce distance to peak as a regressor
    """
    tot_irradiatet_area = 24.505*100 #mm^2
    peak_area = 3*215+170.75 #3 full peaks, 1 trapezoidal peak
    valley_area_ratio = (tot_irradiatet_area-peak_area)/tot_irradiatet_area
    peak_area_ratio = peak_area/tot_irradiatet_area

    SC_open, tot_dose_axis_open = data_stacking_2(False, SC_open_2Gy,
                                                SC_open_5Gy, dose2Gy_open,
                                                dose5Gy_open)
    SC_grid, tot_dose_axis_grid = data_stacking_2(True, SC_grid_2Gy,
                                                  SC_grid_5Gy, SC_grid_10Gy,
                                                  dose2Gy_grid, dose5Gy_grid, dose10Gy_grid)

    grid_dots = True
    if grid_dots == True:
        X_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_X_w.g_factor.npy")
        SC_grid_dots = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_SC_w.g_factor.npy")
        X_grid_dots_train, X_grid_dots_test, SC_grid_dots_train, SC_grid_dots_test = train_test_split(X_grid_dots,SC_grid_dots, test_size = 0.2)


    print(SC_grid.shape)
    """
    Adding the control survival and doses (0Gy) to open. Then make individual design matrix with different G factors
    """
    peak_dist = np.tile(np.ravel(peak_dist),len(dose)*pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1])
    # peak_dist = [i**2 if i > 0 else 0 for i in stacked_dist]
    tmp = np.repeat(dose2Gy_grid*0,pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1])
    X_ctrl = design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, 4,0,0)
    X_open = design_matrix(len(SC_open),tot_dose_axis_open,4, 1,0)
    X_grid = design_matrix(len(SC_grid),tot_dose_axis_grid, 4, peak_area_ratio, valley_area_ratio)

    print(np.tile(np.ravel(peak_dist),len(dose)*pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1]))


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

        SC_train = np.concatenate((SC_ctrl_train,SC_open_train , SC_grid_train))
        X_train = np.vstack((X_ctrl_train, X_open_train, X_grid_train))
        SC_test = np.concatenate((SC_ctrl_test, SC_open_test, SC_grid_test))
        X_test = np.vstack((X_ctrl_test, X_open_test, X_grid_test))

    mean_obs_SC_train = mean_survival(X_train,SC_train)


    model, mean_pred_SC_train = poisson_regression(SC_train,X_train,4,
                              r"GRID: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                              'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                              True)

    plt.show()

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
    #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_train_test_39mm_survival.png", dpi = 1200)

    plt.show()

    # mean_obs_SC_open = mean_survival(X_open_test, SC_open_test)
    # dose_axis_open = np.linspace(0,np.max(X_open_test[:,1]),len(mean_obs_SC_open))
    # mean_obs_SC_open = mean_survival(X_open_test,SC_open_test)
    # predicted_tmp = model.get_prediction(X_open_test).summary_frame()["mean"]
    # mean_pred_SC_open = mean_survival(X_open_test,predicted_tmp)

    mean_obs_SC_grid = mean_survival(X_grid_test, SC_grid_test)
    dose_axis_grid = np.linspace(0,np.max(X_grid_test[:,1]),len(mean_obs_SC_grid))
    mean_obs_SC_grid = mean_survival(X_grid_test,SC_grid_test)
    predicted_tmp = model.get_prediction(X_grid_test).summary_frame()["mean"]
    mean_pred_SC_grid = mean_survival(X_grid_test,predicted_tmp)



    plt.suptitle("Mean SC observed vs predicted")
    plt.subplot(121)
    # pred_vs_true_SC(mean_obs_SC_open, mean_pred_SC_open, dose_axis_open, "OPEN")
    plt.subplot(122)
    pred_vs_true_SC(mean_obs_SC_grid, mean_pred_SC_grid, dose_axis_grid, "GRID")
    #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\ObsVSPred_open_grid_39mm_survival.png", dpi = 1200)
    plt.show()

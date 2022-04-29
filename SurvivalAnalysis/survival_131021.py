from survival_analysis4 import survival_analysis
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
import pickle

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
         "WTF Richard.wav", "you almost made me drop my croissant vine.wav","Martin, Thea og Nikolai, hele klippet.wav",
         "TikTok - My friend here, Justin, is cracked at Fortnite.wav","Wenches Kantine - Tacobaguette_duger.wav","Wenches Kantine - Tacobaguette_fredag.wav"]
weights = np.zeros(len(sounds))
weights[-1] = 1


#playsound(sound_path + np.random.choice(sounds,p = weights))

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021"
time = ["20112019"]
mode = ["Control", "GRID Dots"]
dose = ["02", "05", "10"]
ctrl_dose = ["00"]
template_file_control =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\20112019\\Control\\A549-2011-K1-TemplateMask.csv"
template_file_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\20112019\\GRID Dots\\A549-2011-02-gridC-A-TemplateMask.csv"
dose_path_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\mean_film_dose_map\\mean_dose_grid_1D.npy"
save_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\131021\\ColonyData"

position = ["A","B","C","D"]
# kernel_size = 3.9 #mm
# kernel_size = int(kernel_size*47) #pixels/mm
#cropping_limits = [250,2200,300,1750]

#cropping_limits = [225,2200,300,1750]


# cropping_limits = [225,2200,350,1950]
#cropping_limits = [210,2100,405,1900]

cropping_limits_2D = [100,2000,350,1850] #absolute max area
num_regressors = 5

kernel_size_mm = [3,1,2,0.5,4]
kernel_size_p = [int(i*47) for i in kernel_size_mm]
SC = {}
X_grid = {}
peak_dist = {}

plt.style.use("seaborn")
"""
18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
We therefore combine these data to find alpha beta for open field irradiation.
"""

plt.imshow(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\20112019\\Control\\A549-2011-K1-SegMask.csv"))
plt.close()

"""
Finding the number of counted colonies for control (0Gy) and open field
experiments (2Gy and 5Gy)
"""
#time, mode, position, kernel_size, dose_map_path, template_file, save_path, dose, cropping_limits, data_extract = True
for i in range(1):
    control  = False
    if control == True:
        survival_control = survival_analysis(folder, time, mode[0], position,
                                             kernel_size_p[i], dose_map_path = None,
                                             template_file = template_file_control,
                                             save_path = save_path, dose = ctrl_dose,
                                             cropping_limits = cropping_limits_2D,
                                             data_extract = True)
        ColonyData_control, data_control = survival_control.data_acquisition()
        survival_control.Colonymap()
        #pooled_SC_ctrl = survival_control.Quadrat()
        #pooled_SC_ctrl = np.reshape(pooled_SC_ctrl[:,0,:], (pooled_SC_ctrl.shape[0],
        #                       pooled_SC_ctrl.shape[2], pooled_SC_ctrl.shape[3],pooled_SC_ctrl.shape[4]))
        #mean_SC_ctrl = np.mean(pooled_SC_ctrl)



    grid = True
    if grid == True:
        survival_grid = survival_analysis(folder, time, mode[1], position, kernel_size_p[i],
                                          dose_path_grid, template_file_grid, save_path = save_path,
                                          dose = dose, cropping_limits = cropping_limits_2D,
                                          data_extract = True)
        ColonyData_grid, data_grid  = survival_grid.data_acquisition()
        survival_grid.Colonymap()
        survival_grid.registration()



        _,dose_map = survival_grid.Quadrat("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\2D survival\\grid_survival5Gy_131021_dots_{}mm.png".format(kernel_size_mm[i]))

        sys.exit()
        dose2Gy_grid, SC_grid_2Gy = survival_grid.SC(2)
        dose5Gy_grid, SC_grid_5Gy = survival_grid.SC(5)
        dose10Gy_grid, SC_grid_10Gy = survival_grid.SC(10)


        peak_dist = survival_grid.nearest_peak([225,2100,405,1900])/(47*10) #cm

        """
        All cell flask quadrat centers are assumed to have the same distance to nearest peak.
        """
        peak_dist = np.tile(np.ravel(peak_dist),len(dose)*pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1])



    tot_irradiatet_area = 24.505*100 #mm^2
    hole_diameter = 5 #mm
    peak_area = 7 * np.pi* (hole_diameter/2)**2 #7 grid holes with 5 mm diameter
    valley_area_ratio = (tot_irradiatet_area-peak_area)/tot_irradiatet_area
    peak_area_ratio = peak_area/tot_irradiatet_area

    SC[kernel_size_mm[i]], tot_dose_axis = data_stacking_2(True, SC_grid_2Gy,
                                        SC_grid_5Gy, SC_grid_10Gy,
                                        dose2Gy_grid, dose5Gy_grid, dose10Gy_grid)


    # print((design_matrix(len(SC[i]),tot_dose_axis, 5, peak_area_ratio, valley_area_ratio)).shape)
    X_grid[kernel_size_mm[i]] = design_matrix(len(SC[kernel_size_mm[i]]),tot_dose_axis, num_regressors,peak_area_ratio, valley_area_ratio, peak_dist)
# Store data (serialize)
with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_X_{}regressors_distance.pickle".format(num_regressors), 'wb') as handle:
    pickle.dump(X_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_SC_{}regressors_distance.pickle".format(num_regressors), 'wb') as handle:
    pickle.dump(SC, handle, protocol=pickle.HIGHEST_PROTOCOL)


# np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_X_w.g_factor.npy", X_grid)
# np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GRID_Dots_SC_w.g_factor.npy", SC)

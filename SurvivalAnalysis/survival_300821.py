from survival_analysis3 import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f, ttest_ind
from kernel_density_estimation import kde
import seaborn as sb
import sys
from scipy import stats, optimize
import seaborn as sb
from poisson import poisson
from scipy.interpolate import interp1d
from utils import K_means, logLQ, fit, poisson_regression, data_stacking,design_matrix, data_stacking_2
import sys
from plotting_functions_survival import pooled_colony_hist, survival_curve_grid, survival_curve_open
import cv2
from playsound import playsound

sound_path = "C:\\Users\\jacob\\OneDrive\\Documents\\livet\\veldig viktig\\"
sounds = ["Ah Shit Here We Go Again - GTA Sound Effect (HD).mp3",
         "Anakin Skywalker - Are You An Angel.mp3","Get-in-there-Lewis-F1-Mercedes-AMG-Sound-Effect.wav",
          "he-need-some-milk-sound-effect.wav",
         "MOM GET THE CAMERA Sound Effect.mp3","My-Name-is-Jeff-Sound-Effect-_HD_.wav",
         "Nice (HD) Sound effects.mp3","Number-15_-Burger-king-foot-lettuce-Sound-Effect.wav",
         "oh_my_god_he_on_x_games_mode_sound_effect_hd_4036319132337496168.mp3",
         "Ok-Sound-Effect.wav","PIZZA TIME! Sound Effect (Peter Parker).mp3","WHY-ARE-YOU-GAY-SOUND-EFFECT.wav", "OKLetsGo.mp3"]
#playsound(sound_path + np.random.choice(sounds))

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

cropping_limits = [225,2200,300,1750]

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


    # plt.subplot(121)
    # plt.imshow(dose2Gy_open)
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(dose5Gy_open)
    # plt.colorbar()
    # plt.close()

    """
    Stacking all doses and survival data for open field
    """

    # SC_open, tot_dose_axis_open = data_stacking(dose2Gy_open*0, dose2Gy_open, dose5Gy_open,
    #                                   pooled_SC_ctrl,SC_open_2Gy, SC_open_5Gy)
    print("yoooooooooooooooooo")
    mean_quadrat_survival_open = np.zeros((3,SC_open_2Gy.shape[2],SC_open_2Gy.shape[3]))

    mean_quadrat_survival_open[0] = np.mean(pooled_SC_ctrl, axis = (0,1))
    mean_quadrat_survival_open[1] = np.mean(SC_open_2Gy, axis = (0,1))
    mean_quadrat_survival_open[2] = np.mean(SC_open_5Gy, axis = (0,1))


    plt.style.use("seaborn")
    # plt.subplot(121)
    # poisson_regression(mean_SC_ctrl, SC_open,tot_dose_axis_open, tot_dose_axis_open**2,1,0, 2,
    #                    r"OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
    #                    'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_OPEN_w.o.G_factor.tex',
    #                    False)

    """
    1D survival curve
    """

    #mean_survival = np.ones(3)
    #std_survival = np.zeros(3)
    #mean_dose = np.zeros(3)
    #
    #
    # SF1, mean_survival[1], std_survival[0], mean_dose[1] = \
    # survival_curve_open(2, dose2Gy_open, SC_open_2Gy, mean_SC_ctrl, "deeppink")
    #
    # SF2, mean_survival[2], std_survival[1], mean_dose[2] = \
    # survival_curve_open(5, dose5Gy_open, SC_open_5Gy, mean_SC_ctrl, "royalblue")
    #
    #
    # #normalizing surviving colonies in ctrl to average
    # SF = np.concatenate((np.ravel(pooled_SC_ctrl)/mean_SC_ctrl, np.ravel(SF1), np.ravel(SF2)))
    #
    #
    # print(len(SF))
    #
    # """
    # Tot dose axis needs to contain doses for all SF datapoints for 0 2 and 5 Gy.
    # """
    # dose_0Gy = np.zeros(SC_open_2Gy.shape[0]*SC_open_2Gy.shape[1]*SC_open_2Gy.shape[2]*SC_open_2Gy.shape[3])
    #
    # #tot_dose_axis = tot_dose_axis.append(list(np.zeros(SC_open_2Gy.shape[0]*SC_open_2Gy.shape[1]*SC_open_2Gy.shape[2]*SC_open_2Gy.shape[3])))
    # tot_dose_axis = []
    #
    #
    # for i in range(SC_open_2Gy.shape[0]*SC_open_2Gy.shape[1]):
    #     tot_dose_axis = np.concatenate((tot_dose_axis, np.ravel(dose2Gy_open)))
    # for i in range(SC_open_2Gy.shape[0]*SC_open_2Gy.shape[1]):
    #     tot_dose_axis = np.concatenate((tot_dose_axis, np.ravel(dose5Gy_open)))
    #
    # tot_dose_axis = np.concatenate((dose_0Gy, tot_dose_axis))
    # tot_dose_axis = tot_dose_axis[SF != 0]
    # SF = SF[SF != 0]
    #
    # print(tot_dose_axis.shape)
    # print(SF.shape)
    #
    #
    # plt.plot(tot_dose_axis,np.log(SF), "*")
    # #plt.plot()
    #
    # fitting_params = fit(logLQ, tot_dose_axis, np.log(SF))
    #
    # interp_dose = np.linspace(0,10,100)
    # print(interp_dose)
    # plt.plot(interp_dose, logLQ(interp_dose, fitting_params[0], fitting_params[1]), color  ="salmon",
    #          label = r"fit: -($\alpha \cdot d + \beta \cdot d^2$)" + "\n" + r"$\alpha = {:.4}, \beta = {:.4}$".format(fitting_params[0], fitting_params[1]))
    # plt.title(r"Survival curve for {} X {} $mm^2 kernels open$ ".format(kernel_size/47, kernel_size/47))
    # plt.xlabel("Dose [Gy]")
    # plt.ylabel(r"$\log{\frac{S}{S_{ctrl}}}$")
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\1D Survival\\peak_2.8_4.5_valley_0.4_1.2\\survival_3mm_open.png", bbox_inches = "tight", pad_inches = 0.1)
    # plt.close()
    #
    #
    # plt.title(r"Mean survival using {} X {} $mm^2$ kernels.".format(kernel_size/47,kernel_size/47))
    # plt.plot(0,np.log10(1),"o",color = "black",label = "0 Gy")
    # plt.errorbar(0,np.log10(mean_SC_ctrl/mean_SC_ctrl),yerr = np.std(pooled_SC_ctrl)/np.sqrt(len(np.ravel(pooled_SC_ctrl))), color = "black")
    # plt.xlabel("Dose [Gy]")
    # plt.ylabel(r"log10 $\frac{S_{irradiated}}{S_{ctrl}}$")
    # #plt.show()
    #
    # plt.close()
    # """
    # We need to insert 0 dose element when fitting. We need the array to be in descending order.
    # We therefore sort then reverse
    # """
    #
    # fitting_params = fit(logLQ, mean_dose, np.log(mean_survival))
    #
    # """
    # We extrapolate to 10 Gy because the segmentation algorithm struggles at 10 Gy
    # """
    #
    # interp_dose = np.linspace(0,10,100)
    # print(interp_dose)
    # plt.plot(interp_dose, logLQ(interp_dose, fitting_params[0], fitting_params[1]), color  ="salmon",
    #         label = r"fit: -($\alpha \cdot d + \beta \cdot d^2$)" + "\n" + r"$\alpha = {:.4}, \beta = {:.4}$".format(fitting_params[0], fitting_params[1]))
    # plt.legend()
    # plt.close()



test_image = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))
Grid = True
if Grid == True:

    survival_grid = survival_analysis(folder, time, mode[2], position, kernel_size, dose_path_grid, template_file_grid, dose = dose, cropping_limits = cropping_limits)
    ColonyData_grid, data_grid  = survival_grid.data_acquisition()

    survival_grid.Colonymap()
    survival_grid.registration()

    _,dose_map = survival_grid.Quadrat()

    """
    Finding area fraction peak vs valley
    """

    """
    cropping limits defines the area were interested in analysing
    the shape is x,y
    where len(x) defines the height of the ROI,
    while len(y) defines the width of the ROI
    """
    #roi_width = np.abs(cropping_limits[2] - cropping_limits[3])/47 #47 pixels per mm
    #roi_height = np.abs(cropping_limits[0]-cropping_limits[1])/47
    #roi_area = roi_width*roi_heigth #mm^2
    #peak_width = 4.68 #found from dose profile



    dose2Gy_grid, SC_grid_2Gy = survival_grid.SC(2)
    dose5Gy_grid, SC_grid_5Gy = survival_grid.SC(5)
    dose10Gy_grid, SC_grid_10Gy = survival_grid.SC(10)

    mean_quadrat_survival_grid = np.zeros((4,SC_grid_2Gy.shape[2],SC_grid_2Gy.shape[3]))

    mean_quadrat_survival_grid[0] = np.mean(pooled_SC_ctrl, axis = (0,1))

    """
    1D survival curve
    """
    #
    #pooled_colony_hist(pooled_SC_ctrl,SC_grid_2Gy,SC_grid_5Gy,SC_grid_10Gy, kernel_size)
    #
    #
    #
    # #peak_lower_lim = 3.6
    # #peak_higher_lim = 4.4
    # #valley_lower_lim = 0.6
    # #valley_higher_lim = 1.1
    # peak_lower_lim = 2.8
    # peak_higher_lim = 4.5
    # valley_lower_lim = 0.4
    # valley_higher_lim = 1.2
    #
    # mean_peak_dose = np.zeros(3)
    # mean_valley_dose = np.zeros(3)
    # mean_survival = np.zeros((3,2))
    # std_survival = np.zeros((3,2))
    #
    # #2 Gy
    # mean_survival[0], std_survival[0], mean_peak_dose[0], mean_valley_dose[0] = \
    # survival_curve_grid(peak_lower_lim, peak_higher_lim, valley_lower_lim,
    #                     valley_higher_lim, 2, dose2Gy_grid, SC_grid_2Gy, mean_SC_ctrl, "deeppink")
    #
    # #5Gy
    # mean_survival[1], std_survival[1], mean_peak_dose[1], mean_valley_dose[1] = \
    # survival_curve_grid(peak_lower_lim, peak_higher_lim, valley_lower_lim,
    #                     valley_higher_lim, 5, dose5Gy_grid, SC_grid_5Gy, mean_SC_ctrl,"royalblue")
    # #10Gy
    # mean_survival[2], std_survival[2], mean_peak_dose[2], mean_valley_dose[2] = \
    # survival_curve_grid(peak_lower_lim, peak_higher_lim, valley_lower_lim,
    #                     valley_higher_lim, 10, dose10Gy_grid, SC_grid_10Gy, mean_SC_ctrl,"forestgreen")
    # #print(mean_valley_dose, mean_peak_dose)
    # # print(np.concatenate((mean_valley_dose,mean_peak_dose)))
    #
    #
    # plt.title(r"Mean survival using {} X {} $mm^2$ kernels.".format(kernel_size/47, kernel_size/47))
    # plt.plot(0,np.log10(1),"o",color = "black",label = "0 Gy")
    # plt.errorbar(0,np.log10(mean_SC_ctrl/mean_SC_ctrl),yerr = np.std(pooled_SC_ctrl)/np.sqrt(len(np.ravel(pooled_SC_ctrl))))
    # plt.xlabel("Dose [Gy]")
    # plt.ylabel(r"log10 $\frac{S_{irradiated}}{S_{ctrl}}$")
    # plt.legend()
    # #plt.savefig(r"C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\mean_survival_4mm.png",dpi = 1200,bbox_inches = "tight",pad_inches = 0.1)
    # #plt.show()
    #
    # #we include 0 in the fit
    #
    # #fitting peak and valley survival together
    # tot_dose_axis = np.insert(np.concatenate((mean_valley_dose,mean_peak_dose)),0,0)
    # """
    # We need to insert 0 dose element when fitting. We need the array to be in descending order.
    # We therefore sort then reverse
    # """
    # tot_survival_axis = np.insert(np.sort(np.concatenate((mean_survival[:,0],mean_survival[:,1])))[::-1],0,1)
    # fitting_params = fit(logLQ, tot_dose_axis, np.log(tot_survival_axis))
    #
    # print(tot_survival_axis)
    # print(fitting_params.shape)
    #
    # interp_dose = np.linspace(0,np.max(mean_peak_dose),100)
    # print(interp_dose)
    # plt.plot(interp_dose, logLQ(interp_dose, fitting_params[0], fitting_params[1]))
    #
    # plt.close()



"""
Poisson regression predicting number of survivors
"""

"""
tmp_dose2Gy, tmp_dose5Gy, tmp_dose10Gy, SC_grid_2Gy, SC_grid_5Gy, SC_grid_10Gy, pooled_SC_ctrl, SC_open_2Gy, SC_open_5Gy
"""

tot_irradiatet_area = 24.505*100 #mm^2
peak_area = 3*215+170.75 #3 full peaks, 1 trapezoidal peak
valley_area_ratio = (tot_irradiatet_area-peak_area)/tot_irradiatet_area
peak_area_ratio = peak_area/tot_irradiatet_area
print(peak_area_ratio, valley_area_ratio)
#Want to find peak and valley area fraction. Using dose image that all other images has been registered to
#Cropping it the same way


"""
want to stack all doses and survival for grid
"""
# SC_grid, tot_dose_axis_grid = data_stacking(dose2Gy_grid*0, dose2Gy_grid, dose5Gy_grid,
#                                   pooled_SC_ctrl, SC_grid_2Gy,
#                                   SC_grid_5Gy, dose10Gy_grid, SC_grid_10Gy)

# plt.subplot(122)
# poisson_regression(mean_SC_ctrl, SC_grid,tot_dose_axis_grid, tot_dose_axis_grid**2,peak_area_ratio,valley_area_ratio,
#                    2, r"GRID: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
#                    'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID_w.o.G_factor.tex',
#                    False)
# #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\survival_poisson_OPENvsGRID_3.9mm_w.G_factor.png", dpi = 1200, bbox_inches = "tight", pad_inches = 0.1)
# plt.legend()
# plt.close()

print("okay lets go")


"""
Now we stack all data togheter and perform poisson regression.
"""


SC_open, tot_dose_axis_open = data_stacking_2(False, SC_open_2Gy,
                                            SC_open_5Gy, dose2Gy_open,
                                            dose5Gy_open)
SC_grid, tot_dose_axis_grid = data_stacking_2(True, SC_grid_2Gy,
                                              SC_grid_5Gy, SC_grid_10Gy,
                                              dose2Gy_grid, dose5Gy_grid, dose10Gy_grid)

plt.plot(tot_dose_axis_open, SC_open,"*",label="open")
plt.plot(tot_dose_axis_grid,SC_grid,"*",label = "grid")
plt.legend()
plt.show()
#SC = np.insert(np.concatenate((SC_open,SC_grid)), 0,np.ravel(pooled_SC_ctrl))
#tot_dose_axis = np.insert(np.concatenate((tot_dose_axis_open,tot_dose_axis_grid)), 0 ,np.repeat(dose2Gy_grid*0, pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1]))
"""
Adding the control survival and doses (0Gy) to open. Then make individual design matrix with different G factors
"""
tmp = np.repeat(dose2Gy_grid*0,pooled_SC_ctrl.shape[0]*pooled_SC_ctrl.shape[1])
X_ctrl = design_matrix(len(np.ravel(pooled_SC_ctrl)), tmp, tmp**2, 4, 0, 0)
X_open = design_matrix(len(SC_open),tot_dose_axis_open,tot_dose_axis_open**2,4, 1, 0)
X_grid = design_matrix(len(SC_grid),tot_dose_axis_grid, tot_dose_axis_grid**2, 4, peak_area_ratio, valley_area_ratio)



SC = np.concatenate((np.ravel(pooled_SC_ctrl), SC_open, SC_grid))
X = np.vstack((X_ctrl,X_open,X_grid))

#SC = np.concatenate((np.ravel(pooled_SC_ctrl), SC_open))
#X = np.vstack((X_ctrl,X_open))

#SC = np.concatenate((np.ravel(pooled_SC_ctrl), SC_grid))
#X = np.vstack((X_ctrl,X_grid))


print(X[:,:2].shape)
model = poisson_regression(SC,X,4,
                          r"GRID&OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                          'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                          False)

plt.show()

sys.exit()


X = design_matrix(len(np.concatenate((SC_open,SC_grid))),np.concatenate((tot_dose_axis_open,tot_dose_axis_grid)), np.concatenate((tot_dose_axis_open**2,tot_dose_axis_grid**2)),4,peak_area_ratio,valley_area_ratio)
model = poisson_regression(np.concatenate((SC_open,SC_grid)),X,4,
                          r"GRID&OPEN: Surviving colonies within {:.1f} X {:.1f} $mm^2$ square".format(kernel_size/47, kernel_size/47),
                          'C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\231121\\GLM_results_39mm_GRID&OPEN_w.G_factor.tex',
                          False)


#plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\survival_poisson_OPEN&GRID_3.9mm_w.o.G_factor.png", dpi = 1200)


plt.show()

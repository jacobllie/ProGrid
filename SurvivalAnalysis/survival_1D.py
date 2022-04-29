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
from plotting_functions_survival import pooled_colony_hist, survival_curve_grid, survival_curve_open, pred_vs_true_SC
import cv2
from playsound import playsound
from sklearn.model_selection import train_test_split
import pickle
import skimage.transform as tf




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
save_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\ColonyData"
position = ["A","B","C","D"]

# kernel_size = 3.9 #mm
# kernel_size = int(kernel_size*47) #pixels/mm
#cropping_limits = [250,2200,300,1750]

#cropping_limits = [225,2200,300,1750] #this will not affect 1D analysis
#cropping_limits = [225,2200,350,1950]
# cropping_limits = [210,2100,405,1900]

kernel_size_mm = [3,1,2,3,4]
# kernel_size_mm = [0.5,1,2,3,4]
kernel_size_p = [int(i*47) for i in kernel_size_mm]

num_regressors = 3

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
    #cropping_limits_1D = [50,2700, 10,2000]
    #cropping_limits_1D = [50,2700, 70,1900]  #this matches with dose film the most

    cropping_limits_1D = [100,2700, 70,2100]  #this matches with dose film the most


    flask_template = np.asarray(pd.read_csv(template_file_control))[cropping_limits_1D[0]:cropping_limits_1D[1],cropping_limits_1D[2]:cropping_limits_1D[3]]

    plt.imshow(flask_template)
    plt.show()

    survival_control = survival_analysis(folder, time, mode[0], position,
                       kernel_size_p[0], dose_map_path = None,
                       template_file = template_file_control,
                       save_path = save_path, dose = ctrl_dose, cropping_limits = cropping_limits_1D,
                       data_extract = True)
    ColonyData_control, data_control = survival_control.data_acquisition()

    colony_map_ctrl = survival_control.Colonymap()
    #pooled_SC_ctrl = survival_control.Quadrat()

    #pooled_SC_ctrl = np.reshape(pooled_SC_ctrl[:,0,:], (pooled_SC_ctrl.shape[0],
                          #pooled_SC_ctrl.shape[2], pooled_SC_ctrl.shape[3],pooled_SC_ctrl.shape[4]))
    #mean_SC_ctrl = np.mean(pooled_SC_ctrl)
    #extracting flask template and cropping away edges



    #GREY_chan_cropped = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes\\EBT3_Stripes_310821_Xray220kV_5Gy1_001.tif",-1)[10:-10,10:-10,0]
    #GREY_chan_cropped = tf.rescale(GREY_chan_cropped,4)
    plt.subplot(131)
    plt.imshow(flask_template)
    #flask_template_1D = flask_template[cropping_limits[0]:cropping_limits[1],cropping_limits[2]:cropping_limits[3]]
    flask_template_copy = np.zeros(flask_template.shape)
    flask_template_copy[flask_template == 0] = 1
    flask_template_copy[flask_template == 1] = 0.5



    plt.close()
    #initializing surviving colony matrix and cropping colony_map to match template
    num_colonies_ctrl = np.zeros((len(time),len(ctrl_dose),len(position),len(flask_template)))

    print(num_colonies_ctrl.shape)
    #image height is needed for plotting later

    #want to include all data for cells
    image_height = np.linspace(0,int(flask_template.shape[0]/47),flask_template.shape[0]) #mm
    plt.suptitle("Crop {}".format(flask_template.shape))
    plt.subplot(121)
    plt.imshow(flask_template)

    row_weight = np.zeros(len(flask_template))
    for i in range(len(flask_template)):
        ones = len(flask_template[i,flask_template[i] > 0])
        zeros = len(flask_template[i,flask_template[i] == 0])
        if ones == 0:
            row_weight[i] = 0
        else:
            row_weight[i] = 1 + zeros/ones
    plt.subplot(122)
    plt.plot(row_weight)
    #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\1D analysis\\weightfactor_{}.png".format(flask_template.shape), dpi = 1200)
    plt.show()

    """
    Iterating over all rows in colonymap and summing up surviving colonies shouldn't there be more since they are 47*37 bands?
    """
    for i in range(len(time)):
        #skipping 10 Gy
        for j in range(len(ctrl_dose)):
            for k in range(len(position)):
                for l in range(cropping_limits_1D[0],cropping_limits_1D[1]):
                    m = l - cropping_limits_1D[0] #num_colonies not as big as colony_map
                    num_colonies_ctrl[i,j,k,m] = np.sum(colony_map_ctrl[i,j,k,l])*row_weight[m]
    #finding mean number of surviving colonies for all times and all positions


    mean_colonies_ctrl = np.mean(num_colonies_ctrl, axis = (0,2,3))
    std_colonies_ctrl = np.std(num_colonies_ctrl, axis = (0,2,3))

open_= True

if open_ == True:
    #removing edges
    #initializing class
    survival_open = survival_analysis(folder, time, mode[1], position,
                                      kernel_size_p[0], dose_path_open,
                                      template_file_open,
                                      save_path = save_path, dose = dose,
                                      cropping_limits = cropping_limits_1D, data_extract = True)
    #gathering colonydata, and count data
    ColonyData_open, data_open  = survival_open.data_acquisition()
    #placing colonies in their respective coordinates
    colony_map_open = survival_open.Colonymap()


    #dont need to crop colonymap

    #skipping 10 Gy because the algorithm doesnt work for this dose


    num_colonies_open = np.zeros((len(time),len(dose)-1,len(position),len(flask_template)))

    print(num_colonies_open.shape)

    """
    Iterating over all rows in colonymap and summing up surviving colonies for
    open field and grid comparison
    """
    for i in range(len(time)):
        #skipping 10 Gy
        for j in range(len(dose)-1):
            print(dose[j])
            for k in range(len(position)):
                for l in range(cropping_limits_1D[0],cropping_limits_1D[1]):
                    m = l - cropping_limits_1D[0]
                    num_colonies_open[i,j,k,m] = np.sum(colony_map_open[i,j,k,l]) #*row_weight[m]
    #finding mean number of surviving colonies for all times and all positions

    print(num_colonies_open.shape)


    """
    1D LQ modelling
    We do not crop cell flask for 1D analysis
    """
    #for all times and positions we find the total number of colonies
    #we skip 10 Gy because the segmentation algorithm doesnt work for 10 Gy

    n = 2*4 #we have 8 flasks per dose


    surviving_colonies_open = np.sum(colony_map_open[:,:-1], axis = (3,4))

    surviving_colonies_ctrl = np.sum(colony_map_ctrl, axis = (3,4))

    print(surviving_colonies_ctrl.shape, surviving_colonies_open.shape)


    print("181119 CTRL")
    print(surviving_colonies_ctrl[0,0])
    print("201119 CTRL")
    print(surviving_colonies_ctrl[1,0])

    print("181119 OPEN 2 Gy")
    print(surviving_colonies_open[0,0])
    print("201119 OPEN 2Gy")
    print(surviving_colonies_open[1,0])

    print("181119 OPEN 5Gy")
    print(surviving_colonies_open[0,1])
    print("201119 OPEN 5Gy")
    print(surviving_colonies_open[1,1])

    dose_axis = np.append(np.repeat(0,len(surviving_colonies_ctrl[:,0].flatten())),
                np.append(np.repeat(2,len(surviving_colonies_open[:,0].flatten())),
                np.repeat(5,len(surviving_colonies_open[:,1].flatten())))) #2 dates, 4 positions

    mean_survival_ctrl = np.mean(surviving_colonies_ctrl, axis = 0)
    combined_survival = np.append(surviving_colonies_ctrl[:,0]/mean_survival_ctrl,
                        np.append(surviving_colonies_open[:,0]/mean_survival_ctrl,
                        surviving_colonies_open[:,1]/mean_survival_ctrl))






    plt.plot(dose_axis,np.log(combined_survival), "*")
    plt.show()
    coeff,cov = np.polyfit(dose_axis,np.log(combined_survival),full = False,deg = 2, cov = True) #coeff from x^n to intercept
    cov = np.diag(cov)

    n = len(combined_survival)
    cov = np.diag(cov[:-1]) #only include uncertainty in beta and alpha

    dose_interp = np.linspace(0,10,1000)

    chi2 = chi2.ppf(0.95,1)
    print(chi2)
    #print(chi2.shape)



    """
    For dose = 0, the derivative will become 0, so we need to use
    the second order delta method
    """

    dSdp = np.array([dose_interp + 1, 2*coeff[0]*(dose_interp + 1)])

    cov_S = (dSdp.T.dot(cov).dot(dSdp)) * chi2
    t_crit = stats.t.ppf(0.95, n - k)

    _,MSE,_,_,_ = np.polyfit(dose_axis,np.log(combined_survival),full = True,deg = 2)

    MSE = MSE/n
    print(MSE)
    sx = np.std(dose_interp)
    mean_x = np.mean(dose_interp)
    print(combined_survival)
    plt.title("LQ model fitted to normalized log survival data")
    plt.xlabel("Dose")
    plt.ylabel(r"$\log{S_{irr}/S_{ctrl}}$")
    plt.plot(dose_interp, np.polyval(coeff, dose_interp), label = r"{:+f} $\pm$ {:f} d {:+f} $\pm$ {:f} $d^2$".format(coeff[1], np.diag(np.sqrt(cov))[0], coeff[0], np.diag(np.sqrt(cov))[1]))
    plt.plot(dose_axis,np.log(combined_survival), "*",label = "data")
    #we need to divide the standard deviation with the square root of number of points per dose because the mean is supposed to be inside the confidence interval 95% of the times.
    #plt.fill_between(dose_interp, np.polyval(coeff,dose_interp) - t_crit * np.sqrt(np.diag(cov_S))/np.sqrt(n), np.polyval(coeff, dose_interp) + t_crit * np.sqrt(np.diag(cov_S))/np.sqrt(n), alpha = 0.5, color = "grey")
    CI_p = t_crit *np.sqrt(MSE *(1/n + 1 + (dose_interp - np.mean(dose_interp))**2/np.std(dose_interp)**2))
    CI = t_crit *np.sqrt(MSE *(1/n  + (dose_interp - np.mean(dose_interp))**2/np.std(dose_interp)**2))

    plt.fill_between(dose_interp,np.polyval(coeff,dose_interp) - CI_p,np.polyval(coeff,dose_interp) + CI_p , alpha = 0.6, label = "95% P.I.")
    plt.fill_between(dose_interp,np.polyval(coeff,dose_interp) - CI,np.polyval(coeff,dose_interp) + CI , alpha = 0.6, label = "95% CI")

    plt.legend()
    #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\1D analysis\\LQ_model_open_{}.png".format(flask_template.shape), dpi = 1200)
    plt.show()





test_image = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))
Grid = True
if Grid == True:

    survival_grid = survival_analysis(folder, time, mode[2], position,
                                      kernel_size_p[0], dose_path_grid,
                                      template_file_grid, save_path = save_path,
                                      dose = dose, cropping_limits =  cropping_limits_1D, data_extract = True)
    ColonyData_grid, data_grid  = survival_grid.data_acquisition()
    colony_map_grid = survival_grid.Colonymap()


    print(colony_map_grid.shape,row_weight.shape)
    print(len(flask_template))

    print(colony_map_grid.shape)
    num_colonies_grid = np.zeros((len(time),len(dose),len(position),len(flask_template)))
    print(num_colonies_grid.shape)

    """
    Iterating over all rows in colonymap and summing up surviving colonies for
    open field and grid comparison
    """

    for i in range(len(time)):
        #include 10 Gy and compare with extrapolated data for open field
        for j in range(len(dose)):
            for k in range(len(position)):
                for l in range(cropping_limits_1D[0],cropping_limits_1D[1]):
                    m = l - cropping_limits_1D[0]
                    num_colonies_grid[i,j,k,m] = np.sum(colony_map_grid[i,j,k,l])*row_weight[m]


    print(num_colonies_grid.shape)
    print(num_colonies_ctrl.shape)
    print(num_colonies_open.shape)



    """
    Now we want dose profiles to compare survival of OPEN vs GRID, but these demand uncertainty in the dose profiles
    which we will get from uncertainty analysis, which you will perform
    Then when 1D analysis is complete you can write about the method
    Make dose profiles in film_calib, then export them here, remember to use the same cropping
    """

    """
    1D Dose profiles with survival, will only perform this using 5 Gy
    """
    dose_map_open = np.load("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_film_dose_map_reg_Open.npy", allow_pickle = True)
    dose_map_grid = np.load("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_film_dose_map_reg_GRID Stripes.npy")

    # [250:2750, 100:2100]
    #[cropping_limits_1D[0]:cropping_limits_1D[1],cropping_limits_1D[2],cropping_limits_1D[3]:cropping_limits_1D[4]]
    #we need uncertainty of the doses
    dose_map_open = dose_map_open[cropping_limits_1D[0]:cropping_limits_1D[1], 500:1500]
    dose_map_grid = dose_map_grid[cropping_limits_1D[0]:cropping_limits_1D[1], 500:1500]


    """plt.subplot(121)
    plt.imshow(dose_map_grid[250:2750, 500:1500])
    plt.subplot(122)
    plt.imshow(dose_map_open[250:2750, 500:1500])"""

    plt.close()

    """
    Dividing survival colonies into survival bands of x mm width and analysing survival
    within these bands. The larger the band, the more smoothed dose
    """

    dx = 1 #mm
    survival_bands = int(dx*47)  #pixels
    num_bands = flask_template.shape[0]//survival_bands

    new_image_height = survival_bands*num_bands

    split_image_height = np.array([int(i/47) for i in np.arange(0,new_image_height,survival_bands)])
    split_image_height = split_image_height/10
    #remember to sum before finding mean

    """
    Num_colonies is number of colonies per row of the colony map.
    We divide this matrix into bands with thickness dx. E.g.,
    with a 1 mm band we have, in a 1200 dpi image, 47 pixels per band.
    If the original shape of the num_colonies is (t,d,p,1850), we can fit 39
    bands. Of course 1850%47 != 0, and we loose 17 rows when dividing into 1 mm bands.
    We sum over the 47 pixels, then find the mean within all our scanned images
    for time t and position p.
    """


    survival_profile_ctrl = np.mean(np.sum(num_colonies_ctrl[:,:,:,:new_image_height]\
    .reshape(num_colonies_ctrl.shape[0],num_colonies_ctrl.shape[1],num_colonies_ctrl.shape[2],num_bands,survival_bands), axis = 4),axis = (0,2))


    mean_survival_ctrl = np.mean(survival_profile_ctrl)

    survival_profile_open = np.mean(np.sum(num_colonies_open[:,:,:,:new_image_height]\
    .reshape(num_colonies_open.shape[0],num_colonies_open.shape[1],num_colonies_open.shape[2],num_bands,survival_bands),axis = 4),axis = (0,2))/mean_survival_ctrl

    survival_profile_grid = np.mean(np.sum(num_colonies_grid[:,:,:,:new_image_height]\
    .reshape(num_colonies_grid.shape[0],num_colonies_grid.shape[1],num_colonies_grid.shape[2],num_bands,survival_bands),axis = 4),axis = (0,2))/mean_survival_ctrl




    """
    Add confidence
    """

    plt.title("dx = {} mm".format(dx))
    plt.xlabel("Position in flask [mm]")
    plt.ylabel("Survival [a.u.]")
    plt.plot(split_image_height, survival_profile_grid[2], label = "GRID 5 Gy")
    #plt.plot(split_image_height, survival_profile_grid[1],label = "GRID 2 Gy")
    # plt.plot(split_image_height, survival_profile_ctrl[0],label = "Ctrl")
    #plt.plot(split_image_height,survival_profile_open[1],label = "Open 5 Gy")
    #plt.plot(split_image_height,survival_profile_open[0],label = "Open 2 Gy")
    plt.ylim([-0.5,1.5])
    plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\survival_profile.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)
    plt.legend()

    plt.show()


    idx = np.argwhere(colony_map_ctrl[0,0,0] > 0) #finding ideal index to start on
    print(np.argmin(np.sum(idx,axis = 1))) #seems like first colony is located on pixel 56

    image_height = flask_template.shape[0]

    #expanding dims for dose profile function to work
    dose_profile_open = dose_profile2(image_height,dose_map_open)
    dose_profile_grid = dose_profile2(image_height,dose_map_grid)



    image_height_mm = [int(i/47) for i in np.arange(0,image_height,1)]
    image_height_px = np.arange(0,image_height,1)
    plt.plot(image_height_px,dose_profile_open)
    plt.plot(image_height_px,dose_profile_grid)
    plt.close()
    #plt.show()
    print(dose_profile_open.shape, dose_profile_grid.shape)



    survival_profile_ctrl = np.mean(num_colonies_ctrl,axis = (0,2))
    survival_profile_open = np.mean(num_colonies_open,axis = (0,2))
    survival_profile_grid = np.mean(num_colonies_grid,axis = (0,2))

    print(survival_profile_grid.shape)

    plt.plot(image_height_px,survival_profile_grid[2]/mean_ctrl_survival)
    plt.close()

    #remember to unhash this when performing 2D analysis
    survival_grid.registration()

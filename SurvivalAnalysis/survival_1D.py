from survival_analysis4 import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f, ttest_ind, chi2, t
import seaborn as sb
from scipy import stats, optimize
from scipy.interpolate import interp1d
from utils import  data_stacking, design_matrix, data_stacking_2,  dose_profile2
import cv2
from sklearn.model_selection import train_test_split
import pickle
import statsmodels.api as sm




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

    cropping_limits_1D = [200,2500, 70,2100]  #this matches with dose film the most
    #cropping_limits_1D = [200,2000,350,1850]

    flask_template = np.asarray(pd.read_csv(template_file_control))[cropping_limits_1D[0]:cropping_limits_1D[1],cropping_limits_1D[2]:cropping_limits_1D[3]]

    plt.imshow(flask_template)
    plt.close()

    survival_control = survival_analysis(folder, time, mode[0], position,
                       kernel_size_p[0], dose_map_path = None,
                       template_file = template_file_control,
                       save_path = save_path, dose = ctrl_dose, cropping_limits = cropping_limits_1D,
                       data_extract = False)
    ColonyData_control, data_control = survival_control.data_acquisition()

    colony_map_ctrl = survival_control.Colonymap()
    #pooled_SC_ctrl = survival_control.Quadrat()

    #pooled_SC_ctrl = np.reshape(pooled_SC_ctrl[:,0,:], (pooled_SC_ctrl.shape[0],
                          #pooled_SC_ctrl.shape[2], pooled_SC_ctrl.shape[3],pooled_SC_ctrl.shape[4]))
    #mean_SC_ctrl = np.mean(pooled_SC_ctrl)
    #extracting flask template and cropping away edges

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

    print(flask_template.shape)
    #want to include all data for cells
    image_height_px = np.linspace(0,flask_template.shape[0],10)
    #image_height_mm = np.linspace(0,round(flask_template.shape[0]/47,1),flask_template.shape[0]) #mm
    image_height_mm = [round(pos/(47*10),1) for pos in image_height_px]
    image_width_px = np.linspace(0,flask_template.shape[1],10)
    image_width_mm = [round(pos/(47*10),1) for pos in image_width_px]

    print(image_height_mm, image_width_mm)


    fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
    ax[0].set_title("Flask template")
    ax[0].set_yticks(image_height_px, labels = image_height_mm)
    ax[0].set_xticks(image_width_px, labels = image_width_mm)
    ax[0].set_ylabel("y [cm]")
    ax[0].set_xlabel("x [cm]")
    ax[0].imshow(flask_template)
    ax[0].patch.set_edgecolor('black')
    ax[0].patch.set_linewidth('2')

    row_weight = np.zeros(len(flask_template))
    for i in range(len(flask_template)):
        ones = len(flask_template[i,flask_template[i] > 0])
        zeros = len(flask_template[i,flask_template[i] == 0])
        if ones == 0:
            row_weight[i] = 0
        else:
            row_weight[i] = 1 + zeros/ones
    ax[1].set_title("Row weights")
    ax[1].plot(row_weight)
    ax[1].set_xlabel("Position in cell flask [cm]")
    ax[1].set_ylabel("1 + zeros/ones")
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\1D analysis\\weightfactor_{}.png".format(flask_template.shape), dpi = 1200, pad_inches = 0.4)
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
    plt.plot(num_colonies_ctrl[0,0,0],"*")
    plt.close()

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
                                      cropping_limits = cropping_limits_1D, data_extract = False)
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
                    num_colonies_open[i,j,k,m] = np.sum(colony_map_open[i,j,k,l])*row_weight[m]
    #finding mean number of surviving colonies for all times and all positions

    print(num_colonies_open.shape)


    """
    1D LQ modelling
    We do not crop cell flask for 1D analysis
    """
    #for all times and positions we find the total number of colonies
    #we skip 10 Gy because the segmentation algorithm doesnt work for 10 Gy

    n = 2*4 #we have 8 flasks per dose


    print(colony_map_open[:,:-1,:,cropping_limits_1D[0]:cropping_limits_1D[1],cropping_limits_1D[2]:cropping_limits_1D[3]].shape)

    surviving_colonies_open = np.sum(colony_map_open[:,:-1], axis = (3,4))

    surviving_colonies_ctrl = np.sum(colony_map_ctrl, axis = (3,4))



    #surviving_colonies_open = np.sum(colony_map_open[:,:-1,:,cropping_limits_1D[0]:cropping_limits_1D[1],cropping_limits_1D[2]:cropping_limits_1D[3]], axis = (3,4))

    #surviving_colonies_ctrl = np.sum(colony_map_ctrl[:,:,:,cropping_limits_1D[0]:cropping_limits_1D[1],cropping_limits_1D[2]:cropping_limits_1D[3]], axis = (3,4))



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

    # mean_survival_ctrl = np.mean(surviving_colonies_ctrl, axis = 0)
    mean_survival_ctrl = np.mean(surviving_colonies_ctrl, axis = 0)

    combined_survival = np.append(surviving_colonies_ctrl[:,0]/mean_survival_ctrl, #ctrl
                        np.append(surviving_colonies_open[:,0]/mean_survival_ctrl, #2 Gy
                        surviving_colonies_open[:,1]/mean_survival_ctrl)) #5 Gy


    # combined_survival = np.delete(combined_survival,np.argwhere(combined_survival < 0.7)[0,0])
    # dose_axis = np.delete(dose_axis, np.argwhere(combined_survival < 0.7)[0,0])
    # plt.plot(dose_axis,combined_survival, "*")
    # # plt.plot(2,combined_survival[12], "o")
    # plt.show()
    """
    Using statsmodels to fit instead of polyfit
    """

method1 = True
if method1:
        dose_interp = np.linspace(0,10,1000)
        # X = np.array([np.repeat(1, len(dose_axis)), dose_axis, dose_axis**2]).T
        # X_ = np.array([np.repeat(1, len(dose_interp)), dose_interp, dose_interp**2]).T
        X = np.array([dose_axis, dose_axis**2]).T
        X_ = np.array([dose_interp, dose_interp**2]).T

        fit = sm.OLS(np.log(combined_survival), X).fit()

        S_ = fit.params[0] * dose_interp + fit.params[1]*dose_interp**2
        print(fit.summary())
        predictions = fit.get_prediction(X_)
        frame = predictions.summary_frame(alpha=0.05)
        print(frame.mean_ci_lower)
        plt.plot(dose_axis,np.log(combined_survival), "*", label = "data")
        plt.xlabel("Dose [Gy]", fontsize = 12)
        plt.ylabel(r"$S_{irr}/\bar{S}_{ctrl}$", fontsize = 15)
        #plt.plot(dose_interp, frame.obs_ci_lower, "--", c = "darkviolet")
        #plt.plot(dose_interp, frame.obs_ci_upper, "--", label = "P.I. 95%", c = "darkviolet")
        #plt.plot(dose_interp, frame.mean_ci_lower, "--", c = "navy")
        #plt.plot(dose_interp, frame.mean_ci_upper, "--", label = "C.I. 95%", c = "navy")
        plt.fill_between(dose_interp,  frame.mean_ci_lower, frame.mean_ci_upper, color = "navy", label = "C.I. 95%", alpha = 0.6)
        #plt.fill_between(dose_interp,  frame.obs_ci_lower,  frame.obs_ci_upper, color = "darkviolet", label = "P.I. 95%", alpha = 0.6)

        plt.plot(dose_interp, fit.predict(X_),label = r"({:+.3f} $\pm$ {:.3f} D) {:+.3f} $\pm$ {:.3f} $D^2$".format(fit.params[0], fit.bse[0], fit.params[1], fit.bse[1]) )
        #plt.plot(dose_interp, S_, label = r"({:+f} $\pm$ {:f} d) {:+f} $\pm$ {:f} $d^2$".format(fit.params[1], fit.bse[1], fit.params[2], fit.bse[2]))
        plt.legend(fontsize = 10)
        plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\1D analysis\\LQ_model_1D.png", bbox_inches ="tight", pad_inches = 0.1, dpi = 1200)
        plt.show()

method2 = False
if method2:
    coeff,cov = np.polyfit(dose_axis,np.log(combined_survival),full = False,deg = 2, cov = True) #coeff from x^n to intercept
    stderr_residual = np.sqrt(np.sum((np.polyval(coeff, dose_axis) - np.log(combined_survival))**2)/(n-2))#np.sqrt(np.sum((np.polyval(coeff, dose_axis) - combined_survival)**2/(n-2)))

    print(cov)
    print(stderr_residual)



    cov = np.diag(cov)[::-1] #flipping cov so alpha is second element and beta is third
    print("sdfnjksfdnjksfdnj")
    print(cov)
    n = len(combined_survival)
    cov = np.diag(cov[1:]) #only include uncertainty in beta and alpha

    print(cov)
    dose_interp = np.linspace(0,10,1000)

    # chi2 = chi2.ppf(0.95,1)
    # print(chi2)
    #print(chi2.shape)

    """
    For dose = 0, the derivative will become 0, so we need to use
    the second order delta method but dont know how
    """

    dSdp = np.array([dose_interp, dose_interp**2]) #dS/dalpha, dS/dbeta

    cov_S = (dSdp.T.dot(cov).dot(dSdp))# * chi2
    t_crit = stats.t.ppf(0.95, n - 2)

    print(cov_S)

    _,MSE,_,_,_ = np.polyfit(dose_axis,np.log(combined_survival),full = True,deg = 2)

    MSE = MSE/n
    print(MSE)

    sx = np.std(dose_interp)
    mean_x = np.mean(dose_interp)
    #print(combined_survival)
    plt.title("LQ model fitted to normalized log survival data")
    plt.xlabel("Dose")
    plt.ylabel(r"$\log{S_{irr}/S_{ctrl}}$")
    #first element of cov is uncertainty in highest order x
    plt.plot(dose_interp, np.polyval(coeff, dose_interp), label = r"({:+f} $\pm$ {:f} d) {:+f} $\pm$ {:f} $d^2$".format(coeff[1], np.diag(np.sqrt(cov))[0], coeff[0], np.diag(np.sqrt(cov))[1]))
    plt.plot(dose_axis,np.log(combined_survival), "*",label = "data")
    #we need to divide the standard deviation with the square root of number of points per dose because the mean is supposed to be inside the confidence interval 95% of the times.
    #plt.fill_between(dose_interp, np.polyval(coeff,dose_interp) - t_crit * np.sqrt(np.diag(cov_S)), np.polyval(coeff, dose_interp) + t_crit * np.sqrt(np.diag(cov_S)), alpha = 0.5, color = "grey")
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
                                      dose = dose, cropping_limits =  cropping_limits_1D, data_extract = False)
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

    plt.plot(num_colonies_grid[0,0,0], "*")
    plt.close()


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

    #[cropping_limits_1D[0]:cropping_limits_1D[1],cropping_limits_1D[2],cropping_limits_1D[3]:cropping_limits_1D[4]]
    #we need uncertainty of the doses
    #dose_map_open = dose_map_open[cropping_limits_1D[0]:cropping_limits_1D[1], cropping_limits_1D[2]:cropping_limits_1D[3]]
    dose_map_grid = survival_grid.registration()  #ideal dose map cropping limits


    dose_ = 10

    if dose_ == 2:
        dose_map_grid = dose_map_grid[cropping_limits_1D[0]:cropping_limits_1D[1], 800:1600]*2/5
    elif dose_ == 5:
        dose_map_grid = dose_map_grid[cropping_limits_1D[0]:cropping_limits_1D[1], 800:1600]
    elif dose_ == 10:
        dose_map_grid = dose_map_grid[cropping_limits_1D[0]:cropping_limits_1D[1], 800:1600]*10/5


    #print(dose_map_grid.shape)

    #plt.imshow(dose_map_grid)
    plt.close()

    """
    Dividing survival colonies into survival bands of x mm width and analysing survival
    within these bands. The larger the band, the more smoothed dose
    """

    dx = 1 #mm  similar to our Poisson regression
    survival_bands = int(dx*47)  #pixels
    num_bands = flask_template.shape[0]//survival_bands

    new_image_height = survival_bands*num_bands

    split_image_height = np.arange(0,new_image_height,survival_bands)
    split_image_height = split_image_height/(47*10)
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
    #finding mean dose within the bands


    dose = np.mean(dose_map_grid[:new_image_height], axis = 1)
    #plt.plot(np.arange(0,new_image_height, 1)/(47*10), np.log(dose/np.max(dose)))
    # plt.show()

    #find mean dose in each row, then divide into bands
    tmp_profile =  np.mean(dose_map_grid[:new_image_height],axis = 1).reshape(num_bands,survival_bands)
    #find mean dose within all those bands, finding essentially the mean of the mean
    dose_profile = np.mean(tmp_profile, axis = 1)


    print(split_image_height.shape)

    """
    Splitting survival maps into survival bands, and summing up surviving colonies within these bands
    """
    survival_profile_ctrl = np.mean(np.sum(num_colonies_ctrl[:,:,:,:new_image_height]\
    .reshape(num_colonies_ctrl.shape[0],num_colonies_ctrl.shape[1],num_colonies_ctrl.shape[2],num_bands,survival_bands), axis = 4),axis = (0,2))

    # mean_survival_ctrl = np.mean(survival_profile_ctrl)

    survival_profile_open = np.mean(np.sum(num_colonies_open[:,:,:,:new_image_height]\
    .reshape(num_colonies_open.shape[0],num_colonies_open.shape[1],num_colonies_open.shape[2],num_bands,survival_bands),axis = 4),axis = (0,2))

    survival_profile_grid = np.mean(np.sum(num_colonies_grid[:,:,:,:new_image_height]\
    .reshape(num_colonies_grid.shape[0],num_colonies_grid.shape[1],num_colonies_grid.shape[2],num_bands,survival_bands),axis = 4),axis = (0,2))

    survival_grid_stderr = np.std(np.sum(num_colonies_grid[:,:,:,:new_image_height]\
    .reshape(num_colonies_grid.shape[0],num_colonies_grid.shape[1],num_colonies_grid.shape[2],num_bands,survival_bands), axis = 4), axis = (0,2))/(np.sqrt(num_colonies_grid.shape[0] + num_colonies_grid.shape[2]))


    plt.imshow(num_colonies_grid[:,:,:,:new_image_height]\
    .reshape(num_colonies_grid.shape[0],num_colonies_grid.shape[1],num_colonies_grid.shape[2],num_bands,survival_bands)[0,0,0])
    plt.close()

    survival_grid_stderr = survival_grid_stderr / np.mean(survival_profile_ctrl)
    survival_profile_open= survival_profile_open / np.mean(survival_profile_ctrl)
    survival_profile_grid = survival_profile_grid / np.mean(survival_profile_ctrl)
    survival_profile_ctrl = survival_profile_ctrl / np.mean(survival_profile_ctrl)



    """
    Trying the non vectorized method, but I think the result is the same......
    Yes that was true
    """

    """
    survival_profile_ctrl = np.zeros((1,len(split_image_height)))
    survival_profile_open = np.zeros((2,len(split_image_height))) #2 5 Gy
    survival_profile_grid = np.zeros((3,len(split_image_height))) #2 5 10 Gy
    survival_grid_stderr = np.zeros((3,len(split_image_height)))

    idx = 0
    for i in range(survival_bands,new_image_height+1, survival_bands):
        print(i, new_image_height)
        survival_profile_ctrl[:,idx] = np.mean(np.sum(num_colonies_ctrl[:,:,:,i-survival_bands:i],axis = 3),axis = (0,2))
        survival_profile_open[:,idx] = np.mean(np.sum(num_colonies_open[:,:,:,i-survival_bands:i],axis = 3),axis = (0,2))
        survival_profile_grid[:,idx] = np.mean(np.sum(num_colonies_grid[:,:,:,i-survival_bands:i],axis = 3),axis = (0,2))
        #stderr of survival within each band
        survival_grid_stderr[:,idx] = np.std(np.sum(num_colonies_grid[:,:,:,i-survival_bands:i], axis = 3),axis = (0,2))/np.sqrt(num_colonies_grid.shape[0]+ num_colonies_grid.shape[2])
        idx += 1

    survival_grid_stderr = survival_grid_stderr / np.mean(survival_profile_ctrl)
    survival_profile_open= survival_profile_open / np.mean(survival_profile_ctrl)
    survival_profile_grid = survival_profile_grid / np.mean(survival_profile_ctrl)
    survival_profile_ctrl = survival_profile_ctrl / np.mean(survival_profile_ctrl)"""


    """
    Add prediction survival, using the LQ model with coeff[0,1]
    Insert dose profile with coeff into log lq
    """


    """
    Add confidence
    """

    if method1:
        """
                        mean   mean_se  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
        0  -0.264276  0.041556      -0.350695      -0.177856     -0.512226     -0.016326
        1  -0.068250  0.030350      -0.131365      -0.005134     -0.309070      0.172570
        2  -0.063756  0.030094      -0.126340      -0.001172     -0.304437      0.176926
        3  -0.062470  0.030033      -0.124926      -0.000013     -0.303118      0.178178
        4  -0.061718  0.029999      -0.124106       0.000669     -0.302349      0.178912
        ..       ...       ...            ...            ...           ...           ...
        95 -0.429416  0.034493      -0.501148      -0.357683     -0.672636     -0.186195
        96 -0.424993  0.034678      -0.497110      -0.352876     -0.668327     -0.181658
        97 -0.264669  0.041550      -0.351077      -0.178261     -0.512615     -0.016723
        98 -0.074574  0.030800      -0.138625      -0.010522     -0.315641      0.166494
        99 -0.069516  0.030432      -0.132803      -0.006229     -0.310381      0.171349
        """
        print(dose_profile.shape)
        # X_dose = np.array([np.repeat(1,len(dose_profile)), dose_profile, dose_profile**2]).T
        X_dose = np.array([dose_profile, dose_profile**2]).T

        predframe = fit.get_prediction(X_dose).summary_frame(alpha = .05)
    
        #not pred.mean for some reason
        predicted = np.exp(fit.predict(X_dose))

        #print(predicted_SC)
        # S_ = predicted.
    elif method2:
        predicted = np.polyval(coeff,dose_profile)

    if dose_ == 2:
        observed = survival_profile_grid[0] #2 Gy
    elif dose_ == 5:
        observed = survival_profile_grid[1] #5 Gy
    elif dose_ == 10:
        observed = survival_profile_grid[2] #10 Gy




     #print(np.exp(observed) - t.ppf(0.95, len(dose_profile))*np.exp(survival_grid_stderr[1]))


    diff = np.abs(predicted - observed)/((predicted + observed)/2) #(np.exp(predicted) - np.exp(observed))/np.exp(observed)

    fig,ax = plt.subplots(figsize = (8,6))
    ax.set_title("dx = {} mm".format(dx), fontsize = 15)
    ax.set_xlabel("Position in flask [cm]", fontsize = 15)
    ax.set_ylabel(r"$S_{irr}/\bar{S}_{ctrl}$", fontsize = 15)

    if dose_ == 2:
        ax.plot(split_image_height, observed, label = "GRID 2 Gy observed", color = "navy")
        ax.plot(split_image_height,predicted, "o-", label = "GRID 2 Gy predicted", color = "darkorange")
        ax.fill_between(split_image_height, np.exp(predframe.mean_ci_lower), np.exp(predframe.mean_ci_upper), alpha = 0.6, color = "orange", label = "95% C.I. pred")
        ax.fill_between(split_image_height, observed - t.ppf(0.95, len(dose_profile))*survival_grid_stderr[0],
                                              observed + t.ppf(0.95, len(dose_profile))*survival_grid_stderr[0], alpha = 0.6, color = "deepskyblue", label = "95% C.I. obs.")
    elif dose_ == 5:
        ax.plot(split_image_height, observed, label = "GRID 5 Gy observed", color = "navy")
        ax.plot(split_image_height,predicted, "o-", label = "GRID 5 Gy predicted", color = "darkorange")
        ax.fill_between(split_image_height, np.exp(predframe.mean_ci_lower), np.exp(predframe.mean_ci_upper), alpha = 0.6, color = "orange", label = "95% C.I. pred")
        ax.fill_between(split_image_height, observed - t.ppf(0.95, len(dose_profile))*survival_grid_stderr[1],
                                              observed + t.ppf(0.95, len(dose_profile))*survival_grid_stderr[1], alpha = 0.6, color = "deepskyblue", label = "95% C.I. obs.")
    elif dose_ == 10:
        ax.plot(split_image_height, observed, label = "GRID 10 Gy observed", color = "navy")
        ax.plot(split_image_height,predicted, "o-", label = "GRID 10 Gy predicted", color = "darkorange")
        ax.fill_between(split_image_height, np.exp(predframe.mean_ci_lower), np.exp(predframe.mean_ci_upper), alpha = 0.6, color = "orange", label = "95% C.I. pred")
        ax.fill_between(split_image_height, observed - t.ppf(0.95, len(dose_profile))*survival_grid_stderr[2],
                                              observed + t.ppf(0.95, len(dose_profile))*survival_grid_stderr[2], alpha = 0.6, color = "deepskyblue", label = "95% C.I. obs.")

    ax.legend(fontsize = 12, loc = "upper left")
    ax2 = ax.twinx()
    ax2.plot(split_image_height,diff, "--", label = "RPD")
    ax2.set_ylabel("RPD")

    #plt.plot(split_image_height, np.log(dose_profile/np.max(dose_profile)), label = "dose")
    #plt.plot(split_image_height, survival_profile_grid[1],label = "GRID 2 Gy")
    # plt.plot(split_image_height, survival_profile_ctrl[0],label = "Ctrl")
    #plt.plot(split_image_height,survival_profile_open[1],label = "Open 5 Gy")
    #plt.plot(split_image_height,survival_profile_open[0],label = "Open 2 Gy")
    #plt.ylim([-0.5,1.5])
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\survival_profile_2.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)
    ax2.legend(fontsize = 12, loc = "upper right")

    if dose_ == 2:
        fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\1D analysis\\survival_profile_2Gy_1dx.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 300)
    elif dose_ == 5:
        fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\1D analysis\\survival_profile_5Gy_1dx.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 300)
    elif dose_ == 10:
        fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\1D analysis\\survival_profile_10Gy_1dx.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 300)


    plt.show()

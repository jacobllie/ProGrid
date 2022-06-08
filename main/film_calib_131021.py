from film_calibration_tmp9 import film_calibration
import numpy as np
import matplotlib.pyplot as plt
from dose_profile import dose_profile
from dose_map_registration import phase_correlation#image_reg
import cv2
from utils import dose_fit_error
from scipy.stats import t

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Calibration"
background = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Background"
control = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Control"
test_image = "EBT3_Calib_131021_Xray220kV_01Gy1_001.tif"
background_image = "EBT3_Calib_131021_Xray220kV_black_001.tif"
control_image = "EBT3_Calib_131021_Xray220kV_00Gy1_001.tif"
reg_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\registered_images"
#measurement_crop = [30,500,200,220]#[40,700,195,250] #for best profiles
# measurement_crop = np.array([50,2700,70,1900])//4
measurement_crop = [40,690,205,250] #for best profiles

# measurement_crop = [15,717 ,15 ,492] #not for dose profiles but survival 2D


#measurement_crop = np.array([200,2000,350,1850])//4
film_calib = film_calibration(folder, background, control, test_image, background_image,
                              control_image, reg_path, measurement_crop, calibration_mode = True, image_registration = False, registration_save = False)

film_calib.image_acquisition()


films_per_dose = 8  #[0,1,2,3,4,5]
dose_axis = np.array([0,0.1,0.2,0.5,10,1,2,5])
ROI_size = 4 #mm
num_fitting_params = 3
#Finding netOD values
netOD, sigma_img, sigma_bckg, sigma_ctrl, dOD = film_calib.calibrate(ROI_size, films_per_dose)

"""
Trying to fit all points
"""
method1 = True
if method1:

    fitting_param, param_cov = film_calib.EBT_fit(np.repeat(dose_axis,netOD.shape[2]), num_fitting_params, OD = np.ravel(netOD[2]), model_type = 1)

    print(fitting_param.shape)

    print(np.max(netOD[2]))

    OD_interp = np.linspace(0,np.max(netOD[2]),100)
    estimated_dose = film_calib.EBT_model(OD_interp,fitting_param,model_type = 1)

    plt.plot(estimated_dose,OD_interp, label = r"fit  ($%.3f \pm %.3f) \cdot netOD + (%.3f \pm %.3f) \cdot netOD^{(%.3f \pm %.3f)}$"%(fitting_param[0],np.sqrt(param_cov[0]),
    fitting_param[1],np.sqrt(param_cov[1]),fitting_param[2], np.sqrt(param_cov[2])), color = "r")





#np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\netOD_calib\\netOD_131021.npy",netOD)

channels = ["BLUE", "GREEN", "RED", "GREY"]
colors = ["b","g","r","grey"]
for i in range(netOD.shape[0]):
    for j in range(netOD.shape[1]):
        if j == 0:
            plt.plot(np.repeat(dose_axis[j], netOD.shape[2]), netOD[i,j], "*", color = colors[i], label = channels[i])
        else:
            plt.plot(np.repeat(dose_axis[j], netOD.shape[2]), netOD[i,j], "*", color = colors[i])

plt.legend()
plt.xlabel("Dose [Gy]")
plt.ylabel("netOD")
plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\131021\\no_split_OD_fit.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)

plt.close()


"""
Now make dose profile with no OD split
"""




"""
Here we make the stacked dose array, in our case we have 8 films per dose.
"""
method2 = False
if method2:

    #removing low response OD
    netOD_high = (np.delete(np.ravel(netOD[2]), [6*8 + 6, 6*8 + 7]))
    # netOD_low = list(np.delete(np.ravel(netOD[2]), [6*8, 6*8 + 1, 6*8+2,6*8+3,6*8+4,6*8+5]))


    # dose_axis_octo_low = np.array([0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
    #                 0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
    #                 10,10,10,10,10,10,10,10,1,1,1,1,1,1,1,1,2,2,5,5,5,5,5,5,5,5])
    # plt.plot(dose_axis_octo_low,netOD_low, "*")
    # plt.show()



    dose_axis_octo = np.array([0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
                    0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                    10,10,10,10,10,10,10,10,1,1,1,1,1,1,1,1,2,2,2,2,2,2,5,5,5,5,5,5,5,5])

    plt.plot(dose_axis_octo, netOD_high, "*")
    plt.close()

    #plt.plot(dose_axis_octo, netOD_high,"*")
    #plt.show()


method3 = True
if method3:

    bandwidth_red_channel = np.array([0.0005, 0.003, 0.003,0.0005, 0.003, 0.003, 0.005, 0.003])
    low_response_OD, high_response_OD, low_res_dose, high_res_dose, bandwidth_red_channel, no_split_OD = \
    film_calib.netOD_split(dose_axis, bandwidth_red_channel, bandwidth_stepsize = 0.0001, channel = "RED", no_split = True)


    """print(low_res_dose)
    print(high_res_dose)
    print(low_response_OD.shape)
    print(low_res_dose.shape)
    print(high_response_OD.shape)
    print(high_res_dose.shape)"""

    netOD_3108_low = np.load("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\low_res_netOD_310821.npy")
    netOD_3108_high = np.load("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\high_res_netOD_310821.npy")
    dose_3108_low = np.load("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\low_res_dose_310821.npy")
    dose_3108_high = np.load("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\high_res_dose_310821.npy")

    idx_low = [5,6,7,8,9,10,13,14]

    print(netOD_3108_low[idx_low])
    idx_high = [32,33,34,35,36,37,38,39,40,41]
    low_response_OD = np.append(low_response_OD, netOD_3108_low[idx_low])
    low_response_OD = np.append(low_response_OD, netOD[2,7]) #adding 5 Gy netOD to low response
    low_response_OD = np.append(low_response_OD, netOD[2,4]) #adding 10 Gy netOD to low response
    low_response_OD = np.append(low_response_OD, netOD[2,3]) #adding all netOD from 0.5 because we're not sure which category it belongs to
    low_res_dose =    np.append(low_res_dose,dose_3108_low[idx_low])
    low_res_dose = np.append(low_res_dose, np.repeat(5,8))
    low_res_dose = np.append(low_res_dose, np.repeat(10,8))
    low_res_dose = np.append(low_res_dose, np.repeat(0.5,8))
    high_response_OD = np.append(high_response_OD,netOD_3108_high[idx_high])
    high_response_OD = np.append(high_response_OD, netOD[2,1])
    high_response_OD = np.append(high_response_OD, netOD[2,2])
    high_response_OD = np.append(high_response_OD, netOD[2,3])
    high_response_OD = np.append(high_response_OD, netOD[2,5])
    high_res_dose = np.append(high_res_dose,dose_3108_high[idx_high])
    high_res_dose = np.append(high_res_dose, np.repeat(0.1,8))
    high_res_dose = np.append(high_res_dose, np.repeat(0.2,8))
    high_res_dose = np.append(high_res_dose, np.repeat(0.5,8))
    high_res_dose = np.append(high_res_dose, np.repeat(1,8))



    """
    print(low_response_OD.shape)
    print(low_res_dose.shape)
    print(high_response_OD.shape)
    print(high_res_dose.shape)"""
    #low_response_OD = np.zeros(())

    print(low_res_dose)
    print(high_res_dose)


    plt.plot(low_res_dose, low_response_OD, "*", label = "low response")
    plt.plot(high_res_dose, high_response_OD, "*", label = "high response")
    plt.legend()
    plt.show()


    """no_split_dose = np.repeat(dose_axis[np.unique(np.argwhere(no_split_OD != 0)[:,0])], len(no_split_OD))
    print("sbhfdshb")
    print(no_split_dose)"""



    """
    We need to add OD to low or high based on how they were placed in the netod plot
    from 3108
    low : 0, 2, 5, 10
    high: 0, 0.1,0.2
    """

    fitting_param_low, fitting_param_high, param_var_low, param_var_high, fit_low, fit_high, = \
    film_calib.EBT_fit(dose_axis, num_fitting_params, low_response_OD, high_response_OD,low_res_dose,
                       high_res_dose,model_type = 1)




    """plt.plot(film_calib.EBT_model(np.linspace(0,np.max(new_low_response_OD),100), fitting_param_low, model_type = 1),
             np.linspace(0,np.max(new_low_response_OD),100),  label = r"Low response fit  $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_low[0],
             fitting_param_low[1],fitting_param_low[2]))
    plt.plot(film_calib.EBT_model(np.linspace(0,np.max(new_high_response_OD),100), fitting_param_high, model_type = 1),
             np.linspace(0,np.max(new_high_response_OD),100), label = r"High response fit $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_high[0],
             fitting_param_high[1],fitting_param_high[2]))"""

    plt.plot(film_calib.EBT_model(np.linspace(0,max(low_response_OD),100),fitting_param_low,model_type = 1),
             np.linspace(0,np.max(low_response_OD),100), label = r"Low response fit  $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_low[0],
             fitting_param_low[1],fitting_param_low[2]))
    plt.plot(film_calib.EBT_model(np.linspace(0,np.max(high_response_OD),100), fitting_param_high, model_type = 1),
             np.linspace(0,np.max(high_response_OD),100), label = r"High response fit $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_high[0],
             fitting_param_high[1],fitting_param_high[2]))
    plt.plot(low_res_dose, low_response_OD, "*", label = "low response")
    plt.plot(high_res_dose, high_response_OD, "*", label = "high response")
    #plt.plot(no_split_dose, no_split_OD[no_split_OD !=0], "*", label = "no split")
    plt.legend()
    plt.show()


    # new_dose_axis = [0,0.1,0.2,0.5,1,2,5,10]
    #fitting_param_low, fitting_param_high, param_var_low, param_var_high, fit_low, fit_high = film_calib.EBT_fit(dose_axis,num_fitting_params, OD = netOD_high, model_type  = 1)



    # sys.exit()

    # print(np.argwhere(no_split_OD != 0))
    """
    print(np.shape(dose_axis_octo), np.shape(netOD_high))
    fitting_param_high, res_high = film_calib.EBT_fit(dose_axis_octo, num_fitting_params, OD = netOD_high, model_type = 1)




    OD_axis = np.linspace(0,np.max(netOD_high),100)

    plt.plot(film_calib.EBT_model(OD_axis, fitting_param_high, model_type = 1), OD_axis, label = r" high response fit: D = %.3f $\cdot$ netOD $\cdot$ %.4f $\cdot$ netOD$^{%.4f}$"%(fitting_param_high[0],fitting_param_high[1], fitting_param_high[2]))

    #Same procedure only now we use all the datapoints in our fit


    netOD_ravelled = (np.ravel(netOD[2]))
    dose_axis_octo = np.array([0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
                    0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                    10,10,10,10,10,10,10,10,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5])



    fitting_params, res = film_calib.EBT_fit(dose_axis_octo, num_fitting_params, OD = netOD_ravelled, model_type = 1)

    OD_axis = np.linspace(0,np.max(netOD_ravelled),100)

    print(film_calib.EBT_model(OD_axis, fitting_params, model_type = 1).shape)

    plt.plot(film_calib.EBT_model(OD_axis, fitting_params, model_type = 1), OD_axis,"*",
                        label = r" fit: D = %.3f $\cdot$ netOD $\cdot$ %.4f $\cdot$ netOD$^{%.4f}$"%(fitting_params[0],fitting_params[1], fitting_params[2]))

    #We see that the result doesnt change much when fitting all points or when removing low response

    plt.xlabel("Dose [Gy]")
    plt.ylabel("netOD")
    for i in range(netOD.shape[2]):
        if i == 0:
            plt.plot(dose_axis, netOD[2,:,i],"p",color = "red", label = "red channel netOD")
            plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue", label = "blue channel netOD")
            plt.plot(dose_axis, netOD[1,:,i],".",color = "green", label = "green channel netOD")
            plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey", label = "grey channel netOD")
        else:
            plt.plot(dose_axis, netOD[2,:,i],"p",color = "red")
            plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue")
            plt.plot(dose_axis, netOD[1,:,i],".",color = "green")
            plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey")


    plt.legend()
    plt.show()"""


test_image_holes = "EBT3_Holes_131021_Xray220kV_5Gy1_001.tif"
holes_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Measurements"
#measurement_crop= [50, 650, 210, 260]

image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Measurements\\EBT3_Holes_131021_Xray220kV_5Gy1_001.tif", 1)

# plt.imshow(image[10:722, 10:497])
# plt.show()

# film_measurements_holes = film_calibration(holes_folder, background, control, test_image_holes, background_image,
#                               control_image, reg_path, measurement_crop, calibration_mode = False, image_registration = False, grid = True)
film_measurements_grid = film_calibration(holes_folder, background, control, test_image_holes, background_image,
                              control_image, reg_path, measurement_crop, calibration_mode = False, image_registration = False,registration_save = False, grid = True)
# film_measurements_holes.image_acquisition()
film_measurements_grid.image_acquisition()

# netOD_holes,_,_,_,_ = film_measurements_holes.calibrate(ROI_size, films_per_dose)
netOD_grid,_,_,_,dOD_grid = film_measurements_grid.calibrate(ROI_size, films_per_dose)

if method1:
    dose_grid = film_measurements_grid.EBT_model(netOD_grid, fitting_param, model_type = 1)
    dose_profile_grid = dose_profile(dose_grid.shape[1],dose_grid)

    image_height_cm = dose_grid.shape[1]/(11.81*10)  #11.81/10 pixels per cm for 300 dpi image

    image_height = np.linspace(0,image_height_cm,dose_grid.shape[1])
    print(dose_profile_grid.shape)
    for i in range(dose_profile_grid.shape[0]):
        plt.plot(image_height, dose_profile_grid[i], color = "purple")
    plt.title("Dose profiles 5 Gy nominal dose dotted GRID")
    plt.xlabel("Position in cell flask [cm]")
    plt.ylabel("Dose[Gy]")
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\131021\\no_split_OD_dose_profile.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 300)
    plt.show()
if method3:

    """
    from 310821
    fitting param low response       fitting param high response
    [ 7.23795233 50.37465016  2.62488359] [ 6.20537405 50.8145797   2.95422661]
    """

    bandwidth_grid = 0.005
    bandwidth_stepsize = 0.0001
    # low_img_grid, high_img_grid = film_measurements_holes.netOD_split(dose_axis, bandwidth_grid, bandwidth_stepsize)
    low_img_grid, high_img_grid = film_measurements_grid.netOD_split(dose_axis, bandwidth_grid, bandwidth_stepsize)

    dOD_low_grid = dOD_grid[low_img_grid]
    dOD_high_grid = dOD_grid[high_img_grid]

    low_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[low_img_grid],fitting_param_low, model_type = 1)
    high_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[high_img_grid],fitting_param_high, model_type = 1)


    mean_dose_grid = np.mean(np.concatenate((low_res_dose_grid, high_res_dose_grid)), axis = 0)

    plt.imshow(mean_dose_grid)
    plt.show()

    # np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\mean_film_dose_map\\mean_dose_map_grid_.npy", mean_dose_grid)



    image_height_cm = low_res_dose_grid.shape[1]/(11.81*10)  #11.81/10 pixels per cm for 300 dpi image

    image_height = np.linspace(0,image_height_cm,low_res_dose_grid.shape[1])


    print(low_res_dose_grid.shape)


    low_mean_grid = dose_profile(low_res_dose_grid.shape[1],low_res_dose_grid)

    high_mean_grid =  dose_profile(high_res_dose_grid.shape[1],high_res_dose_grid)




    plt.title("using fitting parameters from 131021")
    plt.xlabel("Cell flask Position [cm]")
    plt.ylabel("Dose [Gy]")
    for i in range(len(low_mean_grid)):
        if i ==0:
            plt.plot(image_height,low_mean_grid[i],color = "b",label = "low response dose grid")
        else:
            plt.plot(image_height,low_mean_grid[i],color = "b")

    for i in range(len(high_mean_grid)):
        if i == 0 :
            plt.plot(image_height,high_mean_grid[i],color = "r",label = "high response dose grid")
        else:
            plt.plot(image_height,high_mean_grid[i],color = "r")

    plt.legend()
    plt.show()





    """for i in range(len(netOD_holes)):
        fig = plt.figure()
        st = fig.suptitle("image {}".format(i), fontsize="x-large")
        plt.imshow(netOD_holes[i])
        # plt.imshow(GREY_chan_cropped[7])
        plt.show(block=False)
        plt.pause(0.0001)
    plt.show()"""





    """
    We try using fitting parameters from calibration curve obtained 310821
    """
    # fitting param low response       fitting param high response
    # [ 7.26523181 50.57992378  2.62967167] [ 6.22143062 50.83472755  2.95669501]
    # paramer variance [a,b,c]
    # [ 0.75801491 31.90390987  0.04410476] [0.0276693  1.56386531 0.00238728]

    fitting_param_low = [ 7.26523181, 50.57992378,  2.62967167]
    fitting_param_high = [ 6.22143062, 50.83472755,  2.95669501] #newest fitting data for RED channel
    param_var_low = [ 0.75801491, 31.90390987,  0.04410476 ] #covariance of fitting parameters
    param_var_high = [0.0276693,  1.56386531, 0.00238728]


    low_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[low_img_grid],fitting_param_low, model_type = 1)
    high_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[high_img_grid],fitting_param_high, model_type = 1)

    mean_dose_map = np.mean(np.concatenate((low_res_dose_grid, high_res_dose_grid)), axis = 0) #mean dose map

    print(mean_dose_map.shape)



    low_mean_grid= dose_profile(low_res_dose_grid.shape[1],low_res_dose_grid)

    high_mean_grid =  dose_profile(high_res_dose_grid.shape[1],high_res_dose_grid)


    mean_grid_dose = np.mean(np.concatenate((low_mean_grid,high_mean_grid)),axis = 0) #mean dose profile

    #will use this for peak and valley dose estimation
    fit_std_grid_profile_low = np.sqrt(np.sum(dose_fit_error(netOD_grid[low_img_grid], dOD_low_grid, param_var_low, fitting_param_low)**2, axis = 2))/low_res_dose_grid.shape[2]
    fit_std_grid_profile_high = np.sqrt(np.sum(dose_fit_error(netOD_grid[high_img_grid], dOD_high_grid, param_var_high, fitting_param_high)**2, axis = 2))/high_res_dose_grid.shape[2]

    """confidence interval"""

    std_fit_grid_low = np.sqrt(np.sum(dose_fit_error(netOD_grid[low_img_grid], dOD_low_grid, param_var_low, fitting_param_low)**2,axis = 0))/len(low_img_grid)
    std_fit_grid_high = np.sqrt(np.sum(dose_fit_error(netOD_grid[high_img_grid], dOD_high_grid, param_var_high, fitting_param_high)**2,axis = 0))/len(high_img_grid)

    #finding pooled standard error of combination

    # mean_dose_map_error_grid = np.sqrt((std_fit_grid_low**2*(len(low_img_grid) - 2) + std_fit_grid_high**2*(len(high_img_grid) - 2))/(len(low_img_grid) + len(high_img_grid) - 2))
    #mean_dose_map_error_grid = np.sqrt(std_fit_grid_low**2 + std_fit_grid_high**2)/2
    mean_dose_map_error_grid = np.sqrt(std_fit_grid_low**2 + std_fit_grid_high**2)

    stderr_grid_profile = np.sqrt(np.sum(mean_dose_map_error_grid**2, axis = 1))/mean_dose_map_error_grid.shape[1] #finding mean std error of each row


    """finding peak and valley dose in dotted grid"""

    dose_grid = np.concatenate((low_mean_grid, high_mean_grid))

    grid_mean_std_low = np.std(low_mean_grid, axis = 0)/np.sqrt(len(low_mean_grid))
    grid_mean_std_high = np.std(high_mean_grid, axis = 0)/np.sqrt(len(high_mean_grid))

    print(grid_mean_std_low.shape)


    #this is combined with the standard error from the fit
    grid_mean_std =  np.sqrt(grid_mean_std_low**2 + grid_mean_std_high**2)

    #this conf grid includes standard deviation from finding
    conf_grid = t.ppf(0.95, len(low_img_grid) + len(high_img_grid) - 2) * np.sqrt(grid_mean_std**2 + stderr_grid_profile**2)

    mean_dose_grid = np.mean(dose_grid, axis = 0)


    d95 = np.max(mean_dose_grid)*0.95  #dividing by 0.8 because OD is opposite to dose
    print(d95)

    d5 = np.min(mean_dose_grid)*1.05
    print(d5)


    peak_doses = {}
    doses_fit_std_peak = {}
    valley_doses = {}
    doses_fit_std_valley = {}
    for i in range(len(dose_grid)):
        peak_idx = dose_grid[i] > d95
        valley_idx = dose_grid[i] < d5

        peak_tmp = [image_height[peak_idx],dose_grid[i][peak_idx]]
        valley_tmp = [image_height[valley_idx],dose_grid[i][valley_idx]]

        doses_fit_std_peak[i] = grid_mean_std[peak_idx]
        doses_fit_std_valley[i] = grid_mean_std[valley_idx]

        """if i < len(low_mean_grid):
            #the first x profiles are low response
            doses_fit_std_peak[i] = fit_std_grid_profile_low[i][peak_idx]
            doses_fit_std_valley[i] = fit_std_grid_profile_low[i][valley_idx]
        else:
            #the last y profiles are high response we need to take i - len(low_mean_grid) for correct index
            doses_fit_std_peak[i] = fit_std_grid_profile_high[i - len(low_mean_grid)][peak_idx]
            doses_fit_std_valley[i] = fit_std_grid_profile_high[i - len(low_mean_grid)][valley_idx]"""


        print(np.shape(peak_tmp),np.shape(valley_tmp))

        peak_doses[i] = peak_tmp #pos, dose
        valley_doses[i] = valley_tmp

        if i == len(dose_grid) - 1:
            plt.plot(peak_doses[i][0], peak_doses[i][1],"o" ,c = "b",label = "peak")
            plt.plot(valley_doses[i][0],valley_doses[i][1],"o",c = "r",label  ="valley")
        else:
            plt.plot(peak_doses[i][0], peak_doses[i][1],"o" ,c = "b")
            plt.plot(valley_doses[i][0],valley_doses[i][1],"o",c = "r")
    plt.legend()
    plt.xlabel("Position in flask [cm]")
    plt.ylabel("Dose")
    plt.show()
    # print(np.shape(peak_doses), np.shape(peak_pos))

    """for sentence in text:
        for word in sentence:
        [word for sentence in text for word in sentence]

        np.mean(list)
        for i in range(len(dose_grid)):
            for list in peak_doses[i]"""


    """
    Finding the mean of the mean to have mean dose in peak and valley
    """


    #stacking all datapoints regardless of profile
    peak = [list for i in range(len(dose_grid)) for list in peak_doses[i][1]]
    valley = [list for i in range(len(dose_grid)) for list in valley_doses[i][1]]

    print("mean peak dose   mean peak stderr    mean valley dose    mean valley stderr")


    peak_fit_std = np.array([list for i in range(len(dose_grid)) for list in doses_fit_std_peak[i]])
    valley_fit_std = np.array([list for i in range(len(dose_grid)) for list in doses_fit_std_valley[i]])

    grid_peak_conf = t.ppf(0.95, len(peak))*np.sqrt((np.std(peak)/np.sqrt(len(peak)))**2 + (np.sqrt(np.sum(peak_fit_std**2)/len(peak)))**2)
    grid_valley_conf = t.ppf(0.95, len(valley))*np.sqrt((np.std(valley)/np.sqrt(len(valley)))**2 + (np.sqrt(np.sum(valley_fit_std**2)/len(valley)))**2)

    # print(np.mean(peak), np.std(peak)/np.sqrt(len(peak)), np.mean(valley), np.std(valley)/np.sqrt(len(valley)))
    print(np.mean(peak),[np.mean(peak) - grid_peak_conf,np.mean(peak) + grid_peak_conf],
          np.mean(valley), [np.mean(valley) - grid_valley_conf, np.mean(valley) + grid_valley_conf])


    plt.title("GRID dots with fitting parameters from 31.08.21")
    plt.plot(image_height, mean_grid_dose, label = "GRID 5 Gy")
    plt.fill_between(image_height, mean_grid_dose - conf_grid, mean_grid_dose + conf_grid, alpha = 0.8, color = "cyan", label = "95% CI GRID")
    plt.legend()
    plt.xlabel("Position in cell flask [cm]")
    plt.ylabel("Dose [Gy]")
    plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\131021\\dose_profile_grid.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)
    plt.show()

    # np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\mean_film_dose_map\\mean_dose_grid_1D.npy", mean_dose_grid)


    plt.title("Using fitting parameters from 310821")
    plt.xlabel("Cell flask Position [cm]")
    plt.ylabel("Dose [Gy]")
    for i in range(len(low_mean_grid)):
        if i ==0:
            plt.plot(image_height,low_mean_grid[i],color = "b",label = "low response dose grid")
        else:
            plt.plot(image_height,low_mean_grid[i],color = "b")

    for i in range(len(high_mean_grid)):
        if i == 0 :
            plt.plot(image_height,high_mean_grid[i],color = "r",label = "high response dose grid")
        else:
            plt.plot(image_height,high_mean_grid[i],color = "r")

    plt.legend()
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\131021\\dose_profile_params_310821_all_films.png", pad_inches = 0.1, bbox_inches = "tight", dpi = 1200)
    plt.show()
    # dose_map = film_measurements_holes.EBT_model(netOD_holes ,fitting_params[0],fitting_params[1],fitting_params[2])


    """
    We take a small area in the dose map of each film and find mean dose. Then we split
    the doses into high and low response, and apply the high and low fit from last
    experiment.
    """

    """
    from sklearn.neighbors import KernelDensity
    from scipy.signal import argrelextrema
    import sys

    np.set_printoptions(threshold=sys.maxsize)
    bandwidth = 0.05
    dose_samples = np.mean(dose_map[:,20:40,:40],axis = (1,2)).reshape(-1,1)
    print(dose_samples.shape)

    low_res = []
    high_res = []

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dose_samples)
    s = np.linspace(0,max(dose_samples),100)
    kde_scores = kde.score_samples(s)
    mi = argrelextrema(kde_scores, np.less)[0]
    print(mi)

    low_res_idx = np.argwhere(dose_samples < s[mi[0]])
    print(low_res_idx[:,0])
    high_res_idx = np.argwhere(dose_samples > s[mi[0]])
    print(high_res_idx[:,0])



    dose_map_low = film_measurements_holes.EBT_model(netOD_holes[low_res_idx[:,0]] ,fitting_param_low[0],fitting_param_low[1],fitting_param_low[2])

    dose_map_high = film_measurements_holes.EBT_model(netOD_holes[high_res_idx[:,0]] ,fitting_param_high[0],fitting_param_high[1],fitting_param_high[2])


    mean_dose_grid = np.mean(np.concatenate((dose_map_low, dose_map_high)), axis = 0)
    plt.imshow(mean_dose_grid)
    plt.show()
    # np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\mean_film_dose_map\\mean_dose_grid.npy", mean_dose_grid)

    image_height_cm = mean_dose_map.shape[1]/(11.81*10)  #11.81/10 pixels per cm for 300 dpi image

    image_height = np.linspace(0,image_height_cm,mean_dose_map.shape[1])


    mean_dose_low = dose_profile(dose_map_low.shape[1],dose_map_low)
    mean_dose_high = dose_profile(dose_map_high.shape[1],dose_map_high)

    for i in range(len(mean_dose_low)):
        if i ==0:
            plt.plot(image_height,mean_dose_low[i],color = "b",label = " low dose grid")
        else:
            plt.plot(image_height,mean_dose_low[i], color = "b")

    for i in range(len(mean_dose_high)):
        if i == 0:
            plt.plot(image_height,mean_dose_low[i],color = "r", label = "high dose grid")
        else:
            plt.plot(image_height, mean_dose_high[i],color = "r")

    plt.legend()
    plt.show()"""

from film_calibration_tmp9 import film_calibration
import numpy as np
import matplotlib.pyplot as plt
from dose_profile import dose_profile
from dose_map_registration import phase_correlation#image_reg
import cv2

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Calibration"
background = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Background"
control = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Control"
test_image = "EBT3_Calib_131021_Xray220kV_01Gy1_001.tif"
background_image = "EBT3_Calib_131021_Xray220kV_black_001.tif"
control_image = "EBT3_Calib_131021_Xray220kV_00Gy1_001.tif"
reg_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\registered_images"
#measurement_crop = [30,500,200,220]#[40,700,195,250] #for best profiles
measurement_crop = np.array([50,2700,70,1900])//4
film_calib = film_calibration(folder, background, control, test_image, background_image,
                              control_image, reg_path, measurement_crop, calibration_mode = True, image_registration = False, registration_save = False)

film_calib.image_acquisition()


films_per_dose = 8  #[0,1,2,3,4,5]
dose_axis = np.array([0,0.1,0.2,0.5,10,1,2,5])
ROI_size = 4 #mm
num_fitting_params = 3
#Finding netOD values
netOD, sigma_img, sigma_bckg, sigma_ctrl, dOD = film_calib.calibrate(ROI_size, films_per_dose)




#np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\netOD_calib\\netOD_131021.npy",netOD)

channels = ["BLUE", "GREEN", "RED", "GREY"]
colors = ["b","g","r","grey"]
for i in range(netOD.shape[0]):
    for j in range(netOD.shape[1]):
        plt.plot(np.repeat(dose_axis[j], netOD.shape[2]), netOD[i,j], "*", color = colors[i])
plt.legend(channels)
plt.close()

"""
Here we make the stacked dose array, in our case we have 8 films per dose.
"""

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


bandwidth_red_channel = np.array([0.0005, 0.003, 0.003,0.0005, 0.003, 0.003, 0.005, 0.003])
low_response_OD, high_response_OD, low_res_dose, high_res_dose, bandwidth_red_channel, no_split_OD = \
film_calib.netOD_split(dose_axis, bandwidth_red_channel, bandwidth_stepsize = 0.0001, channel = "RED")


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
                              control_image, reg_path, measurement_crop, calibration_mode = False, image_registration = True,registration_save = True, grid = True)
# film_measurements_holes.image_acquisition()
film_measurements_grid.image_acquisition()

# netOD_holes,_,_,_,_ = film_measurements_holes.calibrate(ROI_size, films_per_dose)
netOD_grid,_,_,_,_ = film_measurements_grid.calibrate(ROI_size, films_per_dose)

"""
from 310821
fitting param low response       fitting param high response
[ 7.23795233 50.37465016  2.62488359] [ 6.20537405 50.8145797   2.95422661]
"""

bandwidth_grid = 0.005
bandwidth_stepsize = 0.0001
# low_img_grid, high_img_grid = film_measurements_holes.netOD_split(dose_axis, bandwidth_grid, bandwidth_stepsize)
low_img_grid, high_img_grid = film_measurements_grid.netOD_split(dose_axis, bandwidth_grid, bandwidth_stepsize)

print(low_img_grid,high_img_grid)

low_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[low_img_grid],fitting_param_low, model_type = 1)
high_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[high_img_grid],fitting_param_high, model_type = 1)


mean_dose_grid = np.mean(np.concatenate((low_res_dose_grid, high_res_dose_grid)), axis = 0)


image_height_cm = low_res_dose_grid.shape[1]/(11.81*10)  #11.81/10 pixels per cm for 300 dpi image

image_height = np.linspace(0,image_height_cm,low_res_dose_grid.shape[1])


print(low_res_dose_grid.shape)


low_mean_grid, low_std_grid = dose_profile(low_res_dose_grid.shape[1],low_res_dose_grid)

high_mean_grid, high_std_grid =  dose_profile(high_res_dose_grid.shape[1],high_res_dose_grid)


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
#fitting_param_low = [11.02812988, 39.98201961,  2.43936076]
#fitting_param_high = [ 8.94158385, 40.75259796,  2.62391097]
# fitting_param_low = [ 7.23795233, 50.37465016,  2.62488359]
fitting_param_high = [ 6.20537405, 50.8145797,   2.95422661] #newest fitting data

low_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[low_img_grid],fitting_param_low, model_type = 1)
high_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[high_img_grid],fitting_param_high, model_type = 1)

low_mean_grid, low_std_grid = dose_profile(low_res_dose_grid.shape[1],low_res_dose_grid)

high_mean_grid, high_std_grid =  dose_profile(high_res_dose_grid.shape[1],high_res_dose_grid)


mean_dose_map = np.mean(np.concatenate((low_res_dose_grid, high_res_dose_grid)), axis = 0)

print(mean_dose_map.shape, (mean_dose_map.shape[0]*4,mean_dose_map.shape[1]*4))



np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\mean_film_dose_map\\mean_dose_grid_1D.npy", mean_dose_grid)


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
plt.show()"""
# np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\mean_film_dose_map\\mean_dose_grid.npy", mean_dose_grid)

image_height_cm = dose_map.shape[1]/(11.81*10)  #11.81/10 pixels per cm for 300 dpi image

image_height = np.linspace(0,image_height_cm,dose_map.shape[1])


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
plt.show()

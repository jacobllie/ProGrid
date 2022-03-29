from film_calibration_tmp8 import film_calibration
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
measurement_crop = [10,722,10,497]
film_calib = film_calibration(folder, background, control, test_image, background_image,
                              control_image, reg_path, measurement_crop, calibration_mode = True, image_registration = False)

film_calib.image_acquisition()


films_per_dose = 8
dose_axis = np.array([0,0.1,0.2,0.5,10,1,2,5])
ROI_size = 2 #mm
num_fitting_params = 3
#Finding netOD values
netOD = film_calib.calibrate(ROI_size, films_per_dose)



"""
Here we make the stacked dose array, in our case we have 8 films per dose.
"""

netOD_high = list(np.delete(np.ravel(netOD[2]), [6*8 + 6, 6*8 + 7]))

dose_axis_octo = np.array([0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
                0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                10,10,10,10,10,10,10,10,1,1,1,1,1,1,1,1,2,2,2,2,2,2,5,5,5,5,5,5,5,5])

plt.plot(dose_axis_octo, netOD_high,"*")
plt.close()



fitting_param_high = film_calib.EBT_fit(dose_axis_octo, num_fitting_params, netOD_high)



OD_axis = np.linspace(np.min(netOD_high),np.max(netOD_high),100)

plt.plot(film_calib.EBT_model(OD_axis, fitting_param_high[0], fitting_param_high[1], fitting_param_high[2]), OD_axis, label = r" high response fit: D = %.3f $\cdot$ netOD $\cdot$ %.4f $\cdot$ netOD$^{%.4f}$"%(fitting_param_high[0],fitting_param_high[1], fitting_param_high[2]))

"""
Same procedure only now we use all the datapoints in our fit
"""

netOD_ravelled = list(np.ravel(netOD[2]))
dose_axis_octo = np.array([0,0,0,0,0,0,0,0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
                0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,
                10,10,10,10,10,10,10,10,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5])



fitting_params = film_calib.EBT_fit(dose_axis_octo, num_fitting_params, netOD_ravelled)


OD_axis = np.linspace(np.min(netOD_ravelled),np.max(netOD_ravelled),100)

plt.plot(film_calib.EBT_model(OD_axis, fitting_param_high[0], fitting_param_high[1], fitting_param_high[2]), OD_axis,"*", label = r" fit: D = %.3f $\cdot$ netOD $\cdot$ %.4f $\cdot$ netOD$^{%.4f}$"%(fitting_param_high[0],fitting_param_high[1], fitting_param_high[2]))

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
plt.show()


test_image_holes = "EBT3_Holes_131021_Xray220kV_5Gy1_001.tif"
holes_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Measurements"
#measurement_crop= [50, 650, 210, 260]

image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Measurements\\EBT3_Holes_131021_Xray220kV_5Gy1_001.tif", 1)

# plt.imshow(image[10:722, 10:497])
# plt.show()

film_measurements_holes = film_calibration(holes_folder, background, control, test_image_holes, background_image,
                              control_image, reg_path, measurement_crop, calibration_mode = False, image_registration = False, grid = True)

film_measurements_holes.image_acquisition()



netOD_holes = film_measurements_holes.calibrate(ROI_size, films_per_dose)

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
fitting_param_low = [11.02812988, 39.98201961,  2.43936076]
fitting_param_high = [ 8.94158385, 40.75259796,  2.62391097]
dose_map = film_measurements_holes.EBT_model(netOD_holes ,fitting_params[0],fitting_params[1],fitting_params[2])


"""
We take a small area in the dose map of each film and find mean dose. Then we split
the doses into high and low response, and apply the high and low fit from last
experiment.
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

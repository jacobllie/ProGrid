from film_calibration_tmp8 import film_calibration
import numpy as np
import matplotlib.pyplot as plt
from dose_profile import dose_profile
from dose_map_registration import phase_correlation#image_reg
from utils import mean_dose

image_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"
test_image = "EBT3_Calib_310821_Xray220kV_01Gy1_001.tif"
background_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Background"
control_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Control"
background_image = "EBT3_Calib_310821_Xray220kV_black1.tif"
control_image = "EBT3_Calib_310821_Xray220kV_00Gy1_001.tif"
reg_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\registered_images"
measurement_crop = [10,722,10,497]
#crop for finding peak width
peak = False
valley = False
if peak == True:
    measurement_crop = [300,500,50,400]
elif valley == True:
    measurement_crop = [420,550,50,400]
film_calib = film_calibration(image_folder, background_folder, control_folder, test_image, background_image,\
                              control_image, reg_path, measurement_crop, calibration_mode = True, image_registration = False)

#Gathering the images
film_calib.image_acquisition()


films_per_dose = 8
dose_axis = np.array([0,0.1,0.2,0.5,10,1,2,5])
ROI_size = 2 #mm

#Finding netOD values
netOD = film_calib.calibrate(ROI_size, films_per_dose)


"""
Finn riktig bandwidth til alle doser, deretter finn indexen til low response OD og high response OD.
Hvilket lokalt min skal vi være mindre enn? Skal helst bare ha ett lokalt minimum, slik at
vi lett kan si low_res_idx = OD[OD < s[mi[0]]]
"""


"""
We choose the red channel data to fit a low and high response netOD. This is
because the red channel data has the clearest 50/50 split between high and low response.
"""


num_fitting_params = 3


"""
Choosing the bandwidth, which decides the shape of the gaussian kernel that
is drawn over the netOD datapoints of each dosimetry film.
"""
bandwidth_red_channel = np.array([0.0005, 0.003, 0.003,0.0005, 0.003, 0.003, 0.005, 0.003])


"""
Getting the fitting parameters a,b,n from scipy.optimize curve_fit that best fits
the netOD data to the model: a*netOD + b * netOD**n
"""

low_response_OD, high_response_OD, low_res_dose, high_res_dose = film_calib.netOD_split(dose_axis, bandwidth_red_channel)


fitting_param_low, fitting_param_high = film_calib.EBT_fit(dose_axis,num_fitting_params)

print(fitting_param_low, fitting_param_high)

#Plotting dose vs netOD
plt.style.use("seaborn")
plt.title("Calibration curve for EBT3 films")
plt.xlabel("dose [Gy]")
plt.ylabel("netOD")

#looping over all the 8 films, and plotting their netOD

plt.plot(film_calib.EBT_model(np.linspace(0,max(low_response_OD),1000), fitting_param_low[0],
        fitting_param_low[1], fitting_param_low[2]), np.linspace(0,max(low_response_OD),1000), "--",
        color = "red", label = r"Low response fit  $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_low[0],
        fitting_param_low[1],fitting_param_low[2]))
plt.plot(film_calib.EBT_model(np.linspace(min(high_response_OD),max(high_response_OD),1000), fitting_param_high[0],
         fitting_param_high[1], fitting_param_high[2]), np.linspace(0,max(high_response_OD),1000), "--",
         color = "pink", label = r"High response fit $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_high[0],
         fitting_param_high[1],fitting_param_high[2]))

plt.plot(low_res_dose,low_response_OD,"p",color = "red")
plt.plot(high_res_dose, high_response_OD, "d",color = "pink")


for i in range(len(low_response_OD)):
    if i == 0:
        plt.plot(low_res_dose, low_response_OD, "p", color = "red", label = "low response netOD")
        #plt.plot(dose_axis, netOD[2,:,i],"p",color = "red", label = "red channel low response")
        #plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue", label = "blue channel netOD")
        #plt.plot(dose_axis, netOD[1,:,i],".",color = "green", label = "green channel netOD")
        #plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey", label = "grey channel netOD")
    else:
        plt.plot(low_res_dose, low_response_OD, "p", color = "red")
        #plt.plot(dose_axis, netOD[2,:,i],"p",color = "red")
        #plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue")
        #plt.plot(dose_axis, netOD[1,:,i],".",color = "green")
        #plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey")

for i in range(len(high_response_OD)):
    if i == 0:
        plt.plot(high_res_dose, high_response_OD, "p", color = "blue", label = "high response netOD")
    else:
        plt.plot(high_res_dose, high_response_OD, "p", color = "blue")

plt.legend()
plt.close()


"""
Now we gather the measurement films and see which dose the have based on
their netOD. Both for open field and grid
"""




test_image_open = "EBT3_Open_310821_Xray220kV_5Gy1_001.tif"
test_image_grid = "EBT3_Stripes_310821_Xray220kV_5Gy1_003.tif"
open_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Open"
grid_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes"

film_measurements_open = film_calibration(open_folder, background_folder, control_folder,\
                                          test_image_open, background_image, control_image,\
                                          reg_path, measurement_crop, calibration_mode = False,\
                                          image_registration = False, open = True)
film_measurements_grid = film_calibration(grid_folder, background_folder, control_folder,\
                                          test_image_grid, background_image, control_image,\
                                          reg_path, measurement_crop, calibration_mode = False,\
                                          image_registration = False, grid = True)

film_measurements_open.image_acquisition()
film_measurements_grid.image_acquisition()



netOD_open = film_measurements_open.calibrate(ROI_size, films_per_dose)
netOD_grid = film_measurements_grid.calibrate(ROI_size, films_per_dose)


bandwidth_grid = 0.005
bandwidth_open = 0.01

"""
For the measurement films, we have converted alle pixel values to netOD.
We therefore split into high and low
response using the mean OD of each film. The indexes of the films are returned by netOD split.
"""
low_img_open, high_img_open = film_measurements_open.netOD_split(dose_axis, bandwidth_open)
low_img_grid, high_img_grid = film_measurements_grid.netOD_split(dose_axis, bandwidth_grid)


low_res_dose_open = film_measurements_open.EBT_model(netOD_open[low_img_open],fitting_param_low[0],fitting_param_low[1],fitting_param_low[2])

high_res_dose_open =  film_measurements_open.EBT_model(netOD_open[high_img_open],fitting_param_high[0],fitting_param_high[1],fitting_param_high[2])

mean_dose_open = np.mean(np.concatenate((low_res_dose_open, high_res_dose_open)), axis = 0)


#print(mean_dose_open)
# np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open.npy", mean_dose_open)
#mean_dosemap_open = np.

low_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[low_img_grid],fitting_param_low[0],fitting_param_low[1],fitting_param_low[2])
high_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[high_img_grid],fitting_param_high[0],fitting_param_high[1],fitting_param_high[2])

mean_dose_grid = np.mean(np.concatenate((low_res_dose_grid, high_res_dose_grid)), axis = 0)

# np.savetxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_grid.npy", mean_dose_grid)


#phase_correlation(netOD_grid)
#phase_correlation(low_res_dose_grid)
#phase_correlation(high_res_dose_grid)



"""
for i in range(len(high_res_dose_grid)):
    fig = plt.figure()
    st = fig.suptitle("image {}".format(i), fontsize="x-large")
    plt.imshow(high_res_dose_grid[i])
    plt.show(block = False)
    plt.pause(0.0001)
plt.show()"""



"""
Now we'll make dose profiles by calculating the mean dose at a central position
"""


image_height_cm = low_res_dose_grid.shape[1]/(11.81*10)  #11.81/10 pixels per cm for 300 dpi image

image_height = np.linspace(0,image_height_cm,low_res_dose_grid.shape[1])


low_mean_grid = dose_profile(low_res_dose_grid.shape[1],low_res_dose_grid)


high_mean_grid =  dose_profile(high_res_dose_grid.shape[1],high_res_dose_grid)

"""
Finding the mean peak dose for colony categorization
"""

lower_lim = 3.8
higher_lim = 5

peak_dose_low, peak_dose_std_low = mean_dose(lower_lim, higher_lim, low_mean_grid)
peak_dose_high, peak_dose_std_high = mean_dose(lower_lim,higher_lim,high_mean_grid)

print("mean dose in peak for high and low response")
print((peak_dose_low + peak_dose_high)/2)
print("mean standard deviation for high and low response doses peak")
print((peak_dose_std_low + peak_dose_std_high)/2)

lower_lim = 0.5
higher_lim = 1

valley_dose_low, valley_dose_std_low = mean_dose(lower_lim, higher_lim, low_mean_grid)
valley_dose_high, valley_dose_std_high = mean_dose(lower_lim, higher_lim, high_mean_grid)

print("mean dose in valley for high and low response")
print((valley_dose_low + valley_dose_high)/2)
print("mean standard deviation for high and low response doses valley")
print((valley_dose_std_low + valley_dose_std_high)/2)



# print(np.mean(np.concatenate(peak_idx1[])))

low_mean_open = dose_profile(low_res_dose_open.shape[1],low_res_dose_open)
high_mean_open =  dose_profile(high_res_dose_open.shape[1],high_res_dose_open)
plt.subplot(121)
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

for i in range(len(low_mean_open)):
    if i ==0:
        plt.plot(image_height,low_mean_open[i],color = "b",label = "low response dose open")
    else:
        plt.plot(image_height,low_mean_open[i],color = "b")

for i in range(len(high_mean_open)):
    if i == 0 :
        plt.plot(image_height,high_mean_open[i],color = "r",label = "high response dose open")
    else:
        plt.plot(image_height,high_mean_open[i],color = "r")



mean_grid_dose = np.mean(np.concatenate((low_mean_grid,high_mean_grid)),axis = 0)
mean_open_dose = np.mean(np.concatenate((low_mean_open,high_mean_open)),axis = 0)

plt.legend()
plt.subplot(122)


if peak == True:
    peak_idx = np.argwhere(np.logical_and(3.8 < mean_grid_dose, mean_grid_dose < 4.2))
    peak_width = image_height[np.max(peak_idx)] - image_height[np.min(peak_idx)]
    print("Peak width is {} mm".format(peak_width*10))
    plt.plot([image_height[np.min(peak_idx)], image_height[np.max(peak_idx)]], [mean_grid_dose[np.min(peak_idx)], mean_grid_dose[np.max(peak_idx)]], label = "peak width")

elif valley == True:
    valley_idx = np.argwhere(np.logical_and(0.85 < mean_grid_dose, mean_grid_dose < 1.05))
    valley_width = image_height[np.max(valley_idx)] - image_height[np.min(valley_idx)]
    print("Valley width is {} mm".format(valley_width*10))
    plt.plot([image_height[np.min(valley_idx)], image_height[np.max(valley_idx)]], [mean_grid_dose[np.min(valley_idx)], mean_grid_dose[np.max(valley_idx)]], label = "valley width")


plt.plot(image_height,mean_grid_dose,"o", label = "mean grid dose")
plt.plot(image_height,mean_open_dose, label  ="mean open dose")
plt.show()
"""
for i in range(len(high_mean_open)):
    if i ==0:
        plt.plot(image_height,low_mean_open[i],color = "b",label = "low response dose")
    else:
        plt.plot(image_height,low_mean_open[i],color = "b")

for i in range(len(high_mean_open)):
    if i == 0 :
        plt.plot(image_height,high_mean_open[i],color = "r",label = "high response dose")
    else:
        plt.plot(image_height,high_mean_open[i],color = "r")

plt.legend()
plt.show()
"""

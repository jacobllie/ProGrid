from film_calibration_tmp3 import film_calibration
import numpy as np
import matplotlib.pyplot as plt


folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"
background_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Background"
test_image = "EBT3_Calib_310821_Xray220kV_00Gy1_001.tif"
film_calib = film_calibration(folder, test_image)

#Gathering the images
film_calib.image_acquisition()


films_per_dose = 8
dose_axis = np.array([0.1,0.2,0.5,10,1,2,5])
ROI_size = 2 #mm

#Finding netOD values
netOD = film_calib.calibrate(ROI_size, films_per_dose)

#Fitting the netOD values
num_fitting_params = 3

"""
Getting the fitting parameters a,b,n from scipy.optimize curve_fit that best fits
the netOD data to the model: a + b * netOD**n
"""
fitting_param, std_err = film_calib.EBT_fit(dose_axis,num_fitting_params)


sorted_dose_axis = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10])
sorted_dose_axis_octo = []
for i in sorted_dose_axis:
    for j in range(netOD.shape[2]):
        sorted_dose_axis_octo.append(i)

#Plotting dose vs netOD
plt.style.use("seaborn")
"""looping over all the 8 films, and plotting their netOD"""
for i in range(netOD.shape[2]):
    #plt.plot(dose_axis, netOD[0,:,i],".",color = "blue")
    plt.plot(dose_axis, netOD[1,:,i],".",color = "green")
    #plt.plot(dose_axis, netOD[2,:,i],".",color = "red")
    #plt.plot(dose_axis, netOD[3,:,i],".",color = "grey")


#plt.plot(dose_axis_octo, film_calib.EBT_model(dose_axis_octo, fitting_param[0,0], fitting_param[0,1], fitting_param[0,2]),color = "blue")
plt.plot(sorted_dose_axis_octo, film_calib.EBT_model(sorted_dose_axis_octo, fitting_param[1,0], fitting_param[1,1], fitting_param[1,2]),"--",color = "green")
#plt.plot(dose_axis_octo, film_calib.EBT_model(dose_axis_octo, fitting_param[2,0], fitting_param[2,1], fitting_param[2,2]),color = "red")
#plt.plot(dose_axis_octo, film_calib.EBT_model(dose_axis_octo, fitting_param[3,0], fitting_param[3,1], fitting_param[3,2]),color = "grey")
plt.show()

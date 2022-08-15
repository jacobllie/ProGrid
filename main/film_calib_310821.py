from film_calibration_tmp9 import film_calibration
import numpy as np
import matplotlib.pyplot as plt
from dose_profile import dose_profile
from utils import mean_dose, dose_fit_error
from scipy.stats import t
import pandas as pd


image_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"
test_image = "EBT3_Calib_310821_Xray220kV_01Gy1_001.tif"
background_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Background"
control_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Control"
background_image = "EBT3_Calib_310821_Xray220kV_black1.tif"
control_image = "EBT3_Calib_310821_Xray220kV_00Gy1_001.tif"
reg_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\registered_images"
# measurement_crop = [20,650,20,457] #best for dose map
measurement_crop = [35,600,260,290] #best for profiles


"""
Image register for measurement films are cropped 20 pixels in x and y direction,
så when deciding a measurement crop, have one that  crops at least 20 pixels in x and y direction.
Also make sure, that enough of the film is within the frame of the image, to ensure all cells in cell flask have a dose.
also make sure that the dose within the chosen frame are okay.
"""
#measurement_crop = np.array([50,2700, 10,2000])//4 #cropping_limits_1D to have as much as the film within frame as possible
#measurement_crop = np.array([50,2700,70,1900])//4 # 12,675,17,475 equivalent in dose film shape (663, 458) smallest shape possible is (733,487)

#crop for finding peak width
peak = False
valley = False
if peak == True:
    measurement_crop = [300,500,50,400]
elif valley == True:
    measurement_crop = [420,550,50,400]
film_calib = film_calibration(image_folder, background_folder, control_folder,
                              test_image, background_image,control_image,
                              reg_path, measurement_crop, calibration_mode = True,
                              image_registration = False, registration_save = False)

#Gathering the images
film_calib.image_acquisition()


films_per_dose = 8
dose_axis = np.array([0,0.1,0.2,0.5,10,1,2,5])
ROI_size = 3 #mm

#Finding netOD values
netOD, sigma_img, sigma_bckg, sigma_ctrl, dOD = film_calib.calibrate(ROI_size, films_per_dose)


for i in range(netOD.shape[1]):
    if i == 0:
        plt.plot(dose_axis, netOD[2,:,i],"p",color = "red", label = "red channel")
        plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue", label = "blue channel ")
        plt.plot(dose_axis, netOD[1,:,i],".",color = "green", label = "green channel ")
        plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey", label = "grey channel")
    else:
        plt.plot(dose_axis, netOD[2,:,i],"p",color = "red")
        plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue")
        plt.plot(dose_axis, netOD[1,:,i],".",color = "green")
        plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey")
plt.legend()
plt.title("31.08.21")
plt.xlabel("Dose [Gy]")
plt.ylabel("netOD")
# plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\netOD_plot_all_channels.png", bbox_inches = "tight", pad_inches = 0.2, dpi = 1200)
plt.show()
#np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\netOD_310821.npy", netOD)

netOD_1310 = np.load("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\netOD_calib\\netOD_131021.npy")

plt.style.use("seaborn")
print(netOD.shape)
colors = ["b","g","r","grey","m","y","black","saddlebrown"]

print((np.repeat(dose_axis,8)).shape, netOD[0,:].flatten().shape)

plt.title("RED channel netOD")
plt.plot(np.repeat(dose_axis,8),netOD_1310[2,:].flatten(), "*", label = "1310")
#plt.errorbar(np.repeat(dose_axis,8),netOD[2,:].flatten(), yerr = dOD[2,:].flatten(), fmt =  "o", markersize = 5, c = color[2], label = "3110")
plt.plot(np.repeat(dose_axis,8),netOD[2,:].flatten(),"o", label  =" 3108", markersize = 4)
plt.xlabel("Dose [Gy]")
plt.ylabel("netOD")

plt.legend()
# plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\netOD_3108_1310_RED_split.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)
plt.show()

channels = ["BLUE","GREEN","RED","GREY"]
fig,ax = plt.subplots(2,2, sharex = True, sharey = True, figsize= (12,12))

ax = ax.flatten()
plt.suptitle("netOD with 95% CI")

"""
Find confidence band from error in netOD
"""
for i in range(len(netOD)):
    ax[i].set_title(channels[i])
    if i == 2 or i == 3:
        ax[i].set_xlabel("# film")
    if i == 0 or i == 2:
        ax[i].set_ylabel("netOD")
    for j in range(netOD.shape[1]):
        #ax[i].errorbar(x = np.arange(1,films_per_dose + 1,1), y = netOD[i,j], yerr = dOD[i,j], fmt =  "o", markersize = 5, c = color[j],label = "{} Gy".format(dose_axis[j]))
        ax[i].plot(np.arange(1,films_per_dose + 1,1), netOD[i,j], "o", markersize = 7,c = colors[j],label = "{} Gy".format(dose_axis[j]))


        # ax[i].fill_between(x = np.arange(1,films_per_dose + 1,1), y1 = netOD[i,j] - t.ppf(0.95,dOD.shape[1] - 1)*dOD[i,j], y2 = netOD[i,j] + t.ppf(0.95, dOD.shape[1])*dOD[i,j], alpha = 0.7, color = colors[j])

    ax[i].legend(frameon= True, borderpad = 1, markerscale = 1.1, loc = "upper right")
# fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\netOD_per_dose_per_channel.png",bbox_inches = "tight",pad_inches = 0.1, dpi = 1200)
plt.show()


"""
Plotting in the same plot, no subplots
"""
fig,ax = plt.subplots()
ax.set_title("netOD for all channels")
"""
Find confidence band from error in netOD
"""

for i in range(len(netOD)):
    for j in range(netOD.shape[1]):
        ax.errorbar(np.repeat(dose_axis[j],8),netOD[i,j], yerr = dOD[i,j], fmt =  "o", markersize = 5, c = colors[i])
plt.close()


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
bandwidth_blue_channel = np.array([0.0005, 0.003, 0.003,0.0005, 0.003, 0.003, 0.005, 0.003])
bandwidth_green_channel = np.array([0.0005, 0.003, 0.003,0.0005, 0.003, 0.003, 0.005, 0.003])
bandwidh_grey_channel = np.array([0.0005, 0.003, 0.003,0.0005, 0.003, 0.003, 0.005, 0.003])

"""
Getting the fitting parameters a,b,n from scipy.optimize curve_fit that best fits
the netOD data to the model: a*netOD + b * netOD**n
"""
# channel = "BLUE"
# channel = "GREEN"
channel = "RED"
# channel = "GREY"


#getting low and high response indexes
low_response_OD, high_response_OD, low_res_dose, high_res_dose, bandwidth_blue_channel = film_calib.netOD_split(dose_axis, bandwidth_blue_channel, bandwidth_stepsize = 0.0001, channel = channel)
# low_response_OD, high_response_OD, low_res_dose, high_res_dose, bandwidth_blue_channel = film_calib.netOD_split(dose_axis, bandwidth_blue_channel, bandwidth_stepsize = 0.0001, channel = "GREEN")
# low_response_OD, high_response_OD, low_res_dose, high_res_dose, bandwidth_blue_channel = film_calib.netOD_split(dose_axis, bandwidth_blue_channel, bandwidth_stepsize = 0.0001, channel = "RED")
# low_response_OD, high_response_OD, low_res_dose, high_res_dose, bandwidth_blue_channel = film_calib.netOD_split(dose_axis, bandwidth_blue_channel, bandwidth_stepsize = 0.0001, channel = "GREY")


print(low_response_OD.shape, high_response_OD.shape)


low_dose, low_dose_count = np.unique(low_res_dose, return_counts = True)
high_dose, high_dose_count = np.unique(high_res_dose, return_counts = True)


"""
If there are high and low response OD present for all dosepoints, low_OD and high_OD
should be of equal length
"""
low_OD = np.zeros(len(low_dose))
high_OD = np.zeros(len(high_dose))

print(low_OD.shape)
print(high_OD.shape)

idx = 0
for i in range(len(low_dose)):
    print(low_res_dose[idx:idx + low_dose_count[i]])
    low_OD[i] = np.mean(low_response_OD[idx:idx + low_dose_count[i]])
    idx += low_dose_count[i]



print("-------------")

idx = 0
for i in range(len(high_dose)):
    print(idx)
    high_OD[i] = np.mean(high_response_OD[idx:idx + high_dose_count[i]])
    idx += high_dose_count[i]

print("percentage difference between high and low response OD")

percentage_diff = np.abs(low_OD - high_OD)*100/(low_OD+high_OD)/2
print(percentage_diff)

plt.plot([0,0.1,0.2,0.5,1,2,5,10], percentage_diff,"*", label = r"$\frac{|low-high|}{(low + high)/2}$")
plt.legend()
plt.show()
# low_response_OD, high_response_OD, low_res_dose, high_res_dose = film_calib.netOD_split(dose_axis, bandwidth_red_channel)



print("low response OD")
print(low_response_OD)
print("----------------")
print("high response OD")
print(high_response_OD)
print("------------")
print("1310")
print(netOD_1310[2])

# np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\low_res_netOD_310821.npy", low_response_OD)
# np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\high_res_netOD_310821.npy", high_response_OD)
#
# np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\low_res_dose_310821.npy", low_res_dose)
# np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\netOD_3108\\high_res_dose_310821.npy", high_res_dose)


model_type = 1



# fitting_param_low, fitting_param_high, residual_low, residual_high = film_calib.EBT_fit(dose_axis,num_fitting_params, model_type = 1)
# fitting_param_low, fitting_param_high = film_calib.EBT_fit(dose_axis,num_fitting_params)#, model_type = 1)
fitting_param_low, fitting_param_high, param_var_low, param_var_high, fit_low, fit_high = film_calib.EBT_fit(dose_axis,num_fitting_params, model_type = 1)

print(param_var_low, param_var_high)


f = open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Fitting results LM\\MSE_netODfit.txt", "a")
f.write("\n\n{} x {} mm2 ROI\n". format(ROI_size, ROI_size))
f.write("\nMSE [low, high]\t\t\t\t channel\n")
f.write("[{:.5f}, {:.5f},{:.5f}]\t\t\t{}".format(np.sum(fit_low.fun**2)/len(fit_low.fun),
                                                      np.sum(fit_high.fun**2)/len(fit_high.fun),
                                                      np.sum(fit_low.fun**2)/len(fit_low.fun) + np.sum(fit_high.fun**2)/len(fit_high.fun) ,channel))
f.close()



print("fitting param low response       fitting param high response")
print(fitting_param_low, fitting_param_high)

print("paramer variance [a,b,n]")
print(param_var_low, param_var_high)



"""
print("RSS")
print(residual_low, residual_high)
print("Fitting params:\na,b,n")
print(fitting_param_low, fitting_param_high)"""

#Plotting dose vs netOD

confidence = False
if confidence:
    """
    Remember to add confidence band to the regression
    """
    low_response_OD_interp = np.linspace(0,max(low_response_OD),100)
    high_response_OD_interp = np.linspace(0,max(high_response_OD),100)

    D_interp_low = film_calib.EBT_model(low_response_OD_interp, fitting_param_low, model_type)
    D_interp_high = film_calib.EBT_model(high_response_OD_interp, fitting_param_high,model_type)

    dDdp_low = np.array([low_response_OD_interp, low_response_OD_interp**fitting_param_low[2], np.log(low_response_OD_interp)*fitting_param_low[1]*low_response_OD_interp**fitting_param_low[2]])
    dDdp_high = np.array([high_response_OD_interp, high_response_OD_interp**fitting_param_high[2], np.log(high_response_OD_interp)*fitting_param_high[1]*high_response_OD_interp**fitting_param_high[2]])

    #getting covariance matrix for parameters
    k = 3 #num parameters estiamted
    df_low = len(low_response_OD)- k - 1 #degrees of freedom
    df_high = len(high_response_OD) - k - 1
    #the hessian is the approximation of variance for datapoints
    hessian_approx_inv_low = np.linalg.inv(fit_low.jac.T.dot(fit_low.jac)) #follows H^-1 approx J^TJ
    hessian_approx_inv_high = np.linalg.inv(fit_high.jac.T.dot(fit_high.jac))

    std_err_res_low = np.sqrt(np.sum(fit_low.fun**2)/df_low)**2
    std_err_res_high = np.sqrt(np.sum(fit_low.fun**2)/df_high)**2


    param_cov_low = std_err_res_low * hessian_approx_inv_low
    param_cov_high = std_err_res_high * hessian_approx_inv_high


    #now we apply the covariance matrix on the derivative of G

    cov_D_low = dDdp_low.T.dot(param_cov_low).dot(dDdp_low)
    cov_D_high = dDdp_high.T.dot(param_cov_high).dot(dDdp_high)

    #is dof from interpolated function or just number of points (len(dose_axis) - k - 1 or len(dose_interp) - k - 1)
    """
    The different number of datapoints in low and high response affect the confidence
    """
    t_crit_low = t.ppf(0.95, len(low_response_OD_interp) - k - 1)
    t_crit_high = t.ppf(0.95,len(high_response_OD_interp)- k- 1)

#print(D_interp_low.shape, cov_D_low.shape, dDdp_low.shape, param_cov_low.shape)


plt.style.use("seaborn")
plt.title("Calibration curve for EBT3 films using {} color channel".format(channel))
plt.xlabel("dose [Gy]")
plt.ylabel("netOD")

"""
Maybe its best to drop confidence intervals here
"""

if channel == "BLUE":
    color1 = "blue"
    color2 = "navy"

elif channel == "GREEN":
    color1 = "green"
    color2 = "lime"
elif channel == "RED":
    color1 = "red"
    color2 = "pink"
elif channel == "GREY":
    color1 = "dimgrey"
    color2 = "darkgrey"
#looping over all the 8 films, and plotting their netOD
plt.plot(film_calib.EBT_model(np.linspace(0,max(low_response_OD),1000), fitting_param_low,model_type), np.linspace(0,max(low_response_OD),1000), "--",
        color = color1, label = r"Low response fit  ($%.3f \pm %.3f) \cdot netOD + (%.3f \pm %.3f) \cdot netOD^{(%.3f \pm %.3f)}$"%(fitting_param_low[0],np.sqrt(param_var_low[0]),
        fitting_param_low[1],np.sqrt(param_var_low[1]),fitting_param_low[2], np.sqrt(param_var_low[2])))
#plt.fill_betweenx(low_response_OD_interp, D_interp_low - t_crit_low *  np.sqrt(np.diag(cov_D_low)), D_interp_low + t_crit_low * np.sqrt(np.diag(cov_D_low)))
plt.plot(film_calib.EBT_model(np.linspace(0,max(high_response_OD),1000), fitting_param_high,model_type), np.linspace(0,max(high_response_OD),1000), "--",
         color = color2, label = r"High response fit $(%.3f \pm %.3f) \cdot netOD + (%.3f \pm %.3f)\cdot netOD^{(%.3f \pm %.3f)}$"%(fitting_param_high[0],np.sqrt(param_var_high[0]),
         fitting_param_high[1],np.sqrt(param_var_high[1]),fitting_param_high[2], np.sqrt(param_var_high[2])))
#plt.fill_betweenx(high_response_OD_interp, D_interp_high - t_crit_high *  np.sqrt(np.diag(cov_D_high)), D_interp_high + t_crit_high * np.sqrt(np.diag(cov_D_high)))
"""plt.plot(film_calib.EBT_model(np.linspace(0,max(low_response_OD),1000), fitting_param_low[0],
        fitting_param_low[1], fitting_param_low[2]), np.linspace(0,max(low_response_OD),1000), "--",
        color = "red", label = r"Low response fit  $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_low[0],
        fitting_param_low[1],fitting_param_low[2]))"""
"""plt.plot(film_calib.EBT_model(np.linspace(0,max(high_response_OD),1000), fitting_param_high[0],
         fitting_param_high[1], fitting_param_high[2]), np.linspace(0,max(high_response_OD),1000), "--",
         color = "pink", label = r"High response fit $%.3f \cdot netOD + %.3f \cdot netOD^{%.3f}$"%(fitting_param_high[0],
         fitting_param_high[1],fitting_param_high[2]))"""

plt.plot(low_res_dose,low_response_OD,"p",color = color1)
plt.plot(high_res_dose, high_response_OD, "d",color = color2)


for i in range(len(low_response_OD)):
    if i == 0:
        plt.plot(low_res_dose, low_response_OD, "p", color = color1, label = "low response netOD")
        # plt.plot(dose_axis, netOD[2,:,i],"p",color = "red", label = "red channel low response")
        # plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue", label = "blue channel netOD")
        # plt.plot(dose_axis, netOD[1,:,i],".",color = "green", label = "green channel netOD")
        # plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey", label = "grey channel netOD")
    else:
        plt.plot(low_res_dose, low_response_OD, "p", color = color1)
        # plt.plot(dose_axis, netOD[2,:,i],"p",color = "red")
        # plt.plot(dose_axis, netOD[0,:,i],"d",color = "blue")
        # plt.plot(dose_axis, netOD[1,:,i],".",color = "green")
        # plt.plot(dose_axis, netOD[3,:,i],"*",color = "grey")

for i in range(len(high_response_OD)):
    if i == 0:
        plt.plot(high_res_dose, high_response_OD, "p", color = color2, label = "high response netOD")
    else:
        plt.plot(high_res_dose, high_response_OD, "p", color = color2)

plt.legend(loc = "upper left")
# plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\{}_fit.png".format(channel), bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)
plt.show()



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
                                          image_registration = False,registration_save = False, open = True)
film_measurements_grid = film_calibration(grid_folder, background_folder, control_folder,\
                                          test_image_grid, background_image, control_image,\
                                          reg_path, measurement_crop, calibration_mode = False,\
                                          image_registration = False,registration_save = False, grid = True)

film_measurements_open.image_acquisition()
film_measurements_grid.image_acquisition()


"""
Don't know how to handle uncertainty in netOD measurement films. Because you don't use ROI,
but simply convert all pixels to netOD, so what is the uncertainty in PV_img ??
"""

netOD_open,_,_,_,dOD_open = film_measurements_open.calibrate(ROI_size, films_per_dose)
netOD_grid,_,_,_,dOD_grid = film_measurements_grid.calibrate(ROI_size, films_per_dose)

image_height = np.arange(0,netOD_open.shape[0],1) #pixels




bandwidth_grid = 0.005
bandwidth_open = 0.001

"""
For the measurement films, we have converted alle pixel values to netOD.
We therefore split into high and low
response using the mean OD of each film. The indexes of the films are returned by netOD split.
"""
low_img_open, high_img_open = film_measurements_open.netOD_split(dose_axis, bandwidth_open, bandwidth_stepsize = 0.0001)
low_img_grid, high_img_grid = film_measurements_grid.netOD_split(dose_axis, bandwidth_grid, bandwidth_stepsize = 0.0001)

print(low_img_open.shape, high_img_open.shape)
print(low_img_grid.shape, high_img_grid.shape)




#standard deviation of low and high response OD
dOD_low_open = dOD_open[low_img_open]
dOD_high_open = dOD_open[high_img_open]

dOD_low_grid = dOD_grid[low_img_grid]
dOD_high_grid = dOD_grid[high_img_grid]



low_res_dose_open = film_measurements_open.EBT_model(netOD_open[low_img_open],fitting_param_low, model_type)
# low_res_dose_open = film_measurements_open.EBT_model(netOD_open[low_img_open],fitting_param_low[0],fitting_param_low[1],fitting_param_low[2])
high_res_dose_open =  film_measurements_open.EBT_model(netOD_open[high_img_open],fitting_param_high, model_type)
# high_res_dose_open =  film_measurements_open.EBT_model(netOD_open[high_img_open],fitting_param_high[0],fitting_param_high[1],fitting_param_high[2])

dose_map_open = np.concatenate((low_res_dose_open, high_res_dose_open))




print("mean relative error in OPEN field dosemap")
print(np.mean(np.abs(dose_map_open.flatten() - 5)/5)*100)




"""
f = open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\ROI evaluation\\mean_rel_err_open_dose_variable_ROI_2.txt", "a")
f.write("\nMean Relative error from an analysis area of {}, with ROI {} ROI\n".format(measurement_crop, ROI_size))
f.write("REL ERR\n")
f.write("{:.5f}\n".format(np.mean(np.abs(np.mean(dose_map_open) - 5)/5)))
f.close()"""


#measurement_crop = [35,600,260,290] #best for now




#this is used in colony survival
# np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open_test.npy", mean_dose_open)

"""
All the values we need for error in dose
"""
#fitting_param_low, fitting_param_high, param_var_low, param_var_high, fit_low, fit_high
#low_response_OD, high_response_OD
#dose_fit_error(OD, dOD,dparam,param)

# std_err_dose = np.std(dose_map_open,axis = 0)/np.sqrt(len(dose_map_open))


"""
Using error propagation
"""

# std_fit = np.sqrt((dose_fit_error(netOD_open[low_img_open], dOD_low, param_var_low, fitting_param_low))**2/len(low_img_open) + \
#                   (dose_fit_error(netOD_open[high_img_open], dOD_high, param_var_high, fitting_param_high))**2/len(high_img_open))


"""
First we treat each image as a sample. We have n low response images and m high response images.
We find mean stderr for low and high.
"""
std_fit_open_low = np.sqrt(np.sum(dose_fit_error(netOD_open[low_img_open], dOD_low_open, param_var_low, fitting_param_low)**2,axis = 0))/len(low_img_open)
std_fit_open_high = np.sqrt(np.sum(dose_fit_error(netOD_open[high_img_open], dOD_high_open, param_var_high, fitting_param_high)**2,axis = 0))/len(high_img_open)

"""
Then we find mean stderr of these, because now we have reduced from sigma  = n,x,y to x,y.
So we can treat high and low as one sample with x*y data points each.
"""
mean_dose_map_error_open = np.sqrt(std_fit_open_low**2 + std_fit_open_high**2)

# std_err_fit = np.sum(std_fit,axis = 0)/np.sqrt(len(dose_map_open))
# mean_dose_map_error = std_err_fit

stderr_open_profile = np.sqrt(np.sum(mean_dose_map_error_open**2, axis = 1))/mean_dose_map_error_open.shape[1]

# conf_open = t.ppf(0.95, len(low_img_open) + len(high_img_open) - 2) * stderr_open_profile

"""
print("mean dose open field 5 Gy")
print(np.mean(dose_map_open))
print("mean standard error")
print(np.mean(mean_dose_map_error))"""


low_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[low_img_grid],fitting_param_low, model_type)
high_res_dose_grid =  film_measurements_grid.EBT_model(netOD_grid[high_img_grid],fitting_param_high, model_type)

# mean_dose_map_grid = np.mean(np.concatenate((low_res_dose_grid, high_res_dose_grid)), axis = 0)
print(low_res_dose_grid.shape)


#dose map visualization
high_res_dose_grid[0][high_res_dose_grid[0] > 5] = 4


"""plt.imshow(np.pad(high_res_dose_grid[0],(15,15)), cmap = "viridis")
plt.colorbar()
# plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\dose_map_striped_GRID.png",bbox_inches = "tight", pad_inches = 0.1, dpi =1200)
plt.show()"""



"""
Finding confidence interval
"""



#finding mean standard error of all films
# std_fit_grid_low = np.sqrt(np.sum(dose_fit_error(netOD_grid[low_img_grid], dOD_low_grid, param_var_low, fitting_param_low)**2,axis = 0))/len(low_img_grid)
# std_fit_grid_high = np.sqrt(np.sum(dose_fit_error(netOD_grid[high_img_grid], dOD_high_grid, param_var_high, fitting_param_high)**2,axis = 0))/len(high_img_grid)

#will use this for peak and valley dose estimation
fit_std_grid_profile_low = np.sqrt(np.sum(dose_fit_error(netOD_grid[low_img_grid], dOD_low_grid, param_var_low, fitting_param_low)**2, axis = 2))/low_res_dose_grid.shape[2]
fit_std_grid_profile_high = np.sqrt(np.sum(dose_fit_error(netOD_grid[high_img_grid], dOD_high_grid, param_var_high, fitting_param_high)**2, axis = 2))/high_res_dose_grid.shape[2]


#using this for confidence interval
std_fit_grid_low = np.sqrt(np.sum(dose_fit_error(netOD_grid[low_img_grid], dOD_low_grid, param_var_low, fitting_param_low)**2,axis = 0))/len(low_img_grid)
std_fit_grid_high = np.sqrt(np.sum(dose_fit_error(netOD_grid[high_img_grid], dOD_high_grid, param_var_high, fitting_param_high)**2,axis = 0))/len(high_img_grid)



#finding pooled standard error of combination

# mean_dose_map_error_grid = np.sqrt((std_fit_grid_low**2*(len(low_img_grid) - 2) + std_fit_grid_high**2*(len(high_img_grid) - 2))/(len(low_img_grid) + len(high_img_grid) - 2))
#mean_dose_map_error_grid = np.sqrt(std_fit_grid_low**2 + std_fit_grid_high**2)/2
mean_dose_map_error_grid = np.sqrt(std_fit_grid_low**2 + std_fit_grid_high**2)

stderr_grid_profile = np.sqrt(np.sum(mean_dose_map_error_grid**2, axis = 1))/mean_dose_map_error_grid.shape[1] #finding mean std error of each row

# conf_grid = t.ppf(0.95, len(low_img_grid) + len(high_img_grid) - 2) * stderr_grid_profile



"""
Using std of low and high films to find confidence
"""
"""
std_open = np.sum(np.var(low_res_dose_open, axis = 0) + np.var(high_res_dose_open, axis = 0))/(len(high_res_dose_open) + len(low_res_dose_open))  #np.sqrt(std1**2 + std2**2)/len(std1) + len(std2)
#pooled standard error
std_grid = np.sqrt((np.var(low_res_dose_grid, axis = 0) * (len(low_res_dose_grid) - 1) + np.var(high_res_dose_grid, axis = 0) * (len(high_res_dose_grid) - 1))/(len(low_res_dose_grid) + len(high_res_dose_grid) - 2))
#std error profile
stderr_grid = np.sqrt(np.sum(std_grid**2, axis = 1))/std_grid.shape[1]
conf = t.ppf(0.95, len(low_img_grid) + len(high_img_grid) - 2) * np.mean(np.std(np.concatenate((low_res_dose_grid, high_res_dose_grid)), axis = 0), axis = 1)/np.sqrt(len(low_res_dose_grid) + len(high_res_dose_grid))
"""
#this is used in surival
#np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_grid_test.npy", mean_dose_grid)



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

low_mean_open = dose_profile(low_res_dose_open.shape[1],low_res_dose_open)
high_mean_open=  dose_profile(high_res_dose_open.shape[1],high_res_dose_open)

dose_open = np.concatenate((low_mean_open,high_mean_open))
#standard deviation from averaging the dose profiles
open_mean_std_low = np.std(low_mean_open, axis = 0)/np.sqrt(len(low_mean_open))
open_mean_std_high = np.std(high_mean_open, axis = 0)/np.sqrt(len(high_mean_open))

#We sum because of independent errors
open_mean_std = np.sqrt(open_mean_std_low**2 + open_mean_std_high**2)/2

#combining error from fit and error from averaging the profiles
# conf_open = t.ppf(0.95, len(low_img_open) + len(high_img_open) - 2) * np.sqrt(stderr_open_profile**2 + open_mean_std_low**2 + open_mean_std_high**2)
conf_open = t.ppf(0.95, len(low_img_open) + len(high_img_open) - 2) * np.sqrt(stderr_open_profile**2 + open_mean_std**2)


"""
Finding the observed peak dose and valley dose to compare with monte carlo
"""

"""
Finding estimated peak and valley doses and OPEN field dose with standard error
"""
#combining high and low dose profiles
dose_grid = np.concatenate((low_mean_grid, high_mean_grid))

grid_mean_std_low = np.std(low_mean_grid, axis = 0)/np.sqrt(len(low_mean_grid))
grid_mean_std_high = np.std(high_mean_grid, axis = 0)/np.sqrt(len(high_mean_grid))

#this is combined with the standard error from the fit
grid_mean_std =  np.sqrt(grid_mean_std_low**2 + grid_mean_std_high**2)


#this conf grid includes standard deviation from finding
conf_grid = t.ppf(0.95, len(low_img_grid) + len(high_img_grid) - 2) * np.sqrt(stderr_grid_profile**2 + grid_mean_std**2)#grid_mean_std_low**2 + grid_mean_std_high**2)#grid_mean_std**2)

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
    #finding indexes of doses within peak and valley dose category
    peak_idx = dose_grid[i] > d95
    valley_idx = dose_grid[i] < d5

    #extracting the correct datapoints
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
#stacking all the error

peak_fit_std = np.array([list for i in range(len(dose_grid)) for list in doses_fit_std_peak[i]])
valley_fit_std = np.array([list for i in range(len(dose_grid)) for list in doses_fit_std_valley[i]])


print(peak_fit_std.shape, valley_fit_std.shape)

print(np.sqrt(np.sum(peak_fit_std**2))/len(peak))

print("mean OPEN dose   mean OPEN stderr     mean peak dose   mean peak stderr    mean valley dose    mean valley stderr mean peak fit stderr  mean valley fit stderr")



open_estimate_conf = t.ppf(0.95,len(dose_open.flatten()))*np.sqrt(np.sum(open_mean_std**2)/len(open_mean_std)) #one value confidence
grid_peak_conf = t.ppf(0.95, len(peak))*np.sqrt((np.std(peak)/np.sqrt(len(peak)))**2 + (np.sqrt(np.sum(peak_fit_std**2)/len(peak)))**2)
grid_valley_conf = t.ppf(0.95, len(valley))*np.sqrt((np.std(valley)/np.sqrt(len(valley)))**2 + (np.sqrt(np.sum(valley_fit_std**2)/len(valley)))**2)


print(np.mean(dose_open),[np.mean(dose_open) + open_estimate_conf, np.mean(dose_open) - open_estimate_conf],
      np.mean(peak), [np.mean(peak) - grid_peak_conf, np.mean(peak) + grid_peak_conf],
      np.mean(valley),[np.mean(valley) - grid_valley_conf, np.mean(valley) + grid_valley_conf]
      )





"""
Finding the mean peak dose for colony categorization
"""

"""lower_lim = 3.8
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
print((valley_dose_std_low + valley_dose_std_high)/2)"""



# print(np.mean(np.concatenate(peak_idx1[])))

true_dose_open = 5 #5 Gy was given nominally

"""Gathering Monte Carlo data for dose eval"""
grid_MC_data = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Monte Carlo\\MCresults_GRID.dat", unpack = True, skiprows = 1, delimiter = ",")
open_MC_data = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Monte Carlo\\MCresults_open.dat", unpack = True, skiprows = 1, delimiter = ",")

MC_score_grid = grid_MC_data[0]
position_grid = grid_MC_data[1]
mean_grid_dose = np.mean(np.concatenate((low_mean_grid,high_mean_grid)),axis = 0)

MC_score_open = open_MC_data[0]
mean_open_dose = np.mean(np.concatenate((low_mean_open,high_mean_open)),axis = 0)

#if Monte Carlo comparison
for i in range(len(image_height)):
    image_height[i] -= 1  #0.5 mm

new_idx = image_height >= 0
new_image_height = image_height[new_idx]
new_grid_dose = mean_grid_dose[new_idx]
new_open_dose = mean_open_dose[new_idx]

new_MC_idx = position_grid <= np.max(new_image_height)
new_MC_score_grid = MC_score_grid[new_MC_idx]
new_position = position_grid[new_MC_idx]

new_MC_score_open = MC_score_open[new_MC_idx]


fig, ax = plt.subplots(figsize = (14,7))
ax.set_title("Normalized dose", fontsize = 15)
ax.plot(new_image_height, new_open_dose/np.mean(dose_open), color = "green",label ="Obs OPEN")
ax.fill_between(image_height[new_idx], (mean_open_dose[new_idx] - conf_open[new_idx])/np.mean(dose_open), (mean_open_dose[new_idx] + conf_open[new_idx])/np.mean(dose_open), alpha = 0.8, color = "lime", label  = "95% CI OPEN")
ax.plot(new_image_height, new_grid_dose/np.mean(dose_open), color = "navy", label = "Obs GRID")
ax.fill_between(image_height[new_idx], (mean_grid_dose[new_idx] - conf_grid[new_idx])/np.mean(dose_open), (mean_grid_dose[new_idx] + conf_grid[new_idx])/np.mean(dose_open), alpha = 0.8, color = "cyan", label = "95% C.I. GRID")
ax.plot(new_position, new_MC_score_grid, label = "M.C.Score GRID", color = "magenta")
ax.plot(new_position, new_MC_score_open, label = "M.C.Score OPEN", color = "purple")
ax.set_xlabel("Position in cell flask [cm]", fontsize = 15)
ax.set_ylabel(r"$\frac{D_{GRID}}{\bar{D}_{OPEN}}$", fontsize = 17, rotation = 0)
ax.legend(loc = "lower right")
fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\MC_eval_dose_profile.png", pad_inches = 0.1, dpi = 300)
plt.show()




plt.plot(image_height, mean_grid_dose, label = "GRID 5 Gy")
plt.plot(image_height, mean_open_dose, label ="OPEN 5 Gy")
plt.fill_between(image_height, mean_grid_dose - conf_grid, mean_grid_dose + conf_grid, alpha = 0.8, color = "cyan", label = "95% CI GRID")
plt.fill_between(image_height, mean_open_dose - conf_open, mean_open_dose + conf_open, alpha = 0.8, color = "lime", label  = "95% CI OPEN")
plt.title("Dose profile 5 Gy nominal dose")
plt.xlabel("Position in Cell flask [cm]")
plt.ylabel("Dose [Gy]")
plt.legend()
plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\dose_profile.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 300)
plt.show()



plt.title("Dose profile 5 Gy nominal dose striped GRID")
# plt.xlabel("Position in cell flask [cm]")
# plt.ylabel("Dose [Gy]")
# rel_err = np.sum(np.abs((low_mean_open - true_dose_open))/true_dose_open)/(low_mean_open.shape[0]*low_mean_open.shape[1]) + \
#           np.sum(np.abs((high_mean_open - true_dose_open))/true_dose_open)/(high_mean_open.shape[0]*high_mean_open.shape[1])

"""f = open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\ROI evaluation\\mean_rel_err_open_dose_variable_ROI.txt", "a")
f.write("\nMean Relative error from dose profile rows using open field (8,615) {} ROI\n".format(ROI_size))
f.write("REL ERR\n")
f.write("{}\n".format(rel_err))
f.close()
print("Average relative error  ")
print(rel_err)"""

#plt.subplot(121)
plt.xlabel("Position in cell flask [cm]")
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


# mean_grid_con = t.ppf(0.95,len(low_mean_grid) + len(high_mean_grid) - 1)*\
# np.std(np.concatenate((low_mean_grid,high_mean_grid)),axis = 0)/np.sqrt(len(low_mean_grid) + len(high_mean_grid))
# mean_open_con = t.ppf(0.95,len(low_mean_open) + len(high_mean_open)-1)*\
# np.std(np.concatenate((low_mean_open,high_mean_open)),axis = 0)/np.sqrt(len(low_mean_open) + len(high_mean_open))


plt.legend()
plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\dose_profile_all_films.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 300)
print(image_height)
plt.show()


"""plt.subplot(122)
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
plt.show()"""
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

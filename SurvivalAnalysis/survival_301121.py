from survival_analysis import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f, ttest_ind
from kernel_density_estimation import kde
import seaborn as sb
from LQ_model import logLQ, fit, logLQ

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021"
time = ["18112019", "20112019"]
mode = ["Control", "Open"]
dose = ["02", "05"]
template_file =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Control\\A549-1811-K1-TemplateMask.csv"
dose_map_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open.npy"
position = ["A","B","C","D"]

"""
18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
We therefore combine these data to find alpha beta for open field irradiation.
"""

#plt.imshow(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Control\\A549-1811-K1-SegMask.csv"))
#plt.close()

"""
Finding the number of counted colonies for control (0Gy) and open field
experiments (2Gy and 5Gy)
"""
survival_control = survival_analysis(folder, time, mode[0], position, 3, dose_map_path, template_file, dose = None)
ColonyData_control, data_control = survival_control.data_acquisition()


"""max_count = np.max(data_control)

data_open/= max_count
data_control/= max_count  #normalizing counting to max number in ctrl

dose_0 = [0 for i in range(len(np.ravel(data_control)))]

#Identifying outliers in the data, and removing them

data_open_2GY = np.ravel(data_open[:,0,:])

IQR_open = np.quantile(data_open_2GY,0.75) - np.quantile(data_open_2GY, 0.25)

outlier_idx = np.argwhere(data_open_2GY < np.quantile(data_open_2GY, 0.25) - 1.5*IQR_open)

no_outlier = np.delete(data_open_2GY,outlier_idx[:,0])

dose_2 = [2 for i in range(len(no_outlier))]
dose_5 = [5 for i in range(len(np.ravel(data_open[:,1,:])))]


#Combining all doses, to match the count-datapoints

doses = np.append(dose_0, np.append(dose_2,dose_5))


#Combining all counting data for fit.

combined_count = np.log(np.append(np.ravel(data_control), np.append(no_outlier, np.ravel(data_open[:,1,:]))))


fitting_param = fit(logLQ, doses, combined_count)

plt.style.use("seaborn")
plt.xlabel("dose [Gy]")
plt.ylabel(r"$SF_{log}$")

plt.plot(dose_2, np.log(no_outlier),"*", color = "magenta")
plt.plot(dose_5, np.log(np.ravel(data_open[:,1,:])),"*", color = "magenta")
# plt.show()

plt.plot(dose_0, np.log(np.ravel(data_control)), "*", color = "magenta")
#plt.plot(np.linspace(np.min(doses),10,100),logLQ(np.linspace(np.min(doses),10,100),
#        fitting_param[0], fitting_param[1]), color = "salmon",
#        label = r"fit: -($\alpha \cdot d + \beta \cdot d^2$)" + "\n" + r"$\alpha = {:.4}, \beta = {:.4}$".format(fitting_param[0], fitting_param[1]))

plt.plot(np.linspace(np.min(doses),5,100),logLQ(np.linspace(np.min(doses),5,100),
        fitting_param[0], fitting_param[1]), color = "salmon",
        label = r"fit: -($\alpha \cdot d + \beta \cdot d^2$)" + "\n" + r"$\alpha = {:.4}, \beta = {:.4}$".format(fitting_param[0], fitting_param[1]))
plt.legend()
plt.close()"""

#survival_open = survival_analysis(folder, time, mode[1], position, 3, dose_map_path, template_file, dose = dose)
#ColonyData_open, data_open  = survival_open.data_acquisition()


#survival_open.Colonymap()
#survival_open.registration()
#survival_open.Quadrat()

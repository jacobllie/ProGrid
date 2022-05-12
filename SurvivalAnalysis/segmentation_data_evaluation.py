from survival_analysis3 import survival_analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021"
time = ["18112019", "20112019", "03012020","17122020"]
mode = ["Control"]
ctrl_dose = ["00"]
"""
Template has equal shape for all experiments
"""
template_file_control =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\Control\\A549-1811-K1-TemplateMask.csv"
template_file_open = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\20112019\\Open\\A549-2011-02-open-A-TemplateMask.csv"
template_file_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\GRID Stripes\\A549-1811-02-gridS-A-TemplateMask.csv"
dose_path_open = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open.npy"
dose_path_grid = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_grid.npy"
position = ["A","B","C","D"]

cropping_limits = [210,2100,405,1900]
kernel_size_mm = 3
# kernel_size_mm = [0.5,1,2,3,4]
kernel_size_p = int(kernel_size_mm*47)
survival_control = survival_analysis(folder, time, mode[0], position, kernel_size_p,
                                     dose_map_path = None, template_file = template_file_control,
                                     dose = ctrl_dose, cropping_limits = cropping_limits)

ColonyData_control, data_control = survival_control.data_acquisition()

print(data_control[0])

f1 = f_oneway(data_control[0,0],data_control[1,0])
f2 = f_oneway(data_control[2,0], data_control[3,0])

print(f1,f2)

plt.style.use("seaborn")
x = np.arange(1,5,1)
plt.plot(x,data_control[0,0], "*", label = time[0])
plt.plot(x,data_control[1,0], "*", label = time[1])
plt.plot(x,data_control[2,0], "*", label = time[2])
plt.plot(x,data_control[3,0], "*", label = time[3])
plt.xlabel("ctrl flask")
plt.ylabel("counted colonies")
plt.legend()
plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\data acquisition\\segmentation_data.png", pad_inches = 0.1, bbox_inches = "tight", dpi = 1200)

plt.show()

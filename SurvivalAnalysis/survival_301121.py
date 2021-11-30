from survival_analysis import survival_analysis
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021"
time = ["20112019"]
mode = ["Control"]
template_file_control =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Control\\A549-1811-K1-TemplateMask.csv"
dose_map_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open.npy"
position = ["A","B","C","D"]

"""
18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
We therefore combine these data to find alpha beta.
"""


"""
Finding the number of counted colonies for control (0Gy) and open field
experiments (2Gy and 5Gy)
"""
survival_control = survival_analysis(folder, time, mode[0], position, 3, dose_map_path, template_file_control, dose = None)
ColonyData_control, data_control = survival_control.data_acquisition()



survival_control.Colonymap()
survival_control.registration()
survival_control.Quadrat()

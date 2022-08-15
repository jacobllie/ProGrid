import numpy as np

def dose_profile(pixel_height, dose_array):
    mean_dose = np.zeros((len(dose_array),pixel_height))
    for i in range(len(dose_array)):
        mean_dose[i] = [np.mean(dose) for dose in dose_array[i]]
    return mean_dose
def netOD_profile(pixel_height,OD_array):
    mean_OD = np.zeros(len(OD_array))
    for i in range(len(OD_array)):
        mean_OD[i] = np.mean(OD_array[i])
    return mean_OD

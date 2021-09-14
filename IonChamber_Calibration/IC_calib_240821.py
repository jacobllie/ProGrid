from dose_calibration import low_dose_estimation, high_dose_estimation, sec_to_min_and_sec
import numpy as np
import matplotlib.pyplot as plt


T = np.array([5,10,15,20]) #seconds
N_k = 43.77e-3 #Gy/nC
k_u = 1
mu_over_rho = 1.075
P_u = 1.02
k_TP = 1.005090428
C = N_k * k_u * mu_over_rho * P_u * k_TP


"""
D has shape (4,3,4), 4 positions, 4 measurements for 5, 10, 15 and 20 seconds repeated 3 times.
"""
D = np.array([[[0.41, 1.38, 2.59, 3.52] , [0.38,1.45,2.44,3.51] , [0.43,1.46,2.37,3.5]] \
            ,[[0.4,1.34,2.47,3.51] , [0.42,1.43,2.43,3.4] , [0.38,1.36,2.5,3.45]] \
            ,[[0.48,1.51,2.48,3.53] , [0.44,1.55,2.5,3.57] , [0.46,1.47,2.43,3.49]] \
            ,[[0.42,1.49,2.58,3.64],[0.36,1.54,2.46,3.67],[0.43,1.58,2.56,3.64]]]) * C

pos_labels = ["A","B","C","D"]

#low dose estimation

low_dose = np.array([0.1,0.2,0.5])
time = low_dose_estimation(T,D,low_dose,pos_labels)
plt.close()

mean_time = np.mean(time,axis = 0)
for i in range(len(low_dose)):
    print("For {:.1f} Gy {:d} s".format(low_dose[i],int(mean_time[i])))


#high dose estimation

high_dose = np.array([1,2,5,10])*1.0256
doserate = np.array([[11.82,11.68,11.71],[11.7,11.65,11.61],[11.88,11.86,11.92],\
[12.11,12.02,12.08]]) * C/60 #dose/s

time = high_dose_estimation(high_dose,doserate)

for i, dose in enumerate(high_dose):
    print("For {:d} Gy".format(dose),end = " ")
    sec_to_min_and_sec(np.mean(time[i]))

from dose_calibration import low_dose_estimation, high_dose_estimation, sec_to_min_and_sec
import numpy as np
import matplotlib.pyplot as plt


T = np.array([5,10,15,20]) #seconds
N_k = 43.77e-3 #Gy/nC
k_u = 1
mu_over_rho = 1.075
P_u = 1.02
k_TP = 1.006745405
C = N_k * k_u * mu_over_rho * P_u * k_TP


"""
D has shape (4,3,4), 4 positions, 4 measurements for 5, 10, 15 and 20 seconds repeated 3 times.
"""
D = np.array([[[0.46,1.52,2.5,3.45] , [0.49,1.56,2.38,3.49] , [0.45,1.45,2.47,3.58]] \
            ,[[0.48,1.4,2.46,3.44] , [0.47,1.39,2.42,3.58] , [0.39,1.32,2.44,3.39]] \
            ,[[0.48,1.4,2.61,3.55] , [0.46,1.59,2.48,3.56] , [0.4,1.54,2.57,3.5]] \
            ,[[0.41,1.57,2.5,3.75],[0.37,1.51,2.6,3.55],[0.41,1.54,2.53,3.68]]]) * C

print(np.ravel(D[0]))

pos_labels = ["A","B","C","D"]

#low dose estimation

low_dose = np.array([0.1,0.2,0.5])
_,mean_time = low_dose_estimation(T,D,low_dose,pos_labels)

print(mean_time)
plt.show()

for i in range(len(low_dose)):
    print("For {:.1f} Gy {:d} s".format(low_dose[i],int(mean_time[i])))


#high dose estimation

high_dose = np.array([1,2,5,10])*1.0256
doserate = np.array([[11.74,11.89,11.71],[11.71,11.62,11.78],[12.14,12.06,12.08],\
[12.26,12.21,12.13]]) * C/60 #dose/s

_,mean_time = high_dose_estimation(high_dose,doserate)

print(mean_time)

for i, dose in enumerate(high_dose):
    print("For {:.1f} Gy".format(dose),end = " ")
    sec_to_min_and_sec(mean_time[i])

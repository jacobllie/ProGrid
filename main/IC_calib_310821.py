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

positions = ["A", "B", "C", "D"]
"""
D has shape (4,3,4), 4 positions, 4 measurements for 5, 10, 15 and 20 seconds repeated 3 times.
"""
output = np.array([[[0.46,1.52,2.5,3.45] , [0.49,1.56,2.38,3.49] , [0.45,1.45,2.47,3.58]] \
            ,[[0.48,1.4,2.46,3.44] , [0.47,1.39,2.42,3.58] , [0.39,1.32,2.44,3.39]] \
            ,[[0.48,1.4,2.61,3.55] , [0.46,1.59,2.48,3.56] , [0.4,1.54,2.57,3.5]] \
            ,[[0.41,1.57,2.5,3.75],[0.37,1.51,2.6,3.55],[0.41,1.54,2.53,3.68]]])

D = output * C


pos_labels = ["A","B","C","D"]

#low dose estimation

low_dose = np.array([0.1,0.2,0.5])
_,mean_time, dt = low_dose_estimation(T,D,low_dose,pos_labels, 3)

print("uncertainty in t is ")
print(dt)

print(mean_time)
plt.close()

for i in range(len(low_dose)):
    print("For {:.1f} Gy {:d} s".format(low_dose[i],int(mean_time[i])))


#high dose estimation

high_dose = np.array([1,2,5,10])*1.0256
output = np.array([[11.74,11.89,11.71],[11.71,11.62,11.78],[12.14,12.06,12.08],\
[12.26,12.21,12.13]])   #nC

means = []
fig,ax = plt.subplots()
for i in range(len(output)):
    ax.plot(np.repeat(i,3),output[i], "*",label = positions[i])
    ax.plot(i,np.mean(output[i]),"o",label = "Mean")
    means.append(np.mean(output[i]))
print("Percentage difference between position A,B and C,D")
AB_mean = np.sum(means[:2])/2
CD_mean = np.sum(means[2:4])/2
print(np.abs(AB_mean - CD_mean)/(AB_mean + CD_mean)/2*100)
ax.legend()
plt.show()


print(output.shape)
doserate = output * C/60 #dose/s


Mu = np.mean(output)  #finding the mean of the three repeated measurements
dMu = np.std(np.mean(output, axis = 1))/np.sqrt(4)  #finding the mean measurement before finding std for higher precision
print(dMu)

dN_k = 0.39/1000 #mGy -----> Gy
print("total error in doserate")




std_tot = C*np.sqrt((Mu*dN_k)**2 + (N_k*dMu)**2)
print("Gy/min standard error")
print(std_tot*60)


#stderr = std_tot/np.sqrt(4)


mean_time, dt = high_dose_estimation(high_dose,doserate, std_tot)

print(mean_time, dt)

for i, dose in enumerate(high_dose):
    print("For {:.1f} Gy".format(dose),end = " ")
    sec_to_min_and_sec(mean_time[i])
    sec_to_min_and_sec(dt[i])

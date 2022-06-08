import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

results_path = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson"

result_files = np.asarray([file for file in os.listdir(results_path) if "MSE_values" in file and "1regressors" not in file])


print(result_files)

"""
2,3,4 regressors, 4 kernel sizes, 4 irradiation configurations
"""
# peak_dist_results = np.zeros((3,4,4))
# peak_area_results = np.zeros((3,4,4))

# peak_dist_results = np.zeros((3,1,4))
# peak_area_results = np.zeros((3,1,4))

peak_dist_results = np.zeros((3,5,4))
peak_area_results = np.zeros((3,5,4))


dist_idx = 0
area_idx = 0
for filename in result_files:
    print(filename)
    data = pd.read_csv(results_path + "\\" + filename)
    print(data)
    """
    Unnecessary but Method is more intuitive than Unnamed: 0
    """
    data = data.rename(columns = {"Unnamed: 0":'Method'})
    print(data.loc[:,data.columns != "Method"].to_numpy().shape)

    """
    We transpose because we want (kernel size, irradiation config), not (irradiation config, kernel_size)
    """
    tmp = data.loc[:,data.columns != "Method"].to_numpy()
    print(tmp.shape)
    """
    We want a separate curve for peak dist and peak area
    """
    if "peak_dist" in filename:
        print(data.loc[:,data.columns != "Method"].to_numpy().shape)
        peak_dist_results[dist_idx] = data.loc[:,data.columns != "Method"].to_numpy().T
        dist_idx += 1
    elif "peak_area" in filename:
        print(data.loc[:,data.columns != "Method"].to_numpy().shape)
        peak_area_results[area_idx] = data.loc[:,data.columns != "Method"].to_numpy().T
        area_idx += 1
    else:
        print(data.loc[:,data.columns != "Method"].to_numpy().shape)
        peak_dist_results[dist_idx] = data.loc[:,data.columns != "Method"].to_numpy().T
        peak_area_results[area_idx] = data.loc[:,data.columns != "Method"].to_numpy().T
        dist_idx += 1
        area_idx += 1


"""
plotting the MSE for each method CTRL, OPEN, STRIPES, DOTS with increasing number of regressors
"""
methods = ["Ctrl","OPEN","GRID Stripes","GRID Dots"]
regressors = [2,3,4]
# kernel_sizes = ["1"]
kernel_sizes = ["0.5","1","2","3", "4"]
colors1 = ["tab:blue", "tab:orange", "tab:cyan", "tab:purple"]
colors2 = ["b","orange","cyan","purple"]
plt.style.use("seaborn")

# fig, ax = plt.subplots(nrows = 2,ncols = 2, figsize = (8,8), sharex = True, sharey = True)
fig, ax = plt.subplots(figsize = (10,9))

"""ax = ax.flatten()
for i in range(peak_dist_results.shape[1]):
    if i == 2 or i == 3:
        ax[i].set_xlabel("Num regressors")
    if i == 0 or i == 2:
        ax[i].set_ylabel("MSE")

    ax[i].set_title(r"Kernel Size {} x {} $mm^2$".format(kernel_sizes[i], kernel_sizes[i]))
    for j in range(peak_dist_results.shape[1]):
        ax[i].plot(regressors,peak_dist_results[:,i,j],"o-" ,label = methods[j] + " peak dist", color = colors1[j])
        ax[i].plot(regressors,peak_area_results[:,i,j], "o-", label = methods[j] + " peak area", color = colors2[j])
    ax[i].legend()
plt.show()
"""

for i in range(peak_dist_results.shape[2]):
    ax.plot(regressors, peak_dist_results[:,1,i], "o-",label = methods[i] + " peak dist",color = colors1[i])
    ax.plot(regressors, peak_area_results[:,1,i], "o-",label = methods[i] + " peak dist",color = colors2[i])
ax.set_xlabel("# regressors", fontsize = 13)
ax.set_ylabel("MSE", fontsize = 13)
ax.set_title(r"Kernel Size {} x {} $mm^2$".format(kernel_sizes[1],kernel_sizes[1]))
ax.legend()
# fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\survival_poisson_MSE_vs_num_regressors.png", dpi = 300)
plt.close()

"""
AIC analysis
"""
"""
#regressors, kernel size
AIC_results_peak_dist = np.zeros((4, 4))
AIC_results_peak_area = np.zeros((4, 4))
 #4 regressors, kernel_size
AIC_results_peak_dist[0] = np.array([275346.19871, 131951.56217, 52061.73740, 27600.01542]) #1 regressor
AIC_results_peak_dist[1] = np.array([273541.13159, 131797.53360, 51792.03314, 27301.57068]) #2 regressors
AIC_results_peak_dist[2] = np.array([274274.14495, 131827.98076, 51770.45236, 27303.51012]) #3 regressors
AIC_results_peak_dist[3] = np.array([274011.35858, 131545.00763, 51672.61410, 27205.04256]) #4 regressors


AIC_results_peak_area[0] = np.array([275346.19871, 131951.56217, 52061.73740, 27600.01542])
AIC_results_peak_area[1] = np.array([273541.13159, 131797.53360, 51792.03314, 27301.57068])
AIC_results_peak_area[2] = np.array([273941.48217, 131690.09444, 51640.25457, 27200.77774])
AIC_results_peak_area[3] = np.array([274011.35858, 131545.00763, 51672.61410, 27205.04256])"""

regressors = [1,2,3,4]


AIC_results_peak_dist = np.zeros((4, 5))
AIC_results_peak_area = np.zeros((4, 5))

AIC_results_peak_dist[0] = np.array([274717.20791,132091.69775,52099.46621,27579.70918,16918.24164])
AIC_results_peak_dist[1] = np.array([273990.02009,131885.56142,51723.51970,27375.69548,16549.70693])
AIC_results_peak_dist[2] = np.array([274354.79814,131648.86821,51710.15997,27225.09691,16504.72546])
AIC_results_peak_dist[3] = np.array([274415.96487,131763.69794,51799.98101,27225.21677,16508.51137])

AIC_results_peak_area[0] = np.array([274717.20791,132091.69775,52099.46621,27579.70918,16918.24164])
AIC_results_peak_area[1] = np.array([273990.02009,131885.56142,51723.51970,27375.69548,16549.70693])
AIC_results_peak_area[2] = np.array([274165.17796,131871.63901,51708.54426,27186.64747,16473.45543])
AIC_results_peak_area[3] = np.array([274415.96487,131763.69794,51799.98101,27225.21677,16508.51137])

# AIC_results_peak_dist[0] = np.array([132091.69775,16918.24164])
# AIC_results_peak_dist[1] = np.array([131885.56142,16549.70693])
# AIC_results_peak_dist[2] = np.array([131648.86821,16504.72546])
# AIC_results_peak_dist[3] = np.array([131763.69794,16508.51137])
#
# AIC_results_peak_area[0] = np.array([132091.69775,16918.24164])
# AIC_results_peak_area[1] = np.array([131885.56142,16549.70693])
# AIC_results_peak_area[2] = np.array([131871.63901,16473.45543])
# AIC_results_peak_area[3] = np.array([131763.69794,16508.51137])

# kernel_sizes = ["1","4"]

fig,ax = plt.subplots(ncols = 3, nrows = 2, figsize=(13,6))
ax = ax.flatten()
for i in range(AIC_results_peak_area.shape[1]):
    ax[i].set_ylabel("AIC")
    ax[i].set_xlabel("# Regressor")
    ax[i].set_title(r"AIC for  {} x {} $mm^2$ quadrat".format(kernel_sizes[i],kernel_sizes[i]))
    ax[i].plot(regressors, AIC_results_peak_dist[:,i], "o-",label = r"Peak distance", color = "b")
    ax[i].plot(regressors, AIC_results_peak_area[:,i], "o-",label = r"Peak area ratio", color = "g")
    ax[i].legend()
    # for j, AIC in enumerate(zip(AIC_results_peak_area[:,i],AIC_results_peak_dist[:,i])):
    #     if j == 2:
    #         ax[i].annotate(np.round(AIC[0]), (regressors[j], AIC_results_peak_area[j][i]), color = "g")
    #     elif j == 3:
    #         ax[i].annotate(np.round(AIC[0]), (regressors[j], AIC_results_peak_area[j][i] + 25), color = "g")
    #     else:
    #         ax[i].annotate(np.round(AIC[0]), (regressors[j] + .5, AIC_results_peak_area[j][i]), color = "g")
    #     ax[i].annotate(np.round(AIC[1]), (regressors[j], AIC_results_peak_dist[j][i]), color = "b")

ax[5].set_visible(False)
fig.tight_layout()
# fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\AIC.png",dpi = 300, bbox_inches = "tight")

plt.show()

print("dAIC peak dist")
fig,ax = plt.subplots(nrows = 2, ncols = 3, figsize = (10,8))
ax = ax.flatten()
for i in range(AIC_results_peak_dist.shape[1]):
    if i == 0 or i == 3:
        ax[i].set_ylabel(r"$\Delta$AIC")
    if i == 3 or i == 4:
        ax[i].set_xlabel("# Regressor")
    AICmin = np.min(AIC_results_peak_dist[:,i])
    dAIC = AIC_results_peak_dist[:,i] - AICmin
    ax[i].set_title("AIC for  {} x {} $mm^2$ quadrat".format(kernel_sizes[i],kernel_sizes[i]))
    ax[i].plot(regressors, dAIC, "o-",label = r"Peak distance")
    ax[i].legend()
    print(r"dAIC {} mm".format(kernel_sizes[i]))
    print(dAIC)
print("-------------------------")
print("dAIC peak area")
for i in range(AIC_results_peak_area.shape[1]):
    if i == 0 or i == 3:
        ax[i].set_ylabel(r"$\Delta$AIC")
    if i == 3 or i == 4:
        ax[i].set_xlabel("# Regressor")

    AICmin = np.min(AIC_results_peak_area[:,i])
    dAIC = AIC_results_peak_area[:,i] - AICmin
    ax[i].set_title("AIC for  {} x {} $mm^2$ quadrat".format(kernel_sizes[i],kernel_sizes[i]))
    ax[i].plot(regressors, dAIC, "o-",label = r"Peak area", alpha = 0.8)
    ax[i].legend()
    print(r"dAIC {} mm".format(kernel_sizes[i]))
    print(dAIC)

ax[5].set_visible(False)
fig.tight_layout()
# fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\dAIC.png",dpi = 300)
plt.close()


#checking how p-value develop with number of datapoints
p_values = [0, 0.023, 0.124, 0.43, 0.980]
num_datapoints = np.sqrt([243360,56542,13680,5760,3022])
# num_datapoints = [243360,56542,13680,5760,3022]

fit = np.polyfit(num_datapoints[::-1],p_values,deg = 1)

ssreg = np.sum((np.polyval(fit, num_datapoints[::-1])-np.mean(p_values))**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((p_values - np.mean(p_values))**2)    # or sum([ (yi - ybar)**2 for yi in y])
R2 = ssreg / sstot

print(R2)




X = np.linspace(np.min(num_datapoints), np.max(num_datapoints),1000)
Y = np.polyval(fit,X)

plt.plot(num_datapoints[::-1],p_values, "o-")
plt.plot(X,Y)
plt.xlabel("# datapoints")
plt.ylabel("p-value")
plt.show()

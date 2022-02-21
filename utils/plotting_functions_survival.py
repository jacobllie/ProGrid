import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from poisson import poisson
from scipy.interpolate import interp1d


def survival_stripes_viz(colony_map, kernel_size, dose_map_reg, cropping_limits):
    pixels_per_mm = 47
    tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-02-gridS-A-SegMask.csv"))
    #tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))
    tmp_image = colony_map[0,0,0] #vizualising first 2 Gy survival image
    #tmp_image = colony_map[0,1,0] #vizualising first 5 Gy survival image

    coor = np.argwhere(tmp_image == 1)

    fig, ax = plt.subplots()
    ax.set_title(r"Surviving Colonies inside {:.1f} $mm^2$ grid with associated dose".format(kernel_size/47),fontsize = 13)
    dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]*2/5, cmap  ="cool")

    #dose = ax.imshow(dose_map_reg[200:2250,300:1750]*2/5, cmap  ="cool") #2 Gy
    # dose = ax.imshow(dose_map_reg[200:2250,300:1750], cmap  ="cool") #5 Gy

    fig.colorbar(dose, label = "Dose [Gy]")
    ax.imshow(tmp_seg_mask[cropping_limits[0]:cropping_limits[1],cropping_limits[2]:cropping_limits[3]], alpha = 0.8)
    ax.plot(coor[:,1],coor[:,0],".",markersize=0.8, color = "black", label = "Colony mass center")
    y = np.arange(0,tmp_image.shape[0],1)  #height in space
    x = np.arange(0,tmp_image.shape[1],1)

    """print(len(y[141::141]))
    print(len([y[i]/(47) for i in range(kernel_size,len(y),kernel_size)]))
    print(len(x[141::141]))
    print(len([x[i]/(47) for i in range(kernel_size,len(x),kernel_size)]))"""

    ax.set_yticks(y[kernel_size::kernel_size])
    # ax.set_yticklabels(["{:.3f}".format(y[i]/47) if i % 2 != 0 else "" for i in range(kernel_size,len(y),kernel_size)],fontsize = 6)
    ax.set_yticklabels(["{:.1f}".format(y[i]/47) for i in range(kernel_size,len(y),kernel_size)],fontsize = 6)

    ax.set_xticks(x[kernel_size::kernel_size])
    ax.set_xticklabels(["{:.1f}".format(x[i]/47) for i in range(kernel_size,len(x),kernel_size)], fontsize = 6, rotation = 60)

    # ax.set_xticklabels(["{:.3f}".format(x[i]/47) if i % 2 != 0 else "" for i in range(kernel_size,len(x),kernel_size)], fontsize = 6, rotation = 60)
    #ax.set_yticklabels([x[i]/(47) for i in range(0,len(x),kernel_size)],fontsize = 7)
    ax.set_xlabel("[mm]",fontsize = 10)
    ax.set_ylabel("[mm]",fontsize = 10)
    ax.grid(True)
    ax.legend()
    # fig.savefig(r"C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\grid_survival5Gy_1811_3mm.png",dpi = 1200, pad_inches = 1)
    plt.close()
    pass

def pooled_colony_hist(pooled_SC_ctrl, SC_grid_2Gy, SC_grid_5Gy, SC_grid_10Gy, kernel_size):
    hist_len = np.max([np.max(np.unique(pooled_SC_ctrl)), np.max(np.unique(SC_grid_2Gy)),
                       np.max(np.unique(SC_grid_5Gy)), np.max(np.unique(SC_grid_10Gy))])
    x_hist = np.arange(0,hist_len +1 ,1)
    print(x_hist)
    count_axis = np.zeros((4,len(x_hist)))
    label_ctrl, count_ctrl = np.unique(np.ravel(pooled_SC_ctrl), return_counts = True)
    var_ctrl = np.var(pooled_SC_ctrl)
    mean_ctrl = np.mean(pooled_SC_ctrl)

    label_2Gy, count_2Gy = np.unique(np.ravel(SC_grid_2Gy), return_counts = True)
    var_2gy = np.var(SC_grid_2Gy)
    mean_2Gy  = np.mean(SC_grid_2Gy)

    label_5Gy, count_5Gy = np.unique(np.ravel(SC_grid_5Gy),return_counts = True)
    var_5Gy = np.var(SC_grid_5Gy)
    mean_5Gy = np.mean(SC_grid_5Gy)

    label_10Gy, count_10Gy = np.unique(np.ravel(SC_grid_10Gy), return_counts = True)
    var_10Gy = np.var(SC_grid_10Gy)
    mean_10Gy = np.mean(SC_grid_10Gy)

    print("Survival mean and variance")
    print(mean_ctrl,var_ctrl,mean_ctrl-var_ctrl)
    print(mean_2Gy,var_2gy,mean_2Gy-var_2gy)
    print(mean_5Gy,var_5Gy,mean_5Gy-var_5Gy)
    print(mean_10Gy, var_10Gy,mean_10Gy-var_10Gy)


    theoretical_mean = [mean_ctrl,mean_2Gy,mean_5Gy,mean_10Gy]
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\sigma^2$")
    plt.plot(np.log([mean_ctrl,mean_2Gy,mean_5Gy,mean_10Gy]), np.log([var_ctrl,var_2gy,var_5Gy,var_10Gy]),"*")
    plt.plot(np.log(theoretical_mean),np.log(theoretical_mean),label = r"$\mu$ =  $\sigma^2$")
    plt.show()
    count_axis[0,label_ctrl.astype(int)] += count_ctrl
    count_axis[1,label_2Gy.astype(int)] += count_2Gy
    count_axis[2,label_5Gy.astype(int)] += count_5Gy
    count_axis[3,label_10Gy.astype(int)] += count_10Gy


    bin_num = len(np.unique(pooled_SC_ctrl))
    hist_x = np.arange(0,21,1)
    x_inter = np.linspace(0,4,100)
    plt.style.use("seaborn")
    fig,ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].set_title("pooled colony survival control")
    ax[0,0].set_xlabel(r"no. colonies within square of {:.1} X {:.1} $mm^2$".format(kernel_size/47, kernel_size/47))
    ax[0,0].set_ylabel("#")
    #plt.subplot(221)
    #plt.title("pooled colony survival control")
    #plt.xlabel(r"no. colonies within square of {} X {} $mm^2$".format(kernel_size/47, kernel_size/47))
    #plt.ylabel("#")
    #n_ctrl, bins, patches = plt.hist(np.ravel(pooled_SC_ctrl),bins = 5,ec = "black")
    #n_ctrl, bins = np.histogram(np.ravel(pooled_SC_ctrl),bins = bin_num)
    #weights = n_ctrl/np.sum(n_ctrl)
    labels, counts = np.unique(pooled_SC_ctrl, return_counts = True)
    #ax[0,0].bar(labels,counts,align = "center")
    ax[0,0].bar(x_hist, count_axis[0],  align = "center")
    # ax[0,0].bar(bins[:-1], weights, width = bins[1] - bins[0], ec = "black", alpha = 0.9, align = "center")
    # plt.bar(bins[:-1], weights, width = bins[1] - bins[0], ec = "black", alpha = 0.9)
    #f = interp1d(hist_x, poisson(np.sum(hist_x*weights),hist_x), kind = "cubic")
    #plt.plot(x_inter,f(x_inter), label = r"Cubic interpolation of Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))
    #plt.plot(hist_x,poisson(np.sum(hist_x*weights),hist_x), label = "Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))




    # plt.subplot(222)
    # plt.title("pooled colony survival 2Gy")
    #plt.xlabel(r"no. colonies within square of {} X {} $mm^2$".format(kernel_size/47))
    #plt.ylabel("#")
    #n_2Gy, bins = np.histogram(np.ravel(SC_grid_2Gy),bins = 5)
    #weights = n_2Gy/np.sum(n_2Gy)
    ax[0,1].set_title("pooled colony survival 2Gy")
    # ax[0,1].bar(bins[:-1], weights, width = bins[1] - bins[0], ec = "black", alpha = 0.9)
    labels, counts = np.unique(SC_grid_2Gy, return_counts = True)
    #ax[0,1].bar(labels,counts,align = "center")
    ax[0,1].bar(x_hist,count_axis[1],align = "center")

    # plt.bar(bins[:-1],weights, width = bins[1] - bins[0], ec = "black", alpha = 0.9)
    #f = interp1d(hist_x, poisson(np.sum(hist_x*weights),hist_x), kind = "cubic")
    #plt.plot(x_inter,f(x_inter), label = r"Cubic interpolation of Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))
    #plt.plot(hist_x,poisson(np.sum(hist_x*weights),hist_x), label = "Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))
    #plt.legend()


    #n_5Gy, bins, patches = plt.hist(np.ravel(SC_grid_5Gy),bins = 5,ec = "black")
    # n_5Gy, bins = np.histogram(np.ravel(SC_grid_5Gy),bins = 5)
    # weights = n_5Gy/np.sum(n_5Gy)
    ax[1,0].set_title("pooled colony survival 5Gy")
    # ax[1,0].bar(bins[:-1], weights, width = bins[1] - bins[0], ec = "black", alpha = 0.9)
    labels, counts = np.unique(SC_grid_5Gy, return_counts = True)
    # ax[1,0].bar(labels,counts,align = "center")
    ax[1,0].bar(x_hist,count_axis[2],align = "center")

    #f = interp1d(hist_x, poisson(np.sum(hist_x*weights),hist_x), kind = "cubic")
    #plt.plot(x_inter,f(x_inter), label = r"Cubic interpolation of Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))
    #plt.plot(hist_x,poisson(np.sum(hist_x*weights),hist_x), label = "Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))
    #plt.legend()

    #n_5Gy, bins, patches = plt.hist(np.ravel(SC_grid_5Gy),bins = 5,ec = "black")
    # n_10Gy, bins = np.histogram(np.ravel(SC_grid_10Gy),bins = 5)
    # weights = n_10Gy/np.sum(n_10Gy)
    ax[1,1].set_title("pooled colony survival 10Gy")
    # ax[1,1].bar(bins[:-1], weights, width = bins[1] - bins[0], ec = "black", alpha = 0.9)
    labels, counts = np.unique(SC_grid_10Gy, return_counts = True)
    # ax[1,1].bar(labels,counts,align = "center")
    ax[1,1].bar(x_hist,count_axis[3],align = "center")

    #f = interp1d(hist_x, poisson(np.sum(hist_x*weights),hist_x), kind = "cubic")
    #plt.plot(x_inter,f(x_inter), label = r"Cubic interpolation of Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))
    #plt.plot(hist_x,poisson(np.sum(hist_x*weights),hist_x), label = "Poisson with $\lambda$ = {:.3}".format(np.sum(hist_x*weights)))
    #plt.legend()
    fig.tight_layout()
    #fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\hist survival\\survival_histogram_02mm.png", bbox_inches = "tight", pad_inches = 0.2, dpi = 1200)
    plt.show()

    pass

def survival_curve_open(nominal_dose, pooled_dose, pooled_survival, mean_ctrl_survival, plot_color):
    SF = np.zeros((pooled_survival.shape[0],pooled_survival.shape[1],len(np.ravel(pooled_dose))))


    mean_dose = np.mean(pooled_dose)


    print(mean_dose)
    """
    finding std for errorbar
    """

    mean_survival = 0
    std_survival = 0
    for i in range(pooled_survival.shape[0]): #date (1811 or 2011)
        for j in range(pooled_survival.shape[1]): #position (A,B,C or D)
            """
            ravel pooled survival of then extracting only survivors with peak dose or valley dose
            divide with mean survivors for ctrl to achieve surviving fraction SF
            """
            SF[i,j] = np.ravel(pooled_survival[i,j])/mean_ctrl_survival

            std_survival += np.std(SF[i,j])/np.sqrt(len(np.ravel(pooled_dose)))
            mean_survival += np.sum(SF[i,j])


            #plt.plot(np.ravel(pooled_dose), np.ravel(SF[i,j]),"*",color = "blue",label = "peak")
            #if i == 0 and j == 0:
            #    plt.legend()


    std_survival /= (pooled_survival.shape[0]*pooled_survival.shape[1])
    #mean survival has to be divided by 2 dates * 4 positions * pooled dose length
    mean_survival /= (pooled_survival.shape[0]*pooled_survival.shape[1]*len(np.ravel(pooled_dose)))

    plt.style.use("seaborn")

    #plt.plot(mean_dose, np.log(mean_survival),"o", color = plot_color,label = "{} Gy nominal".format(nominal_dose))
    #plt.errorbar(mean_dose, np.log(mean_survival),yerr = std_survival,color = plot_color)

    return SF, mean_survival,std_survival, mean_dose

def survival_curve_grid(peak_lower_lim, peak_higher_lim, valley_lower_lim, valley_higher_lim, nominal_dose, pooled_dose, pooled_survival, mean_ctrl_survival, plot_color):
    """
    This functions takes pooled dose and pooled survival from a grid irradiated cell flask
    with either 2, 5 or 10 Gy. It categorizes kernels having either peak or valley
    dose. Then it finds the mean SF fraction and STD to plot a cell survival curve.
    Might potentially make it return plotting object
    """


    if nominal_dose == 2:
        peak_doses = np.argwhere(np.logical_and(peak_lower_lim*2/5 < np.ravel(pooled_dose), np.ravel(pooled_dose) < peak_higher_lim*2/5))
        valley_doses = np.argwhere(np.logical_and(valley_lower_lim*2/5 < np.ravel(pooled_dose), np.ravel(pooled_dose) < valley_higher_lim*2/5))
    elif nominal_dose == 5:
        peak_doses = np.argwhere(np.logical_and(peak_lower_lim < np.ravel(pooled_dose), np.ravel(pooled_dose) < peak_higher_lim))
        valley_doses = np.argwhere(np.logical_and(valley_lower_lim < np.ravel(pooled_dose), np.ravel(pooled_dose) < valley_higher_lim))

    elif nominal_dose == 10:
        peak_doses = np.argwhere(np.logical_and(peak_lower_lim*10/5 < np.ravel(pooled_dose), np.ravel(pooled_dose) < peak_higher_lim*10/5))
        valley_doses = np.argwhere(np.logical_and(valley_lower_lim*10/5 < np.ravel(pooled_dose), np.ravel(pooled_dose) < valley_higher_lim*10/5))

    print("number of peak and valley doses")
    print(len(peak_doses))
    print(len(valley_doses))

    """
    E.g. for 3 mm^2 quadrats we get 20 peak doses and 80 valley doses.
    The argwhere will have shape (20,2) or (80,2)
    We'll use this example for all comments.

    """
    # peak_doses = np.argwhere(np.logical_and(peak_lower_lim < np.ravel(pooled_dose), np.ravel(pooled_dose) < peak_higher_lim))
    # valley_doses = np.argwhere(np.logical_and(valley_lower_lim < np.ravel(pooled_dose), np.ravel(pooled_dose) < valley_higher_lim))

    """
    Shape (2,4,20) or (2,4,80)
    """
    SF_peak = np.zeros((pooled_survival.shape[0],pooled_survival.shape[1],len(peak_doses)))
    SF_valley = np.zeros((pooled_survival.shape[0],pooled_survival.shape[1],len(valley_doses)))

    """
    finding one mean dose for peak and valley
    """
    mean_peak_dose = np.mean(np.ravel(pooled_dose)[peak_doses[:,0]])

    print(mean_peak_dose)
    mean_valley_dose = np.mean(np.ravel(pooled_dose)[valley_doses[:,0]])

    """
    finding std for errorbar for peak and valley
    """
    std = np.zeros(2)
    mean = np.zeros(2)
    counter = 0
    for i in range(pooled_survival.shape[0]): #date (1811 or 2011)
        for j in range(pooled_survival.shape[1]): #position (A,B,C or D)
            """
            ravel pooled survival of then extracting only survivors with peak dose or valley dose
            divide with mean survivors for ctrl to achieve surviving fraction SF
            """
            SF_peak[i,j] = np.ravel(pooled_survival[i,j])[peak_doses[:,0]]/mean_ctrl_survival
            SF_valley[i,j] = np.ravel(pooled_survival[i,j])[valley_doses[:,0]]/mean_ctrl_survival

            std[0] += np.std(SF_peak[i,j])/np.sqrt(len(peak_doses))
            std[1] += np.std(SF_valley[i,j])/np.sqrt(len(valley_doses))
            mean[0] += np.sum(SF_peak[i,j])
            mean[1] += np.sum(SF_valley[i,j])

            #plt.plot(np.ravel(pooled_dose)[peak_doses[:,0]], SF_peak[i,j],"*",color = "blue",label = "peak")
            #plt.plot(np.ravel(pooled_dose)[valley_doses[:,0]], SF_valley[i,j],"*", color = "red",label = "valley")
            #if i == 0 and j == 0:
            #    plt.legend()

    #plt.show()
    #dividing with 2*4*sqrt(N) to get average standard deviation of the mean for nominal X Gy dose
    std[0] /= (pooled_survival.shape[0]*pooled_survival.shape[1])
    std[1] /= (pooled_survival.shape[0]*pooled_survival.shape[1])
    """
    dividing with 2*4*20 or 2*4*80 to get mean surviving fraction for dose X
    """
    mean[0] /= (pooled_survival.shape[0]*pooled_survival.shape[1]*len(peak_doses))
    mean[1] /= (pooled_survival.shape[0]*pooled_survival.shape[1]*len(valley_doses))
    plt.style.use("seaborn")

    plt.plot(mean_peak_dose, np.log(mean[0]),"o", color = plot_color,label = "{}Gy peak".format(nominal_dose))
    plt.plot(mean_valley_dose, np.log(mean[1]),"o", color = plot_color, label = "{}Gy valley".format(nominal_dose))
    plt.errorbar(mean_peak_dose, np.log(mean[0]),yerr = std[0],color = plot_color)
    plt.errorbar(mean_valley_dose, np.log(mean[1]), yerr = std[1], color = plot_color)

    return mean,std, mean_peak_dose, mean_valley_dose

def pred_vs_true_SC(true_SC,predicted_SC,true_SC_grid,predicted_SC_grid,true_SC_open,predicted_SC_open):
    fig,ax = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(16, 9)
    ax[0].set_title("GRID&OPEN stacked")
    ax[0].set_xlabel("Dose [Gy]")
    ax[0].set_ylabel("SC")
    ax[0].plot(true_SC,"*", label = "True Mean")
    ax[0].plot(predicted_SC,"*",label = "Predicted mean")
    ax[0].plot(np.abs(predicted_SC-true_SC)/true_SC, label = "Relative Error")
    ax[0].legend()

    ax[1].set_title("GRID")
    ax[1].set_xlabel("Dose [Gy]")
    ax[1].set_ylabel("SC")
    ax[1].plot(true_SC_grid,"*", label = "True Mean")
    ax[1].plot(predicted_SC_grid,"*",label = "Predicted mean")
    ax[1].plot(np.abs(predicted_SC_grid-true_SC_grid)/true_SC_grid, label = "Relative Error")
    ax[1].legend()

    ax[2].set_title("OPEN")
    ax[2].set_xlabel("Dose [Gy]")
    ax[2].set_ylabel("SC")
    ax[2].plot(true_SC_open,"*", label = "True Mean")
    ax[2].plot(predicted_SC_open,"*",label = "Predicted mean")
    ax[2].plot(np.abs(predicted_SC_open-true_SC_open)/true_SC_open, label = "Relative Error")
    ax[2].legend()
    #fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\plots\\survival 301121\\Poisson regression\\mean_survival_error_39mm_w.G_factor.png",dpi = 1200)
    plt.close()
    pass

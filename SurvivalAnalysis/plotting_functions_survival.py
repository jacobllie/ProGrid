import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson, gamma, chisquare


def survival_viz(colony_map, kernel_size, dose_map_reg, cropping_limits, Stripes):
    pixels_per_mm = 47
    doi = 10  #dose of interest
    if Stripes == True:
        if doi == 2:
            tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\GRID Stripes\\A549-1811-02-gridS-A-SegMask.csv"))
            tmp_image = colony_map[0,0,0]
        elif doi == 5:
            tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))
            tmp_image = colony_map[0,1,0]
        elif doi == 10:
            tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\18112019\\GRID Stripes\\A549-1811-10-gridS-A-SegMask.csv"))
            tmp_image = colony_map[0,2,0]

    elif Stripes != True:
        if doi == 2:
            tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\20112019\\GRID Dots\\A549-2011-02-gridC-A-SegMask.csv"))
            tmp_image = colony_map[0,0,0]
        elif doi == 5:
            tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\20112019\\GRID Dots\\A549-2011-05-gridC-A-SegMask.csv"))
            tmp_image = colony_map[0,1,0]
        elif doi == 10:
            tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 23.11.2021\\20112019\\GRID Dots\\A549-2011-10-gridC-A-SegMask.csv"))
            tmp_image = colony_map[0,2,0]
    #tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))


    coor = np.argwhere(tmp_image == 1)

    fig, ax = plt.subplots(figsize = (8,8))
    ax.set_title(r"Surviving Colonies inside {:.1f} X {:.1f} $mm^2$ grid with associated dose".format(kernel_size/47, kernel_size/47),fontsize = 13)

    """if doi == 2:
        dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]*2/5, cmap  ="viridis")
    elif doi == 5:
        dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]], cmap  ="viridis")
    elif doi == 10:
        dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]*10/5, cmap  ="viridis")"""


    # dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]], cmap  ="cool")
    # dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]*10/5, cmap  ="cool")

    #dose = ax.imshow(dose_map_reg[200:2250,300:1750]*2/5, cmap  ="cool") #2 Gy
    # dose = ax.imshow(dose_map_reg[200:2250,300:1750], cmap  ="cool") #5 Gy

    # fig.colorbar(dose, label = "Dose [Gy]")
    ax.imshow(tmp_seg_mask[cropping_limits[0]:cropping_limits[1],cropping_limits[2]:cropping_limits[3]])#, alpha = 0.8)
    ax.plot(coor[:,1],coor[:,0],".",markersize=4, color = "magenta", label = "Colony mass center")
    leg = ax.legend(loc = "upper right", frameon = True, borderpad = 1, markerscale = 7)
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
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
    ax.set_xlabel("[mm]",fontsize = 15)
    ax.set_ylabel("[mm]",fontsize = 15)
    ax.grid(True)
    if doi == 2:
        dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]*2/5, cmap  ="viridis",alpha = 0.7)
    elif doi == 5:
        dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]], cmap  ="viridis", alpha = 0.7)
    elif doi == 10:
        dose = ax.imshow(dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]*10/5, cmap  ="viridis", alpha = 0.7)
    fig.colorbar(dose, label = "Dose [Gy]")
    #legend.get_frame().set_edgecolor('b')
    # frame = legend.get_frame()
    # frame.set_facecolor('green')
    # frame.set_edgecolor('red')
    pass

def survival_histogram(dose_map, pooled_dose, pooled_survival, dose, kernel_size,mode = "GRID"):

    if mode == "Control":
        n, bins, patches = plt.hist(pooled_survival.ravel(), bins = len(np.unique(pooled_survival)), density = True,
                                    facecolor='magenta', alpha = 0.8, edgecolor = "black", align = "left")
        plt.close()
        return len(pooled_survival[pooled_survival < 0])
    else:
        d70 = np.max(dose_map)*0.70  #dividing by 0.8 because OD is opposite to dose
        # d70_idx  = np.abs(dose_map-d70).argmin()
        d15 = np.min(dose_map)*1.15

        peak_dose = pooled_dose[pooled_dose > d70]
        valley_dose = pooled_dose[pooled_dose < d15]
        peak_survival =pooled_survival[:,:,pooled_dose > d70].flatten()
        valley_survival = pooled_survival[:,:,pooled_dose < d15].flatten()


        #expected_peak = poisson.pmf(np.unique())


        # bins_peak = len(np.unique(peak_survival))
        # bins_valley = len(np.unique(valley_survival))
        #pois_peak = np.arange(np.min(peak_survival), np.max(peak_survival),1)
        #pois_valley = np.arange(np.min(valley_survival), np.max(valley_survival),1)

        #chi squared test

        colonies_peak,occurences_peak = np.unique(peak_survival, return_counts = True)

        #numpy unique returns 18.99999999999 not 19
        expected_peak = poisson.pmf(colonies_peak.round(0).astype(int), np.mean(peak_survival))
        p_value_peak = chisquare(occurences_peak/np.sum(occurences_peak), expected_peak)[1]

        colonies_valley,occurences_valley = np.unique(valley_survival, return_counts = True)
        expected_valley = poisson.pmf(colonies_valley.round(0).astype(int), np.mean(valley_survival))
        p_value_valley = chisquare(occurences_valley/np.sum(occurences_valley), expected_valley)[1]

        #bug fixing

        if dose == 10 and kernel_size == 0.5:
            print(expected_peak)
            print(colonies_peak)
            print(occurences_peak)
            """plt.plot(peak_survival, "*")
            plt.show()"""
            print(np.mean(valley_survival))
            # print(colonies_valley[10].astype(int))


        #plotting observed vs pred
        """plt.plot(colonies,occurences/np.sum(occurences), "*", label = "observed")
        plt.plot(colonies, expected_peak, "o", label = "Expected")
        plt.legend()
        plt.show()"""





        #print(chisquare(peak_survival), poisson.pmf(peak_survival,np.mean(peak_survival)))

        #plt.show()

        """Include theoretical values"""


        # rel_diff_peak = np.abs(np.mean(peak_survival)-np.var(peak_survival))/np.mean(peak_survival)
        # rel_diff_valley = np.abs(np.mean(valley_survival)-np.var(valley_survival))/np.mean(valley_survival)
        rel_diff_peak = np.abs(np.mean(peak_survival)-np.var(peak_survival))/(np.mean(peak_survival) + np.var(peak_survival))/2   #relative percentage difference
        rel_diff_valley = np.abs(np.mean(valley_survival)-np.var(valley_survival))/(np.mean(valley_survival) + np.var(peak_survival))/2

        bins_peak = np.unique(peak_survival)
        bins_peak = np.append(bins_peak,np.max(bins_peak) + 1) #including right bin edge
        bins_valley = np.unique(valley_survival)
        bins_valley = np.append(bins_valley,np.max(bins_valley) + 1)


        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (9,8))
        plt.suptitle(r"Dose {} Gy Quadrat size  {} x {} $mm^2$".format(dose, kernel_size, kernel_size), fontsize = 15)
        ax[0].set_title(r"Peak Survival $\bar{x}$ = %.5f , $s^2$ = %.5f" %(np.mean(peak_survival),
                  np.var(peak_survival)), fontsize = 15)
        n, bins, patches = ax[0].hist(peak_survival.ravel(), bins = bins_peak, density = True,
                                    facecolor='magenta', alpha = 0.8, edgecolor = "black", align = "left", label = "data")
        ax[0].plot(colonies_peak, expected_peak, label = "theoretical")
        #plt.plot(pois_peak, gamma.pdf(pois_peak, np.mean(peak_survival))) #gamma distribution
        ax[0].set_xticks(bins_peak)
        ax[0].set_xlabel("# surviving colonies", fontsize = 15)
        ax[0].set_ylabel("occurences", fontsize = 15)
        ax[0].legend(fontsize = 15)


        ax[1].set_title(r"Valley Survival $\bar{x}$ = %.5f , $s^2$ = %.5f" %(np.mean(valley_survival),
                  np.var(valley_survival)), fontsize = 15)
        n, bins, patches = ax[1].hist(valley_survival.ravel(), bins = bins_valley,
                                    density = True,facecolor='cyan', alpha = 0.8,
                                    edgecolor = "black", align = "left", label = "data")
        ax[1].plot(colonies_valley, expected_valley, label = "theoretical")
        #plt.plot(pois_valley, gamma.pdf(pois_valley, np.mean(valley_survival))) #gamma distribution

        ax[1].set_xlabel("# surviving colonies",fontsize = 15)
        ax[1].set_ylabel("occurences", fontsize = 15)
        ax[1].set_xticks(bins_valley)
        ax[1].legend(fontsize = 15)
        fig.tight_layout()

        fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\poisson_eval_histogram_{}mm_{}Gy_2.png".format(kernel_size, dose), pad_inches = 0.5, dpi = 300)
        # fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\Poisson\\poisson_eval_histogram_{}_{}Gy.png".format(kernel_size, dose), pad_inches = 0.5, dpi = 1200)
        if kernel_size == 4 and dose == 10:
            plt.close()
        else:
            plt.close()

    print([[rel_diff_peak,p_value_peak],[rel_diff_valley, p_value_valley]])
    return np.array([[rel_diff_peak,p_value_peak],[rel_diff_valley, p_value_valley]])
    # return [rel_diff_peak,rel_diff_valley]

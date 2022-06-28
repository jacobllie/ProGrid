import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def low_dose_estimation(T,D,wanted_dose,pos_labels, repetitions):
    plt.style.use("seaborn")
    time = np.zeros((D.shape[0],D.shape[1]))
    mean_slope = 0
    mean_intercept = 0
    r2 = np.zeros(len(D))
    stderr_slope = np.zeros((len(D)))
    stderr_intercept = np.zeros((len(D)))
    stack_T = np.tile(T,repetitions)
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (10,6))
    ax = ax.flatten()

    for i in range(len(D)):
        print(np.ravel(D[i]))
        n = len(np.ravel(D[i]))
        if i == 2 or i == 3:
            ax[i].set_xlabel("Time [s]")
        if i == 0 or i  == 2:
            ax[i].set_ylabel("Dose [Gy]")

        ax[i].plot(stack_T,np.ravel(D[i]),"o", label = "Data")
        #plt.plot(T,D[i,3],"p",label="Data 4")


        #stack_T = np.append(np.append(np.append(T,T),T),T)

        #np.append(T,[T,T])

        Y_A = stats.linregress(stack_T,np.ravel(D[i]))

        ax[i].set_title(r"Position {}, $R^2$ {:.4}".format(pos_labels[i], Y_A.rvalue**2))
        slope = Y_A.slope
        intercept = Y_A.intercept
        mean_slope += slope
        mean_intercept += intercept
        stderr_slope[i] = Y_A.stderr
        stderr_intercept[i] = Y_A.intercept_stderr
        r2[i] = Y_A.rvalue**2
        Y = stack_T*slope + intercept
        T_ = np.linspace(5,20,100)
        Y_ = T_*slope + intercept
        # stderr_estimate[i] = np.sum((np.ravel(D[i]) - (stack_T*slope + intercept))**2)/(len(D)-2) #-2 because we fit two parameters
        #dD_array[i] = np.sum(dD**2, axis = 0)/D.shape[0] #finding mean standard error of dose for all positions to get one standard error for 5 s 10 s etc.
        #print(((wanted_dose*1.0256 - intercept)/slope).shape)
        #time[i] = (wanted_dose*1.0256 - intercept)/slope #Time for individual positions

        ax[i].plot(stack_T,Y,label=r"D = {:.4f} ($\pm$ {:.4f})T + {:.5f} ($\pm$ {:.4f})".format(slope,Y_A.stderr,intercept,Y_A.intercept_stderr),linewidth = 1)
        MSE = np.sum((Y - np.ravel(D[i]))**2)/(n-2)

        conf = stats.t.ppf(0.95, n - 2) * np.sqrt(MSE*(1/n + (T_ - np.mean(T_))**2/np.sum((T_ - np.mean(T_))**2)))
        # plt.plot(T_,conf)
        ax[i].plot(T_, Y_ - conf, "--", linewidth = .5, c = "r", label = "95% CI")
        ax[i].plot(T_,Y_ + conf, linestyle = "--", linewidth = .5, c = "r")
        ax[i].legend(fontsize = 7)
    # fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\IC Calibration\\310821\\low_dose_regression.png", pad_inches = 1,dpi = 1200)
    plt.show()
    mean_slope /= len(D)
    mean_intercept /= len(D)
    mean_r2 = np.mean(r2)
    stderr_r2 = np.std(r2)/np.sqrt(len(D))
    mean_stderr_slope = np.sqrt(np.sum(stderr_slope**2)/(len(D))) #must square to sum errors
    mean_stderr_intercept = np.sqrt(np.sum(stderr_intercept**2))/(len(D))
    # mean_stderr_estimate = np.sqrt(np.sum(stderr_estimate**2)/(len(D)-2))
    print("mean slope with stderr")
    print(mean_slope*60, mean_stderr_slope)


    print("sfdjhsfdhf")
    print(mean_slope)
    print(mean_intercept)

    mean_time = (wanted_dose * 1.0256 - mean_intercept)/mean_slope
    print(mean_time, mean_intercept,mean_slope)
    dt = np.sqrt((-1/mean_slope*mean_stderr_intercept)**2 + (-(wanted_dose*1.026-mean_intercept)/mean_slope**2*mean_stderr_slope)**2)  #no uncertainty in
    print(dt)
    stack_T = np.tile(T, D.shape[0]*D.shape[1])
    Y = (stack_T*mean_slope) + mean_intercept
    T_ = np.linspace(5,20,100)
    Y_ = T_*mean_slope + mean_intercept
    MSE = np.sum((Y - np.ravel(D))**2)/(n-2)

    n = len(np.ravel(D))
    conf = stats.t.ppf(0.95, n - 2) * np.sqrt(MSE*(1/n + (T_ - np.mean(T_))**2/np.sum((T_ - np.mean(T_))**2)))
    plt.title(r"Mean fit with mean $R^2$ = {:.8f} $\pm$ {:f}".format(mean_r2, stderr_r2))
    plt.xlabel(" Time [s]")
    plt.ylabel("Dose [Gy]")
    plt.plot(T_,Y_,label = "D = {:.4f} ($\pm$ {:.4f})T + {:.4f} ($\pm$ {:.4f})".format(mean_slope, mean_stderr_slope,mean_intercept, mean_stderr_intercept))
    plt.plot(stack_T, np.ravel(D), "*", label = "All data")
    plt.fill_between(T_, Y_ - conf, Y_ + conf, alpha = 0.5, color = "cyan")
    #plt.plot(T_, Y_ - conf, "--", linewidth = .5, c = "r", label = "95% CI")
    #plt.plot(T_,Y_ + conf, linestyle = "--", linewidth = .5, c = "r")
    plt.legend()
    # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\IC Calibration\\310821\\mean_fit.png", pad_inches = 0.2, bbox_inches = "tight", dpi = 1200)
    plt.show()
    """plt.plot(T*mean_slope + mean_intercept,T,label = "T = {:.4f} * (D + {:.4f})".format(1/mean_slope,-mean_intercept))
    plt.legend()
    plt.show()"""
    return time, mean_time, dt



def high_dose_estimation(wanted_dose,doserate, std_doserate):
    """
    This function divides each wanted dose [Gy] with doserate [Gy/m] to get number
    of minutes required to achieve a certain dose x. This is done for each position.
    time therefore has shape (4,4). 4 wanted high doses, 4 positions.
    """
    #time = np.zeros((len(wanted_dose),4))
    mean_time = np.zeros(len(wanted_dose))
    mean_doserate =  np.mean([np.mean(doserate[j]) for j in range(len(doserate))])*1.0256
    dt = np.zeros(len(wanted_dose))
    print("mean doserate Gy/min")
    print(mean_doserate*60)
    print(mean_doserate,std_doserate)

    print("Mean doserate per m over all positions and all repetitions is {:.5f} Gy/s".format(mean_doserate))
    for i,dose in enumerate(wanted_dose):
        #time[i] = [dose/np.mean(doserate[j]) for j in range(len(doserate))]  #er det riktig å ta gjennomsnittet av alle outputmålingene?
        mean_time[i] = dose/mean_doserate
        dt[i] = np.sqrt((dose*1.026/mean_doserate**2*std_doserate)**2)#/np.sqrt(doserate.shape[0]*doserate.shape[1])  #divide with number of positions*repeated measurements
        print(dose)
    return mean_time, dt


def sec_to_min_and_sec(s):
    check_int = isinstance(s, list)
    if check_int == False:
        minutes = s // 60 #fraction of minute
        seconds = (s/60 - minutes)*60
        #print("{:.2f} sekunder gir:".format(s))
        print("{:d} min {:d} sec".format(int(minutes),int(seconds)))
    else:
        for i in range(len(s)):
            minutes = s[i] // 60 #fraction of minute
            seconds = (s[i]/60 - minutes)*60
            #print("{:.2f} sekunder gir:".format(s[i]))
            print("{:d} min {:d} sec".format(int(minutes),int(seconds)))

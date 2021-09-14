import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def low_dose_estimation(T,D,wanted_dose,pos_labels):
    plt.style.use("seaborn")
    time = np.zeros((4,3))
    mean_slope = 0
    mean_intercept = 0
    for i in range(len(D)):
        plt.subplot(2,2,i+1)
        plt.title("Position {}".format(pos_labels[i]))
        plt.plot(T,D[i,0],"*",label="Data 1")
        plt.plot(T,D[i,1],"o",label="Data 2")
        plt.plot(T,D[i,2],".",label="Data 3")
        plt.legend()

        tripple_T = np.append(np.append(T,T),T)


        Y_A = stats.linregress(tripple_T,np.ravel(D[i]))

        slope = Y_A.slope
        intercept = Y_A.intercept
        mean_slope += slope
        mean_intercept += intercept

        time[i] = (wanted_dose*1.0256 - intercept)/slope #Time for individual positions

        plt.plot(tripple_T,tripple_T*slope + intercept,label="D = {:.4f}T + {:.4f}".format(slope,intercept),linewidth = 1)
        plt.legend()
    plt.show()
    mean_slope /= len(D)
    mean_intercept /= len(D)
    print(mean_intercept)
    mean_time = (wanted_dose * 1.0256 - mean_intercept)/mean_slope
    plt.plot(tripple_T,(tripple_T*mean_slope) + mean_intercept,label = "D = {:.4f}T + {:.4f}".format(mean_slope,mean_intercept))
    plt.legend()
    plt.show()
    return time, mean_time



def high_dose_estimation(wanted_dose,doserate):
    """
    This function divides each wanted dose [Gy] with doserate [Gy/m] to get number
    of minutes required to achieve a certain dose x. This is done for each position.
    time therefore has shape (4,4). 4 wanted high doses, 4 positions.
    """
    time = np.zeros((len(wanted_dose),4))
    mean_time = np.zeros(len(wanted_dose))
    mean_doserate =  np.mean([np.mean(doserate[j]) for j in range(len(doserate))])
    print("Mean doserate per second over all positions and all repetitions is {:.3f} Gy/s".format(mean_doserate))
    for i,dose in enumerate(wanted_dose):
        time[i] = [dose/np.mean(doserate[j]) for j in range(len(doserate))]  #er det riktig å ta gjennomsnittet av alle outputmålingene?
        mean_time[i] = dose/mean_doserate
    return time,mean_time


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

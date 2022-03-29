import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def low_dose_estimation(T,D,wanted_dose,pos_labels):
    plt.style.use("seaborn")
    time = np.zeros((D.shape[0],D.shape[1]))
    mean_slope = 0
    mean_intercept = 0
    mean_stderr_slope = 0
    mean_stderr_intercept = 0
    for i in range(len(D)):
        plt.subplot(2,2,i+1)
        plt.plot(T,D[i,0],"*",label="Data 1")
        plt.plot(T,D[i,1],"o",label="Data 2")
        plt.plot(T,D[i,2],".",label="Data 3")
        #plt.plot(T,D[i,3],"p",label="Data 4")
        plt.legend()

        #quadruple_T = np.append(np.append(np.append(T,T),T),T)

        quadruple_T = np.append(T,[T,T])

        Y_A = stats.linregress(quadruple_T,np.ravel(D[i]))

        plt.title(r"Position {}, $R^2$ {:.4}".format(pos_labels[i], Y_A.rvalue**2))
        slope = Y_A.slope
        intercept = Y_A.intercept
        mean_slope += slope
        mean_intercept += intercept
        mean_stderr_slope += Y_A.stderr
        mean_stderr_intercept += Y_A.intercept_stderr
        #print(((wanted_dose*1.0256 - intercept)/slope).shape)
        #time[i] = (wanted_dose*1.0256 - intercept)/slope #Time for individual positions

        plt.plot(quadruple_T,quadruple_T*slope + intercept,label=r"D = {:.4f} ($\pm$ {:.4f})T + {:.5f} ($\pm$ {:.4f})".format(slope,Y_A.stderr,intercept,Y_A.intercept_stderr),linewidth = 1)
        plt.legend()
    plt.show()
    mean_slope /= len(D)
    mean_intercept /= len(D)
    mean_stderr_slope /= len(D)
    mean_stderr_intercept /= len(D)

    print("sfdjhsfdhf")
    print(mean_slope)
    print(mean_intercept)

    mean_time = (wanted_dose * 1.0256 - mean_intercept)/mean_slope
    plt.plot(T,(T*mean_slope) + mean_intercept,label = "D = {:.4f} ($\pm$ {:.4f})T + {:.4f} ($\pm$ {:.4f})".format(mean_slope, mean_stderr_slope,mean_intercept, mean_stderr_intercept))
    plt.legend()
    plt.show()
    plt.plot(T*mean_slope + mean_intercept,T,label = "T = {:.4f} * (D + {:.4f})".format(1/mean_slope,-mean_intercept))
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
    mean_doserate =  np.mean([np.mean(doserate[j]) for j in range(len(doserate))])*1.0256
    print("mean doserate Gy/min")
    print(mean_doserate*60)
    print("Mean doserate per second over all positions and all repetitions is {:.5f} Gy/s".format(mean_doserate))
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

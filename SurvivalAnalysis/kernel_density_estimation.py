from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import numpy as np
import matplotlib.pyplot as plt

def kde(data, bandwidth, num_min):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit((data).reshape(-1,1))
    s = np.linspace(np.min(data),np.max(data),100).reshape(-1,1)
    kde_scores = kde.score_samples(s)
                #Henter lokal minima for red channel
    mi = argrelextrema(kde_scores, np.less)[0]
    plt.plot(s,kde_scores)
    plt.show()
    #print(mi)
    if num_min == 1:
        low_res_idx = np.argwhere((data) < s[mi[0]])
        high_res_idx = np.argwhere((data) > s[mi[0]])
        return low_res_idx, high_res_idx
    elif num_min == 3:
        min_min = np.argwhere(s[mi[0]])
        min_max = np.argwhere(np.logical_and(s[mi[0]] < data, data < s[mi[1]]))
        max_min = np.argwhere(np.logical_and(s[mi[1]] < data, data < s[mi[2]]))
        max_max = np.argwhere(s[mi[2]] < data)
        return min_min, min_max, max_min, max_max

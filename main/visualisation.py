import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

def K_means(dose, n_clusters,x,y):
    kmeans = KMeans(n_clusters = n_clusters)
    df = pd.DataFrame({"dose{}Gy".format(dose):np.ravel(x),"SF_{}Gy".format(dose):np.ravel(y)})
    fit = kmeans.fit(df)
    df["Clusters"] = fit.labels_
    if dose == 2:
        idx_1 = np.argwhere(np.logical_and(0.33 < x, x < 0.41))
        idx_2 = np.argwhere(np.logical_and(0.52 < x, x < 0.70))
        idx_3 = np.argwhere(np.logical_and(0.8 < x, x < 1.20))
        idx_4 = np.argwhere(np.logical_and(1.5 < x, x < 1.8))


        cluster_center1 = [np.mean(x[idx_1[:,0]]),np.mean(y[idx_1[:,0]])]
        cluster_center2 = [np.mean(x[idx_2[:,0]]),np.mean(y[idx_2[:,0]])]
        cluster_center3 = [np.mean(x[idx_3[:,0]]),np.mean(y[idx_3[:,0]])]
        cluster_center4 = [np.mean(x[idx_4[:,0]]),np.mean(y[idx_4[:,0]])]
        print(cluster_center1, cluster_center2)
        return df, cluster_center1, cluster_center2, cluster_center3, cluster_center4

    else:
        return df, fit.cluster_centers_[:,0],fit.cluster_centers_[:,1]

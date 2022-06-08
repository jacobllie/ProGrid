import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import random
from scipy.stats import norm
from scipy.interpolate import interp1d
import seaborn as sb
plt.style.use("seaborn")

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(41)



N = 100
x = np.random.gamma(1.5, size = N)
x_ = x[::10]
bins = np.arange(10)

w_ = [0.5,0.1]
downscale_factor = [0.05,0.01]
fig,ax = plt.subplots(1,2, sharex = True, sharey = True)
fig.set_size_inches(18.5, 10.5)
for i,w in enumerate(w_):
    if i == 0:
        ax[i].set_xlabel("x [a.u.]")
        ax[i].set_ylabel("y [a.u.]")
    ax[i].set_title("Bandwidth = {}".format(w_[i]))
    KDE = stats.gaussian_kde(x)
    xx = np.linspace(0, np.max(x), 1000)

    n,bins = np.histogram(x, density=True, bins=bins)

    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(x[:,None])
    s = np.linspace(0,np.max(x),1000)
    kde_scores = kde.score_samples(s[:,None])

    ax[i].plot(s,np.exp(kde_scores))

    ax[i].plot(xx, KDE(xx))

    print(bins)
    print(n)

    #ax[i].fill_between(xx,kde(xx),alpha = 0.6)

    ax[i].plot(x_, np.repeat(0,len(x_)), "*")


    #f = kde.covariance_factor()
    #bw = f * x.std()

    #print(bw)


    #for j in range(len(x_)):
        #x_axis = np.linspace(x_[j]-w*4,x_[j] + w*4,50)
        #kernel= norm.pdf(x_axis,x_[j], w)


        #ax[i].plot(x_axis,kernel*downscale_factor[i])
# fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\KDE_ilustration.png",bbox_inches = "tight",pad_inches = 0.5, dpi = 1200)
plt.show()



random.seed(100)
sample = np.random.gamma(shape = 1.5,size = 1000)

x = np.random.choice(sample, 10, replace = False)#.reshape(-1,1)
x = np.insert(x,0,0)

values, counts = np.unique(x, return_counts=True)

y = [np.repeat(value, count) for value,count in zip(values,counts)]
y = [item for sublist in y for item in sublist]



"""
If no kernel then remain 0
"""
x_ = np.linspace(0,np.max(x),100)
kernel_sum = np.zeros(100)


#pseudo code
dist = np.zeros((len(x),len(x)))
for i, elem in enumerate(x):

    """
    For each point in x, we find the distance to the other points
    """
    dist[i] = np.abs(elem - x)


kernel = np.zeros((len(x),50))
kernel_ = np.zeros((len(x),50))
sum = np.zeros((len(x),len(x)))
kde_x_axis = np.linspace(np.min(x),np.max(x),50)
bandwidth = [0.05,0.5]
kde_value = np.zeros((len(x),len(x)))
for c,w in enumerate(bandwidth):
    #print(c)
    plt.subplot(1,2,c+1)
    for i in range(len(x)):
        x_axis = np.linspace(x[i]-w*2,x[i] + w*2,50)


        kernel[i] = norm.pdf(x_axis,x[i], w)
        kernel_[i] = norm.pdf(kde_x_axis,x[i], w)

        #print(np.repeat(x,10))
        #nearest_x = np.argmin(np.abs(x[i] - kde_x_axis))

        #x_ = kde_x_axis[nearest_x]

        #print(x[nearest_x],kernel_[i,nearest_x])


        #weight = np.sum(dist[i]/w)  #weight of the first point
        #sum[i] = kernel_[i,x_]*weight   #kernel value at that point

        #print(sum.shape)
        plt.plot(x_axis, kernel[i])

    for i in range(len(x)):
        dist = np.abs(x[i] - x)
        for j,d in enumerate(dist):
            if j == i:
                print("skip")
            else:
                nearest_x = np.argmin(np.abs(kde_x_axis - x[i]))
                kde_value[i,j] = kernel_[j,nearest_x]  #(0,0), (1,1) etc. is zero
                # print(kde_value[i,j])
                weight = np.sum(dist[j]/w)  #weight of the first point
                sum[i,j] = kde_value[i,j]*weight   #kernel value at that point



    #n,bins = np.histogram(np.repeat(x,10), bins = 5)
    #print(n)
    #tmp = np.linspace(0,bins[-1],len(n))
    #f = interp1d(tmp,n, kind = "cubic")





    #plt.fill_between(x_, np.sum(kernel,axis = 0), alpha = 0.5)



    #print(x_)
    #print(tmp)
    kde = KernelDensity(kernel = "gaussian", bandwidth = w).fit(x.reshape(-1,1))
    s = np.linspace(np.min(x), np.max(x),100).reshape(-1,1)
    kde_scores = kde.score_samples(s)
    plt.plot(x, np.repeat(0,len(x)), "*")
    plt.plot(0,40)
plt.show()
print(np.sum(sum,axis = 1))
plt.fill_between(x,np.sum(sum,axis = 1), "*")

plt.show()




kde = KernelDensity(kernel = "gaussian", bandwidth = bandwidth).fit(x)


#s = np.linspace(np.min(mean_netOD), np.max(mean_netOD),1000).reshape(-1,1)
#kde_scores = kde.score_samples(s)
x = np.arange(-1,1,0.001)
y = norm.pdf(x)

plt.plot(x,y)
plt.show()

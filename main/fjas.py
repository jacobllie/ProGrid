import string
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
import cv2
import skimage.transform as tf
from crop_pad import crop, pad
import torch.nn as nn
import torch
import sys
from scipy.optimize import least_squares
from kapteyn import kmpfit

"""from numpy.random import default_rng
rng = default_rng()
def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
    rng = default_rng(seed)

    y = a + b * np.exp(t * c)

    error = noise * rng.standard_normal(t.size)
    outliers = rng.integers(0, t.size, n_outliers)
    error[outliers] *= 10

    return y + error

a = 0.5
b = 2.0
c = -1
t_min = 0
t_max = 10
n_points = 15

t_train = np.linspace(t_min, t_max, n_points)
y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)



def fun(x, t, y):
    return x[0] + x[1] * np.exp(x[2] * t) - y


x0 = np.array([1.0, 1.0, 0.0])

res_lsq = least_squares(fun, x0, args=(t_train, y_train))"""



def func(params,x):
    a,b,c = params
    return  a * b*np.exp(x * c)

def res(params,data):
    x,y = data
    return (func(params,x) - y)
a = 1
b = 0.5
c = 0.6
x = np.linspace(0,10,100)
x = x[::10]
x0 = np.array([1,0])
y = func(x,params = (a,b,c)) + (1+np.random.randn(len(x)))






# fit = least_squares(res,x0, args = (x,y,model_type), method = "lm")

# print(fit.x)


fitobj = kmpfit.Fitter(residuals =res, data = (x,y))




sys.exit()



fig = plt.figure(figsize=(4,4))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(2,3,4) # plot the point (2,3,4) on the figure

plt.show()




# cropping_limits = [225,2100,350,1400]

cropping_limits = [225,2000,250,1950]

#mean_dose_map = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_film_dose_map_reg.npy")
#plt.imshow(mean_dose_map)
#plt.close()


mean_dose_map = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Measurements\\EBT3_Holes_131021_Xray220kV_5Gy1_001.tif",-1)
mean_dose_map = mean_dose_map[10:722,10:497]
mean_dose_map  = 0.299*mean_dose_map[:,:,0] + 0.587*mean_dose_map[:,:,1] + 0.114*mean_dose_map[:,:,2]
mean_dose_map = tf.rescale(mean_dose_map, 4)
shape_diff = (2999 - mean_dose_map.shape[0], 2173 - mean_dose_map.shape[1])
mean_dose_map = pad(mean_dose_map, shape_diff)
mean_dose_map = mean_dose_map[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]


# image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes\\EBT3_Stripes_310821_Xray220kV_5Gy1_001.tif", -1)
# #2999, 2173
kernel_size = round(3.9*47)
# mean_dose_map  = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
# mean_dose_map = tf.rescale(mean_dose_map, 4)
# shape_diff = (2999 - mean_dose_map.shape[0], 2173 - mean_dose_map.shape[1])
# mean_dose_map = pad(mean_dose_map, shape_diff)
# mean_dose_map = mean_dose_map[250:2050, 350:1940]


sum_pooling = nn.LPPool2d(1, kernel_size = kernel_size, stride = kernel_size)
pooled_dose_map = sum_pooling(torch.tensor(mean_dose_map).unsqueeze(0))[0]

print(pooled_dose_map.shape)
x = np.arange(0,mean_dose_map.shape[0],1)  #height in space
y = np.arange(0,mean_dose_map.shape[1],1)  #width in space


fig,ax = plt.subplots()
ax.set_yticks(x[kernel_size::kernel_size])
# ax.set_yticklabels(["{:.3f}".format(y[i]/47) if i % 2 != 0 else "" for i in range(kernel_size,len(y),kernel_size)],fontsize = 6)
ax.set_yticklabels(["{:.1f}".format(x[i]) for i in range(kernel_size,len(x),kernel_size)],fontsize = 6)

ax.set_xticks(y[kernel_size::kernel_size])
ax.set_xticklabels(["{:.1f}".format(y[i]) for i in range(kernel_size,len(y),kernel_size)], fontsize = 6, rotation = 60)

ax.imshow(mean_dose_map)
ax.grid(True)



"""
Identifying peak and valley intensities
"""
# valley_idx = np.argwhere(mean_dose_map > 2.6e4)
# peak_idx = np.argwhere(mean_dose_map < 2.6e4)



# d95 = np.amin(mean_dose_map)/0.95
# d95_idx = np.abs(mean_dose_map-d95).argmin()
"""
Identifying dose that is 80 percent of maximum
"""
d80 = np.min(mean_dose_map)/0.8  #dividing by 0.8 because OD is opposite to dose
d80_idx  = np.abs(mean_dose_map-d80).argmin()

d80 = mean_dose_map[d80_idx//mean_dose_map.shape[1], d80_idx%mean_dose_map.shape[1]]
print(d80)
#d95 = mean_dose_map[d95_idx//mean_dose_map.shape[1], d95_idx%mean_dose_map.shape[1]]

"""
Using contour lines to identify beginning and end of peak
"""
print(len(x), len(y))

isodose = ax.contour(y, x, mean_dose_map, levels = [d80], colors  = "yellow") #y represents



"""
Getting index values from contour lines to find their position
"""
lines = []
for line in isodose.collections[0].get_paths():
    if line.vertices.shape[0] > 100: #less than hundred points is not a dose peak edge
        lines.append(line.vertices) #vertices is (column,row)

nearest_peak = np.zeros(np.array(pooled_dose_map.shape))

"""
We jump from quadrat centre to quadrat centre to find the smallest distance to a peak
"""
print(mean_dose_map.shape)
odd_i = 1
for i in range(kernel_size//2, mean_dose_map.shape[0]- kernel_size, kernel_size):
    print(i - kernel_size//2*odd_i)
    #print(i - kernel_size//2*odd - rest)
    odd_j = 1
    for j in range(kernel_size//2,mean_dose_map.shape[1], kernel_size):  # - kernel size to get right amount of
        #print(j - kernel_size//2*odd_j - rest_j)
        min_d = 1e6                                 #not possible distance
        centre = [i + kernel_size/2-kernel_size//2, j + kernel_size/2-kernel_size//2]
        for line in lines:
            x = line[:,1] #as vertices is (column, row) we need to get index 1
            y = line[:,0]
            d = np.sqrt((x -centre[0])**2 + (y-centre[1])**2)
            tmp = np.min(d)
            #print(tmp)
            if tmp < min_d:
                min_d = tmp
        #scatter for some reason wants the x and y axis values. Not x as rows
        plt.scatter(j + kernel_size/2 - kernel_size//2, i + kernel_size/2-kernel_size//2 )
        if mean_dose_map[i,j] < d80:
            nearest_peak[i - kernel_size//2*odd_i,j - kernel_size//2*odd_j] = 0
        else:
            nearest_peak[i - kernel_size//2*odd_i,j - kernel_size//2*odd_j] = min_d

        odd_j += 2

    odd_i += 2

        # dx = np.subtract(lines, )

print(nearest_peak)
plt.show()




def plot(x,y):
    fig,ax = plt.subplots()
    ax.plot(x,y)
    return fig, ax

x = np.linspace(0,10)
fig, ax = plot(x,2*x)
fig.set_size_inches(18.5, 10.5)
ax.plot(x,x**2)
#fig.savefig("C:\\Users\\jacob\\OneDrive\\Desktop\\figure")
plt.show()

tmp_seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv"))
tmp_seg_mask = tmp_seg_mask[200:2250,300:1750]

y = np.arange(0,2961,1)  #height in space
x = np.arange(0,2162,1) #width in space

fig, ax = plt.subplots()

ax.imshow(tmp_seg_mask)
ax.set_title("I spy with my little eye")

ax.set_xticklabels([y[i] for i in range(0,len(y),47)],fontsize = 7)
ax.set_yticklabels([x[i] for i in range(0,len(x),47)],fontsize = 7)
#y = np.arange(0,2961+47,1)  #height in space
#x = np.arange(0,2162+47,1) #width in space
#plt.yticks([y[i] for i in range(0,len(y),47)], fontsize=7)
#plt.xticks([x[i] for i in range(0,len(x),47)], fontsize=7)
plt.grid(True)
plt.show()



x = [1,1,1,1,1,1,2,2,2,2,2,2]
y = [1 + 0.5*np.random.randn(1),1 + 0.5*np.random.randn(1),1 + 0.5*np.random.randn(1),1 + 0.5*np.random.randn(1),1 + 0.5*np.random.randn(1),1 + 0.5*np.random.randn(1),
    2 + 0.25*np.random.randn(1),2 + 0.25*np.random.randn(1),2 + 0.25*np.random.randn(1),2 + 0.25*np.random.randn(1),2 + 0.25*np.random.randn(1),2 + 0.25*np.random.randn(1)]

plt.plot(x,y,"*")
plt.vlines(x[0],0,np.max(y))
plt.grid()
plt.show()


z = {"x":x,"y":y}

df = pd.DataFrame(z)
print(df)

kmeans = KMeans(n_clusters = 2).fit(df)

print(kmeans.cluster_centers_.shape)
print(kmeans.labels_)

plt.scatter(x,y)
plt.plot(kmeans.cluster_centers_[0,0],kmeans.cluster_centers_[0,1],"*")
plt.plot(kmeans.cluster_centers_[1,0],kmeans.cluster_centers_[1,1],"*")
plt.show()

plt.scatter(x,y)
plt.scatter(x,z)
plt.scatter(x,a)
plt.scatter(x,b)
plt.show()



x = np.random.randn(500)

kde = stats.gaussian_kde(x)

xx = np.linspace(np.min(x),np.max(x),100)

#n,bins,patches = plt.hist(np.ravel(x),align = "mid")
n,edges,patches = plt.hist(x,edgecolor = "black",bins = 10, density = True)
plt.plot(xx,kde(xx))




print(edges)

plt.close()
plt.bar(edges[:-1],n,width = edges[1]-edges[0],ec = "black",align = "edge")
plt.close()

sns.histplot(x,stat = "density")
sns.kdeplot(x)
plt.show()



"""
count_mat = np.zeros((3,3))
stride = 3

shape_x = (count_mat.shape[0], count_mat.shape[1], 3, 3)
strides_x = (stride*x.strides[0], stride*x.strides[1], x.strides[0], x.strides[1])

x_w = as_strided(x, shape_x, strides_x)

# Return the result of pooling


print(x_w.sum(axis = (2,3)))
import pandas as pd

df = pd.read_csv("C:\\Users\\jacob\\Downloads\\owid-covid-data.csv")
print(df)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pystackreg import StackReg
import skimage.transform as tf
control_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Control"
control_files = np.asarray([file for file in os.listdir(control_folder)])

first = [i for i, s in enumerate(control_files) if "001" in s]

ROI = 4
ROI_pixel_size = int(4*11.8)
control = control_files[first]
print(control)
test = cv2.imread(control_folder + "\\" + control[0],-1).shape

print(test)

rows = test[0]
columns = test[1]

print(rows//2-ROI_pixel_size)

control_img = np.zeros((len(first),test[0],test[1],test[2]))

for i in range(control_img.shape[0]):
    control_img[i] = cv2.imread(control_folder + "\\" + control[i], -1)

print(control_img[0].shape)
"""GREY_chan_ctrl = 0.299*control_img[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
                                   columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size,2] + 0.587* control_img[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
                                                                      columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size,1] + 0.114* control_img[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
                                                                                                         columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size,0]"""
#not really GREY but RED

RED_chan_ctrl = control_img[:,:,:,2]
# plt.plot(np.mean(GREY_chan_ctrl[:],axis = 0))
# plt.show()

sr = StackReg(StackReg.RIGID_BODY)

RED_chan_reg = np.zeros((RED_chan_ctrl.shape))
RED_chan_reg[0] = RED_chan_ctrl[0]
for i in range(1,len(RED_chan_ctrl)):
    tmat = sr.register(RED_chan_ctrl[0],RED_chan_ctrl[i])
    RED_chan_reg[i] = tf.warp(RED_chan_ctrl[i],tmat, order = 3)
    plt.subplot(121)
    plt.imshow(RED_chan_ctrl[i])
    plt.subplot(122)
    plt.imshow(RED_chan_reg[i])
    plt.close()
    #tr.warp(GREY_chan_ctrl[i],tmat)

print("mean ROI ")

print(np.mean(RED_chan_reg[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
                                   columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size]))
sys.exit()
"""--------------------------"""

bckg_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Background"
bckg_files = np.asarray([file for file in os.listdir(bckg_folder)])


ROI = 2


test = cv2.imread(bckg_folder + "\\" + bckg_files[0],-1).shape


bckg_img = np.zeros((len(bckg_files),test[0],test[1],test[2]))



for i in range(bckg_img.shape[0]):
    bckg_img[i] = cv2.imread(bckg_folder + "\\" + bckg_files[i], -1)

GREY_chan_bckg = 0.299*bckg_img[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
            columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size,2] + 0.587*bckg_img[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
                        columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size,1] + 0.114* bckg_img[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
                                    columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size,0]
# GREY_chan_bckg = bckg_img[:,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, 2]


print(GREY_chan_bckg.shape)
plt.imshow(GREY_chan_bckg[0])
plt.show()
GREY_chan_bckg_mean = np.mean(GREY_chan_bckg)


netOD = np.zeros((len(GREY_chan_ctrl)))
for i in range(len(GREY_chan_ctrl)):
    mean = np.mean(GREY_chan_ctrl[i])
    netOD[i] =np.log10((mean-GREY_chan_bckg_mean)/(mean-GREY_chan_bckg_mean))

print(np.mean(GREY_chan_ctrl))

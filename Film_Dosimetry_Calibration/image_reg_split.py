from pystackreg import StackReg
import numpy as np
import os
import cv2
import sys
import matplotlib.pyplot as plt
from skimage import transform as tf


def image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan, calibration_mode):
    #maxCorners = 4
    #Cropping away edges for easier corner detection
    #GREY_chan = cv2.GaussianBlur(GREY_chan, (2, 2),cv2.BORDER_DEFAULT)
    rows = BLUE_chan.shape[1]
    columns = BLUE_chan.shape[2]
    if calibration_mode:
        BLUE_chan_cropped = BLUE_chan[:,10:rows-10,10:columns-10]
        GREEN_chan_cropped = GREEN_chan[:,10:rows-10,10:columns-10]
        RED_chan_cropped = RED_chan[:,10:rows-10,10:columns-10]
        GREY_chan_cropped = GREY_chan[:,10:rows-10,10:columns-10]
        """BLUE_chan_cropped = BLUE_chan[:,10:722,10:493]
        GREEN_chan_cropped = GREEN_chan[:,10:722,10:493]
        RED_chan_cropped = RED_chan[:,10:722,10:493]
        GREY_chan_cropped = GREY_chan[:,10:722,10:493]"""
    else:
        BLUE_chan_cropped = BLUE_chan[:,10:rows-10,10:columns-10]
        GREEN_chan_cropped = GREEN_chan[:,10:rows-10,10:columns-10]
        RED_chan_cropped = RED_chan[:,10:rows-10,10:columns-10]
        GREY_chan_cropped = GREY_chan[:,10:rows-10,10:columns-10]
    #padded_GREY = np.pad(ref_img,pad_width = 10, mode = "constant",constant_values = np.max(GREY_chan[7]))
    sr = StackReg(StackReg.RIGID_BODY)
    ref_img = GREY_chan_cropped[0]
    """tmats = np.zeros((16,3,3))
    padded_GREY = np.pad(ref_img,pad_width = 10, mode = "constant",constant_values = np.max(GREY_chan[7]))
    ref_corners = cv2.goodFeaturesToTrack(padded_GREY, maxCorners, qualityLevel = 0.01,
                                      minDistance = 250).reshape((maxCorners,2)) #min euclidean distance
    """
    """plt.imshow(padded_GREY)
    plt.scatter(ref_corners[:,:,0],ref_corners[:,:,1])
    plt.show()"""
    for i in range(1,len(GREY_chan)):
        tmat = sr.register(ref_img,GREY_chan_cropped[i])
        BLUE_chan_cropped[i] = tf.warp(BLUE_chan_cropped[i],tmat,order = 1)
        GREEN_chan_cropped[i] = tf.warp(GREEN_chan_cropped[i],tmat,order = 1)
        RED_chan_cropped[i] = tf.warp(RED_chan_cropped[i],tmat, order = 1)
        GREY_chan_cropped[i] = tf.warp(GREY_chan_cropped[i],tmat,order = 1)
        print("{}/{} images registered".format(i,len(GREY_chan)-1))
        """
        fig = plt.figure()
        st = fig.suptitle("image {}".format(i), fontsize="x-large")

        plt.imshow(GREY_chan[7])
        plt.show(block=False)
        plt.pause(0.0001)
    plt.show()"""

    return BLUE_chan_cropped, GREEN_chan_cropped, RED_chan_cropped, GREY_chan_cropped

if __name__ == "__main__":

    """folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"

    files = np.asarray([file for file in os.listdir(folder)])
    first = [i for i, s in enumerate(files) if "001" in s]
    first_img = files[first]

    test_img = cv2.imread(os.path.join(folder,"EBT3_Calib_310821_Xray220kV_00Gy1_001.tif"), -1)

    test_img_shape = (len(first_img),test_img.shape[0], test_img.shape[1], test_img.shape[2])

    print(test_img_shape)

    images = np.zeros((test_img_shape))

    for i,filename in enumerate(first_img):
        images[i] = cv2.imread(os.path.join(folder,filename), -1)


    BLUE_chan = images[:,:,:,0]
    GREEN_chan = images[:,:,:,0]
    RED_chan = images[:,:,:,0]
    GREY_chan = np.mean(images,axis = 3)



    BLUE_chan, GREEN_chan, RED_chan, GREY_chan = image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan)


    for i in range(len(BLUE_chan)):
        fig = plt.figure()
        st = fig.suptitle("image {}".format(i), fontsize="x-large")
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title("BLUE")
        ax1.imshow(BLUE_chan[i])

        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title("GREEN")
        ax2.imshow(GREEN_chan[i])

        ax3 = fig.add_subplot(2,2,3)
        ax3.set_title("RED")
        ax3.imshow(RED_chan[i])

        ax4 = fig.add_subplot(2,2,4)
        ax4.set_title("GREY")
        ax4.imshow(GREY_chan[i])

        plt.show()"""



    folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes"

    files = np.asarray([file for file in os.listdir(folder)])
    first = [i for i, s in enumerate(files) if "001" in s and "00Gy" not in s]
    first_img = sorted(files[first],key = len)

    test_img = cv2.imread(os.path.join(folder,"EBT3_Stripes_310821_Xray220kV_5Gy3_003.tif"), -1)

    test_img_shape = (len(first_img),test_img.shape[0], test_img.shape[1], test_img.shape[2])

    print(test_img_shape)

    images = np.zeros((test_img_shape))

    for i,filename in enumerate(first_img):
        images[i] = cv2.imread(os.path.join(folder,filename), -1)

    """from PIL import Image
    from PIL.TiffTags import TAGS

    with Image.open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes\\EBT3_Stripes_310821_Xray220kV_5Gy3_003.tif") as img:
        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.iterkeys()}"""

    BLUE_chan = images[:,:,:,0]
    GREEN_chan = images[:,:,:,1]
    RED_chan = images[:,:,:,2]
    GREY_chan  = np.float32(0.2989 * RED_chan + 0.5870 * GREEN_chan + 0.1140 * BLUE_chan) #cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)#np.mean(images,axis = 3)


    BLUE_chan, GREEN_chan, RED_chan, GREY_chan = image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan, calibration_mode = False)


    maxCorners = 4

    ref_img = GREY_chan[7]
    padded_GREY = np.pad(ref_img,pad_width = 10, mode = "constant",constant_values = np.max(ref_img))
    dst = np.argwhere((padded_GREY < 3e4))
    dst = dst[::1000]
    print(dst.shape)
    #ref_corners = cv2.goodFeaturesToTrack(padded_GREY, maxCorners, qualityLevel = 0.001,
                                      #minDistance = 400).reshape((maxCorners,2))

    #plt.subplot(121)
    #plt.imshow(padded_GREY)
    #plt.scatter(ref_corners[:,0],ref_corners[:,1])


    padded_tmp = np.pad(GREY_chan[0],pad_width = 10, mode = "constant",constant_values = np.max(ref_img))




    #corners = cv2.goodFeaturesToTrack(padded_tmp, maxCorners, qualityLevel = 0.001,
    #                                  minDistance = 400).reshape((maxCorners,2))
    #plt.subplot(122)
    #plt.imshow(padded_tmp)
    #plt.scatter(corners[:,0],corners[:,1])
    #plt.show()
    #tmats = tf.estimate_transform("euclidean", src, dst)
    #registered_image = tf.warp(GREY_chan[2],inverse_map = tmats.inverse,order = 3)



    plt.subplot(131)
    plt.imshow(GREY_chan[7])
    plt.subplot(132)
    plt.imshow(GREY_chan[0])
    plt.subplot(133)
    plt.imshow(registered_image)
    plt.show()


    GREY_chan_reg = np.zeros((GREY_chan.shape))

    """BLUE_chan, GREEN_chan, RED_chan, GREY_chan_reg = image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan, calibration_mode = False)


    for i in range(len(BLUE_chan)):
        fig = plt.figure()
        st = fig.suptitle(first_img[i], fontsize="x-large")
        plt.imshow(GREY_chan_reg[i])
        plt.show(block=False)
        plt.pause(0.0001)

    plt.show()"""

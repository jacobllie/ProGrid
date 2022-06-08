from pystackreg import StackReg
import numpy as np
import os
import cv2
import sys
import matplotlib.pyplot as plt

def phase_correlation(dose_map):
    #cv2.blur(dose_map, (1,1))
    ref_image = dose_map[0]
    transformed_image = np.zeros(dose_map.shape)
    img1_fs = np.fft.fft2(ref_image)
    for i in range(1,len(dose_map)):
        img2_fs = np.fft.fft2(dose_map[i])
        cross_power_spectrum = np.multiply(img1_fs,img2_fs.conj()) / np.abs(np.multiply(img1_fs, img2_fs.conj()))

        r = np.abs(np.fft.ifft2(cross_power_spectrum))
        r = np.fft.fftshift(r)





        peak = np.argwhere(r == np.max(r))
        row_shift = round(dose_map[i].shape[0]/2) - peak[0,0]
        column_shift = round(dose_map[i].shape[1]/2) - peak[0,1]
        print(row_shift, column_shift)

        translated = np.roll(dose_map[i], -row_shift, axis = 0)
        transformed_image[i] = np.roll(translated, -column_shift, axis = 1)

        fig = plt.figure()
        st = fig.suptitle("image {}".format(i), fontsize="x-large")
        plt.subplot(221)
        plt.imshow(dose_map[0])
        plt.subplot(222)
        plt.imshow(r,cmap = "magma")
        plt.subplot(223)
        plt.imshow(dose_map[i])
        plt.subplot(224)
        plt.imshow(transformed_image[i])
        plt.show(block = False)
        plt.pause(0.0001)
    plt.show()
def image_reg(ref_img, dose_map):
    sr = StackReg(StackReg.RIGID_BODY)
    sr.register(ref_img, dose_map)
    out_img = sr.transform(dose_map)

    return out_img

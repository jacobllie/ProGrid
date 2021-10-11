from pystackreg import StackReg
import numpy as np
import os
import cv2
import sys
import matplotlib.pyplot as plt
import tifffile

def image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan, num_images):
    print(np.max(BLUE_chan), np.max(GREEN_chan), np.max(RED_chan), np.max(GREY_chan))

    sr = StackReg(StackReg.RIGID_BODY)
    ref_img = GREY_chan[0]

    for i in range(1,len(BLUE_chan)):
        sr.register(GREY_chan[i-1], GREY_chan[i])
        print("{}/{}".format(i,len(GREY_chan)))
        #sys.stdout.write("{:d}% images registered   \r".format(int((i+1)*100/len(BLUE_chan))))
        #sys.stdout.flush()
    for i in range(len(BLUE_chan)):
        BLUE_chan[i] = sr.transform(BLUE_chan[i])
        GREEN_chan[i] = sr.transform(GREEN_chan[i])
        RED_chan[i] = sr.transform(RED_chan[i])
        GREY_chan[i] = sr.transform(GREY_chan[i])

    return BLUE_chan, GREEN_chan, RED_chan, GREY_chan

if __name__ == "__main__":

    folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"

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


    """
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

    BLUE_chan, GREEN_chan, RED_chan, GREY_chan = image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan,1)


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

        plt.show()

"""
Skal prøve å bruke koden på en enkel form.
"""

"""theta = 30*np.pi/180
rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

square = np.zeros((200,200))

square[200//2-75:200//2+75,200//2-50:200//2+50] = 1

plt.subplot(131)
plt.imshow(square)




for j in range(50,150):
    for i in range(50,150):
        square[i,j] = rotation.dot(square[i,j])
        #square[i , 200//2-j:200//2+j] = rotation.dot(square[i, 200//2-j:200//2+j].T)


def rotate90Clockwise(A):
    N = len(A[0])
    for i in range(N // 2):
        for j in range(i, N - i - 1):
            temp = A[i][j]
            A[i][j] = A[N - 1 - j][i]
            A[N - 1 - j][i] = A[N - 1 - i][N - 1 - j]
            A[N - 1 - i][N - 1 - j] = A[j][N - 1 - i]
            A[j][N - 1 - i] = temp

    return A
rotated_square = rotate90Clockwise(square)

plt.subplot(132)
plt.imshow(rotated_square)
rotated_square[200//2 - 50: 200//2 + 50, 25:50] = 0
rotated_square[200//2 - 50: 200//2 + 50, 175:200] = 1
plt.subplot(133)
plt.imshow(rotated_square)
plt.show()



sr = StackReg(StackReg.RIGID_BODY)

sr.register(square,rotated_square)

out = sr.transform(rotated_square)

plt.subplot(121)
plt.imshow(rotated_square)
plt.subplot(122)
plt.imshow(out)
plt.show()"""

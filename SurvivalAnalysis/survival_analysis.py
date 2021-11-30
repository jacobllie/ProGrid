import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import f, ttest_ind
from kernel_density_estimation import kde
import seaborn as sb
from LQ_model import logLQ, fit, logLQ
import cv2
import torch.nn as nn
import torch
import cv2
import skimage.transform as tf
from pystackreg import StackReg



class survival_analysis:
    """
    Folder is folder path for segmentation data.
    Time is date of experiment.
    Mode is either Open or control
    Dose is dose in Gy
    Stack is either True or False.
    If Stack is True, then all colony counts should be summed to get mean plating
    efficiency and Survival for all experiments.
    If Stack is False, then the individual counts from the experiments should
    be returned.
    """

    def __init__(self,folder, time, mode, position, kernel_size, dose_map_path, template_file, dose):
        self.folder = folder
        self.time = time
        self.mode = mode
        self.position = position
        self.kernel_size = kernel_size
        self.dose_map_path = dose_map_path
        self.template_file = template_file
        self.dose = dose
    def data_acquisition(self):
        """
        Finds the average cell count from an experiment performed on a specific date.
        Then adds it to the count array.

        NB! Do not have excel sheet open, when running program. It will
        lead to error message: PermissionError: [Errno 13] Permission denied
        """

        self.flask_template = np.asarray(pd.read_csv(self.template_file))

        self.flask_template_shape = self.flask_template.shape

        self.ColonyData = {}

        if self.mode == "Open":
            count_data = np.zeros((len(self.time),len(self.dose),4))
        else:
            count_data = np.zeros((len(self.time),4))


        #ColonyData[1] = {}
        index = 0
        for i, date in enumerate(self.time):
            self.ColonyData[date] = {}
            if self.mode == "Open":
                for j,dose in enumerate(self.dose):
                    files = np.asarray([file for file in os.listdir(os.path.join(folder,date,self.mode))\
                                        if dose in file  and "ColonyData" in file])
                    for k, file in enumerate(files):
                        print(index,file)
                        index+=1
                        self.ColonyData[date] = pd.read_excel(os.path.join(self.folder, date, self.mode, file))
                        count_data[i,j,k] = self.ColonyData[date].shape[0]


            if self.mode == "Control":

                #print(os.listdir(os.path.join(self.folder,date,self.mode)))
                files = np.asarray([file for file in os.listdir(os.path.join(self.folder,date,self.mode)) if "ColonyData" in file])
                for j,file in enumerate(files):

                    #print(index,file)
                    #index += 1
                    self.ColonyData[date][self.position[j]] = pd.read_excel(os.path.join(self.folder, date, self.mode, file))
                    count_data[i,j] = self.ColonyData[date][self.position[j]].shape[0]


        return self.ColonyData, count_data


    def Colonymap(self):
        """
        Based on colony coordinates, and mask size, it creates a matrix with colony centroids
        as ones, and none colony centroids as 0.
        """


        self.colony_map = np.zeros((self.flask_template_shape[0], self.flask_template_shape[1]))
        print(self.colony_map.shape)
        """
        Matrices are indexed (row,column) = (x,y).
        But in an image we have (y,x) because y represents the height of the image,
        while x represents the width of the image
        """
        self.x_coor = np.array([int(i) for i in round(self.ColonyData["20112019"]["A"]["Centroid y-Coordinate (px)"])])
        self.y_coor = np.array([int(i) for i in round(self.ColonyData["20112019"]["A"]["Centroid x-Coordinate (px)"])])

        """if np.max(self.x_coor) > new_mask_shape[0]:
            self.y_coor = np.delete(self.y_coor, np.argwhere(self.x_coor > new_mask_shape[0]))
            self.x_coor = np.delete(self.x_coor, np.argwhere(self.x_coor > new_mask_shape[0]))
        if np.max(self.y_coor) > new_mask_shape[1]:
            self.x_coor = np.delete(self.x_coor, np.argwhere(self.y_coor > new_mask_shape[1]))
            self.y_coor = np.delete(self.y_coor, np.argwhere(self.y_coor > new_mask_shape[1]))"""


        for i, x in enumerate(self.x_coor):
            self.colony_map[x,self.y_coor[i]] = 1

        print(self.colony_map.shape)
        pass


    def registration(self):
        """
        We want to register the colony flasks to the dosemap made in the dosimetry
        script film_calibration.py. all the measurement films that recieved 5 Gy
        were registered to the first image EBT3_Open_310821_Xray220kV_5Gy1_001.
        therefore, if we register this image to the template mask. We can use the
        transformation matrix on the dose map. This way, we know exactly which dose
        each pixel in the cell flask received.
        """
        plt.subplot(121)
        plt.imshow(self.colony_map)

        self.colony_map = tf.rescale(self.colony_map.astype(float), 1/4,order = 3, preserve_range = True, clip = False)
        plt.subplot(122)
        plt.imshow(self.colony_map)
        plt.show()
        colony_map_shape = self.colony_map.shape
        flask_image = self.flask_template.astype(float)

        print("Colonymap shape after rescaling")
        print(colony_map_shape)


        """
        We crop the flask template to match the colony_map. Which has to be divisible
        by the kernel size
        """

        tmp_image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Open\\EBT3_Open_310821_Xray220kV_5Gy1_001.tif", -1)
        tmp_image = tmp_image[10:722,10:497]
        #we use the gray channel of the images
        film_image = 0.299*tmp_image[:,:,0] + 0.587*tmp_image[:,:,1] + 0.114*tmp_image[:,:,2]
        """
        Make sure mean dose map has the same dimention as moving image. Makes
        everything easier.
        """
        mean_dose_map = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open.npy")
        #moving_image = cv2.bitwise_not(moving_image)
        #moving_image = np.pad(moving_image, 5)

        #import sys
        #np.set_printoptions(threshold=sys.maxsize)

        film_image[film_image >= 3e4] = 0
        film_image[np.logical_and(0 < film_image, film_image < 3e4)] = 1#2**16

        """
        the dpi difference in the cell flask template and EBT3 film is 1200/300 = 4.
        But when downscaling the
        template image, we do not get a match in pixel height and width, because
        FOV is different when scanning EBT3 films compared to cell flasks.
        Therefore, we pad the EBT3 film image, to match the downscaled
        template image . The difference
        is always positive, because scanning the cell flask requires a larger
        FOV compared to EBT3 films.
        """
        flask_image = tf.rescale(flask_image.astype(float), 1/4,order = 3, preserve_range = True, clip = False)
        print("Flask template after rescaling")
        print(flask_image.shape)
        #ref_image = np.pad(ref_image,1)

        shape_diff = (flask_image.shape[0] - film_image.shape[0], flask_image.shape[1] - film_image.shape[1])

        if shape_diff[0] % 2 != 0:
            film_image_pad = np.pad(film_image, ((shape_diff[0]//2 + 1, shape_diff[0]//2),(shape_diff[1]//2,shape_diff[1]//2)))
            self.dose_map_pad = np.pad(mean_dose_map, ((shape_diff[0]//2 + 1, shape_diff[0]//2),(shape_diff[1]//2,shape_diff[1]//2)))
        elif shape_diff[1] % 2 != 0:
            film_image_pad = np.pad(film_image, ((shape_diff[0]//2, shape_diff[0]//2),(shape_diff[1]//2 + 1,shape_diff[1]//2)))
            self.dose_map_pad = np.pad(mean_dose_map, ((shape_diff[0]//2, shape_diff[0]//2),(shape_diff[1]//2 + 1,shape_diff[1]//2)))

        elif shape_diff[0] % 2 != 0 and shape_diff[1] % 2 != 0:
            film_image_pad = np.pad(film_image, ((shape_diff[0]//2 + 1, shape_diff[0]//2),(shape_diff[1]//2 + 1,shape_diff[1]//2)))
            self.dose_map_pad = np.pad(mean_dose_map, ((shape_diff[0]//2 + 1, shape_diff[0]//2),(shape_diff[1]//2 + 1,shape_diff[1]//2)))

        else:
            film_image_pad = np.pad(film_image, ((shape_diff[0]//2, shape_diff[0]//2),(shape_diff[1]//2,shape_diff[1]//2)))
            self.dose_map_pad = np.pad(mean_dose_map, ((shape_diff[0]//2, shape_diff[0]//2),(shape_diff[1]//2,shape_diff[1]//2)))

        #sr = StackReg(StackReg.RIGID_BODY)
        sr = StackReg(StackReg.SCALED_ROTATION)
        tmat = sr.register(film_image_pad, flask_image)
        self.flask_image_reg = tf.warp(flask_image,tmat,order = 1)
        #dose_map_reg = tf.warp(dose_map_pad, tmat, order = 1)

        """
        plt.subplot(131)
        plt.title("Mean dose map")
        plt.imshow(dose_map_pad, cmap = "gray")
        plt.subplot(132)
        plt.title("Rescaled flask template mask")
        plt.imshow(flask_image, cmap = "gray")
        plt.subplot(133)
        plt.title("Registered flask template")
        plt.imshow(flask_image_reg, cmap = "gray")
        plt.imshow(dose_map_pad, cmap = "magma", alpha = 0.6)
        plt.show()"""

        return tmat

    def Quadrat(self):
        """
        Performing sum pooling, where a square of size n x m traverse the image
        and sums up number of colonies within the square. The original mask has size
        2999 x 2173. We crop it down to a size corresponding to a multiple of
        the square size. E.g. with a square of 3 x 3 pixels, we need the square to
         count colonies in position: (0,0),(0,1),(0,2), Then it jumps to position:  (0,3),(0,4),(0,5),
                                     (1,0),(1,1),(1,2),                             (1,3),(1,4),(1,5),
                                     (2,0),(2,1),(2,2).                             (2,3),(2,4),(2,5).

        """

        new_mask_shape = (self.colony_map.shape[0]//self.kernel_size * self.kernel_size, self.colony_map.shape[1]//self.kernel_size * self.kernel_size)

        shape_diff = (self.colony_map.shape[0] - new_mask_shape[0], self.colony_map.shape[1] - new_mask_shape[1])


        if shape_diff[0] % 2 != 0:
            self.colony_map = self.colony_map[shape_diff[0]//2 + 1: self.colony_map.shape[0] - shape_diff[0]//2, shape_diff[1]//2: self.colony_map.shape[1] - shape_diff[1]//2]
        elif shape_diff[1] % 2 != 0:
            self.colony_map = self.colony_map[shape_diff[0]//2 : self.colony_map.shape[0] - shape_diff[0]//2, shape_diff[1]//2 + 1: self.colony_map.shape[1] - shape_diff[1]//2]

        elif shape_diff[0] % 2 != 0 and shape_diff[1] % 2 != 0:
            self.colony_map = self.colony_map[shape_diff[0]//2 + 1 : self.colony_map.shape[0] - shape_diff[0]//2, shape_diff[1]//2 + 1: self.colony_map.shape[1] - shape_diff[1]//2]

        else:
            self.colony_map = self.colony_map[shape_diff[0]//2: self.colony_map.shape[0] - shape_diff[0]//2, shape_diff[1]//2: self.colony_map.shape[1] - shape_diff[1]//2]



        pooling = nn.LPPool2d(1, kernel_size = self.kernel_size, stride = self.kernel_size) #p = 1 = norm_type

        #print(np.shape(torch.tensor(colony_map)))
        self.count_mat = pooling(torch.tensor(self.colony_map).unsqueeze(0))


        plt.imshow(self.count_mat[0])
        plt.show()

        return self.count_mat

    def logistic(self):
        pass







    #def Quadrat_anal(self, Colony_mask_shape, Colony_coor, kernel_size = 3):





if __name__ == "__main__":
    folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021"
    time = ["03012020", "17122020", "18112019", "20112019"]
    mode = ["Control", "Open"]
    dose = ["02", "05"]
    template_file_control =  "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Control\\A549-1811-K1-TemplateMask.csv"
    template_file_open = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\Open\\A549-1811-02-open-A-TemplateMask.csv"
    mean_dose_open = np.loadtxt("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_dose_open.npy")


    """
    18112019 and 20112019 data is much closer, compared with 1712202 and 03012020.
    We therefore combine these data to find alpha beta.
    """



    time = np.delete(time, [0,1])


    """
    Finding the number of counted colonies for control (0Gy) and open field
    experiments (2Gy and 5Gy)
    """
    survival_control = survival_analysis(folder, time, mode[0], template_file_control, dose = None)
    ColonyData_control, data_control = survival_control.data_acquisition()

    x_coor =[int(i) for i in round(ColonyData_control["20112019"]["Centroid x-Coordinate (px)"])]
    y_coor = [int(i) for i in round(ColonyData_control["20112019"]["Centroid y-Coordinate (px)"])]



    survival_open = survival_analysis(folder, time, mode[1], template_file_open, dose = dose)
    ColonyData_open, data_open  = survival_open.data_acquisition()

    #finding the normalization value
    max_count = np.max(data_control)

    data_open/= max_count
    data_control/= max_count  #normalizing counting to max number in ctrl

    dose_0 = [0 for i in range(len(np.ravel(data_control)))]

    """
    Identifying outliers in the data, and removing them
    """
    data_open_2GY = np.ravel(data_open[:,0,:])

    IQR_open = np.quantile(data_open_2GY,0.75) - np.quantile(data_open_2GY, 0.25)

    outlier_idx = np.argwhere(data_open_2GY < np.quantile(data_open_2GY, 0.25) - 1.5*IQR_open)


    no_outlier = np.delete(data_open_2GY,outlier_idx[:,0])



    #sb.boxplot(data_open[:,0,:])
    #sb.scatterplot(data_open[:,0,:])
    #plt.show()

    dose_2 = [2 for i in range(len(no_outlier))]
    dose_5 = [5 for i in range(len(np.ravel(data_open[:,1,:])))]

    """
    Combining all doses, to match the count-datapoints
    """
    doses = np.append(dose_0, np.append(dose_2,dose_5))

    """
    Combining all counting data for fit.
    """
    combined_count = np.log(np.append(np.ravel(data_control), np.append(no_outlier, np.ravel(data_open[:,1,:]))))


    fitting_param = fit(logLQ, doses, combined_count)

    plt.style.use("seaborn")
    plt.xlabel("dose [Gy]")
    plt.ylabel(r"$SF_{log}$")

    plt.plot(dose_2, np.log(no_outlier),"*", color = "magenta")
    plt.plot(dose_5, np.log(np.ravel(data_open[:,1,:])),"*", color = "magenta")
    # plt.show()

    plt.plot(dose_0, np.log(np.ravel(data_control)), "*", color = "magenta")
    #plt.plot(np.linspace(np.min(doses),10,100),logLQ(np.linspace(np.min(doses),10,100),
    #        fitting_param[0], fitting_param[1]), color = "salmon",
    #        label = r"fit: -($\alpha \cdot d + \beta \cdot d^2$)" + "\n" + r"$\alpha = {:.4}, \beta = {:.4}$".format(fitting_param[0], fitting_param[1]))

    plt.plot(np.linspace(np.min(doses),5,100),logLQ(np.linspace(np.min(doses),5,100),
            fitting_param[0], fitting_param[1]), color = "salmon",
            label = r"fit: -($\alpha \cdot d + \beta \cdot d^2$)" + "\n" + r"$\alpha = {:.4}, \beta = {:.4}$".format(fitting_param[0], fitting_param[1]))
    plt.legend()
    plt.show()

    #data_control = np.delete(data_control, [2,3], axis = 0)
    #t_test = ttest_ind(data_control[0], data_control[1])
    #print(t_test)






    """
    #Anova test on the datasets

    RSS1 = np.zeros(4)
    RSS2 = np.zeros(4)

    x_bar = np.mean(data_control,axis = 1)
    for i in range(1,len(data_control)):
        RSS1[i] = np.sum((x_bar[i]-data_control[i])**2)

    SSW = np.sum(RSS1)  #Sum of squares within

    SST = np.sum((np.mean(np.ravel(data_control))-np.ravel(data_control))**2) #total sum of squares

    for i in range(len(data_control)):
        RSS2[i] = np.sum((np.mean(np.ravel(data_control)) - x_bar[i])**2)

    SSB = np.sum(RSS2) * data_control.shape[1] #sum of squares between

    SSB_df = data_control.shape[0]-1 #3-1 = 2

    SSW_df = data_control.shape[0]*data_control.shape[1] - data_control.shape[0]  #3*4-3 = 9

    F = (SSB/SSB_df)/(SSW/SSW_df)

    p_value = 2*(1-f.cdf(F, SSB_df, SSW_df))

    print("F-test results:\n")
    print("F   p-value\n{:.2f}   {:.2e}".format(F,p_value))"""



    """
    #Identifying which of the data is low and high count

    #low_res_idx, high_res_idx = kde(np.ravel(data_control), 100,1)

    #print(low_res_idx[:,0])
    #print("----------------")
    #print(high_res_idx[:,0])



    low_res_idx, high_res_idx = kde(np.ravel(data_open[:,0,:]), 100,1)

    print(low_res_idx[:,0])
    print("**********")
    print(high_res_idx[:,0])

    low_res_idx, high_res_idx = kde(np.ravel(data_open[:,1,:]), 100,1)

    print(low_res_idx[:,0])
    print("**********")
    print(high_res_idx[:,0])
    """

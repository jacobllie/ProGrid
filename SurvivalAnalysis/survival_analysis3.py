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
from crop_pad import crop, pad
import sys



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

        plt.imshow(self.flask_template)
        plt.close()

        self.flask_template_shape = self.flask_template.shape

        self.ColonyData = {}


        count_data = np.zeros((len(self.time),len(self.dose),len(self.position)))


        #ColonyData[1] = {}
        index = 0
        for i, date in enumerate(self.time):
            self.ColonyData[date] = {}
            for j,dose in enumerate(self.dose):
                self.ColonyData[date][dose] = {}
                if self.mode == "Control":
                    files = np.asarray([file for file in os.listdir(os.path.join(self.folder,date,self.mode))\
                                        if "-K" in file  and "ColonyData" in file])
                else:
                    files = np.asarray([file for file in os.listdir(os.path.join(self.folder,date,self.mode))\
                                        if dose in file  and "ColonyData" in file])

                for k, file in enumerate(files):
                    print(index,file)

                    index+=1
                    self.ColonyData[date][dose][self.position[k]] = pd.read_excel(os.path.join(self.folder, date, self.mode, file))
                    count_data[i,j,k] = self.ColonyData[date][dose][self.position[k]].shape[0]

        print("-----------------")
        print("ColonyData gathered")
        return self.ColonyData, count_data


    def Colonymap(self):
        """
        Based on colony coordinates, and mask size, it creates a matrix with colony centroids
        as ones, and none colony centroids as 0.
        """

        #converting to float for rescaling
        self.flask_image = self.flask_template.astype(float)

        #Creating empty matrix to be filled with cell colony centroids
        self.colony_map = np.zeros((len(self.time), len(self.dose), len(self.position),  self.flask_image.shape[0], self.flask_image.shape[1]))


        for i, date in enumerate(self.time): #looping over experiments
            for j, dose in enumerate(self.dose): #looping over doses
                for k, pos in enumerate(self.position): #looping over pos A B C D
                    #getting x and y coordinate
                    self.x_coor = np.array([int(l) for l in round(self.ColonyData[date][dose][pos]["Centroid y-Coordinate (px)"])])
                    self.y_coor = np.array([int(l) for l in round(self.ColonyData[date][dose][pos]["Centroid x-Coordinate (px)"])])
                    #Setting pixels where we have colony = 1
                    for m, x in enumerate(self.x_coor):
                        self.colony_map[i,j,k,x,self.y_coor[m]] = 1
                    #plt.imshow(self.colony_map[i,j,k].astype(float))
                    #plt.show()

        """
        Matrices are indexed (row,column) = (x,y).
        But in an image we have (y,x) because y represents the height of the image,
        while x represents the width of the image
        """
        print("-----------------")
        print("Colonymap  of shape {} is created".format(self.colony_map.shape))
        pass


    def registration(self):
        """
        We want to register the EBT3 dose map to the cell flasks. All EBT3 images were
        registered to EBT3_Open_310821_Xray220kV_5Gy1_001.tif, so we only need
        to register this image.
        Because of the difference in resolution, we upscale the image from 300 to
        1200 dpi. And pad to adjust for difference in FOV when scanning the films.
        """

        if self.mode == "Open":
            tmp_image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Open\\EBT3_Open_310821_Xray220kV_5Gy1_001.tif", -1)

            #edge is removed, because it affects the registration.
            tmp_image = tmp_image[10:722,10:497]

            #we use the gray channel of the images
            film_image = 0.299*tmp_image[:,:,0] + 0.587*tmp_image[:,:,1] + 0.114*tmp_image[:,:,2]

            #Upscaling image to match cell flask of 1200 dpi
            film_image = tf.rescale(film_image, 4)

            #converting the EBT3 image to a mask.

            film_image[film_image >= 3e4] = 0
            film_image[np.logical_and(0 < film_image, film_image < 3e4)] = 1#2**16

            #finding the difference in shape
            shape_diff = (self.flask_image.shape[0] - film_image.shape[0], self.flask_image.shape[1] - film_image.shape[1])

            #pad to achieve equal shapes
            film_image_pad = pad(film_image, shape_diff)

            mean_dose_map = np.loadtxt(self.dose_map_path)

            dose_map_scaled = tf.rescale(mean_dose_map, 4)

            dose_map_pad = pad(dose_map_scaled, shape_diff)

            sr = StackReg(StackReg.RIGID_BODY)
            tmat = sr.register(self.flask_image, film_image_pad)
            self.film_image_reg = tf.warp(film_image_pad,tmat,order = 3)
            self.dose_map_reg = tf.warp(dose_map_pad, tmat, order = 3)

            """
            All EBT3 films have been registered to image 0. All cell flasks have
            been registered to flask 0. Therefore, we only need to register our image 0 to
            one of the flask templates, and use the transformation matrix on the other cell flasks.
            """

            print("-----------------")
            print("Registration complete")

            plt.subplot(131)
            plt.imshow(self.flask_image)
            plt.subplot(132)
            plt.imshow(film_image_pad, cmap = "magma")
            plt.imshow(self.flask_image, alpha = 0.9)
            plt.subplot(133)
            #plt.imshow(self.dose_map_reg, cmap = "magma")
            plt.imshow(self.film_image_reg, alpha = 0.8)
            plt.imshow(self.flask_image, alpha = 0.7)

            plt.close()

            return tmat

        elif self.mode == "GRID Stripes":
            seg_mask_5Gy = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv")).astype(float)
            tmp_image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes\\EBT3_Stripes_310821_Xray220kV_5Gy1_001.tif", -1)

            plt.imshow(seg_mask_5Gy[150:2300,300:1750])
            plt.close()

            #edge is removed, because it affects the registration.
            tmp_image = tmp_image[10:722,10:497]

            #we use the gray channel of the images
            film_image = 0.299*tmp_image[:,:,0] + 0.587*tmp_image[:,:,1] + 0.114*tmp_image[:,:,2]

            #Upscaling image to match cell flask of 1200 dpi
            film_image = tf.rescale(film_image, 4)

            #converting the EBT3 image to a mask.
            film_image[film_image >= 4e4] = 0
            film_image[np.logical_and(0 < film_image, film_image < 2.8e4)] = 0
            film_image[np.logical_and(2.8e4 < film_image, film_image < 4e4)] = 1

            #finding the difference in shape
            shape_diff = (self.flask_image.shape[0] - film_image.shape[0], self.flask_image.shape[1] - film_image.shape[1])

            #pad to achieve equal shapes
            film_image_pad = pad(film_image, shape_diff)

            """
            Now we load in mean dose map, which is a mean of 16 EBT3 films, which
            all have been registered to EBT3_Open_310821_Xray220kV_5Gy1_001.tif.
            So we can apply the registration matrix tmat, on the mean dose map.
            """

            mean_dose_map = np.loadtxt(self.dose_map_path)

            dose_map_scaled = tf.rescale(mean_dose_map, 4)

            dose_map_pad = pad(dose_map_scaled, shape_diff)

            sr = StackReg(StackReg.RIGID_BODY)
            tmat = sr.register(seg_mask_5Gy, film_image_pad.astype(float))
            self.film_image_reg = tf.warp(film_image_pad,tmat,order = 3)
            self.dose_map_reg = tf.warp(dose_map_pad,tmat,order = 3)

            """
            All EBT3 films have been registered to image 0. All cell flasks have
            been registered to flask 0. Therefore, we only need to register our image 0 to
            one of the flask templates, and use the transformation matrix on the other cell flasks.
            """

            print("-----------------")
            print("Registration complete")

            plt.subplot(131)
            plt.imshow(film_image_pad)
            plt.subplot(132)
            plt.imshow(film_image_pad)
            plt.imshow(seg_mask_5Gy, alpha = 0.9)
            plt.subplot(133)
            plt.imshow(self.film_image_reg)
            #plt.imshow(transformed_image)
            plt.imshow(seg_mask_5Gy, alpha = 0.9)
            plt.close()

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

        """
        We focus on a central area of the colony map, where the segmentation is
        most reliable
        """
        self.colony_map = self.colony_map[:,:,:,150:2300,300:1750]
        #the dose map need to match colony map
        self.dose_map = self.dose_map_reg[150:2300,300:1750]

        plt.imshow(self.dose_map)
        plt.close()
        new_mask_shape = (self.colony_map.shape[3]//self.kernel_size * self.kernel_size, self.colony_map.shape[4]//self.kernel_size * self.kernel_size)

        shape_diff = (self.colony_map.shape[3] - new_mask_shape[0], self.colony_map.shape[4] - new_mask_shape[1])


        assert shape_diff[0] >= 0 and shape_diff[1] >= 0, "Shape difference between scaled colony map and new_mask_shape is negative"

        self.colony_map = crop(self.colony_map, shape_diff)
        #The crop function needs shape (a,b,c,d,e)
        self.dose_map = crop(self.dose_map, shape_diff)[0,0,0]

        self.count_mat = np.zeros((len(self.time), len(self.dose), len(self.position),self.colony_map.shape[3]//self.kernel_size, self.colony_map.shape[4]//self.kernel_size))

        #p = 1  gives sum pooling
        sum_pooling = nn.LPPool2d(1, kernel_size = self.kernel_size, stride = self.kernel_size)
        #sum_pooling = nn.AvgPool2d(kernel_size = self.kernel_size, stride = self.kernel_size)
        for i, date in enumerate(self.time):
            for j,dose in enumerate(self.dose):
                for k, pos in enumerate(self.position):
                    self.count_mat[i,j,k] = (sum_pooling(torch.tensor(self.colony_map[i,j,k]).unsqueeze(0))[0])
                    #print(np.max(self.count_mat[i,j,k]))
                    #plt.title("{}, {}, {}".format(date,dose,pos))
                    #plt.imshow(self.count_mat[i,j,k])
                    #plt.show()

        print("----------------")
        print("Sum pooling complete")
        print(self.count_mat.shape)
        return self.count_mat

    def logistic(self, dose):
        """
        This function does average pooling using a 3x3 kernel with stride 3, to get
        the average dose within 3x3 kernels in the dose map. This way we know which dose the pooled survival
        counts (count mat) recieved, and we can plot them together.
        """
        #if a colony has been counted, set it to one.
        avg_pooling = nn.AvgPool2d(kernel_size = self.kernel_size, stride = self.kernel_size)
        pooled_dose = avg_pooling(torch.tensor(self.dose_map).unsqueeze(0))[0]

        if dose == 2:
            dose_map = np.ravel(pooled_dose)*2/5
            survival = np.reshape(self.count_mat[:,0,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3] * self.count_mat.shape[4]))
        elif dose == 5:
            dose_map = np.ravel(pooled_dose)
            survival = np.reshape(self.count_mat[:,1,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3] * self.count_mat.shape[4]))
        elif dose == 10:
            dose_map = np.ravel(pooled_dose)*2
            survival = np.reshape(self.count_mat[:,2,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3] * self.count_mat.shape[4]))

        print(pooled_dose)
        """
        Out EBT3 films recieved nominal 5 Gy dose, we scaled them down to 2Gy
        for survival analysis of cell flasks receiving 2 Gy.
        """

        print("dose shape")
        print(dose_map.shape)

        """
        count_mat[:,0] gives results from experiments performed on 1811 and 2011 in 2019.
        Which has shape (2,4,x,y) 4 positions, and image with shape (x,y). We want to find the mean over all
        positions and all experiments. Therefore we average over axis 0 and 1.
        """


        print("survival 2 Gy shape")
        print(survival.shape)


        """IQR_open = np.quantile(np.sort(dose_map),0.75) - np.quantile(np.sort(dose_map), 0.25)
        outlier_idx = np.argwhere(dose_map > np.quantile(dose_map, 0.25) + 1.5*IQR_open)
        no_outlier_dose = np.delete(dose_map,outlier_idx[:,0])

        no_outlier_survival = np.zeros((survival.shape[0], survival.shape[1], survival.shape[2] - len(outlier_idx)))
        for i in range(survival.shape[0]):
            for j in range(survival.shape[1]):
                no_outlier_survival[i,j] = np.delete(survival[i,j], outlier_idx[:,0])

        print("survival w.o. outliers in dose")
        print(no_outlier_survival.shape)
        print("dose w.o. outliers")
        print(no_outlier_dose.shape)

        sorted(list(zip(no_outlier_dose, no_outlier_survival)))"""

        idx = 0
        for i in range(survival.shape[0]):
            for j in range(survival.shape[1]):
                #plt.subplot(121)
                plt.title("{} {}".format(self.time[i], self.position[j]))
                plt.scatter(dose_map, np.ravel(survival[i,j]))
                #plt.subplot(122)
                #plt.bar(no_outlier_dose, no_outlier_survival[i,j], facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
                #plt.scatter(no_outlier_dose,no_outlier_survival[i,j])
                idx += 1
                plt.show()




        pass


    def SF(self, dose):





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

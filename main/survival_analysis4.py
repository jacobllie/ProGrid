import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import f, ttest_ind_from_stats
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
from plotting_functions_survival import survival_viz
import pickle
from utils import dose_profile2
"""
todo: make cropping edges a variable
"""


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

    def __init__(self,folder, time, mode, position, kernel_size, dose_map_path, template_file, save_path, dose, cropping_limits, data_extract = True):
        self.folder = folder
        self.time = time
        self.mode = mode
        self.position = position
        self.kernel_size = kernel_size
        self.dose_map_path = dose_map_path
        self.template_file = template_file
        self.save_path = save_path
        self.data_extract = data_extract
        self.dose = dose
        self.cropping_limits = cropping_limits

    def data_acquisition(self):
        """
        Finds the average cell count from an experiment performed on a specific date.
        Then adds it to the count array.

        NB! Do not have excel sheet open, when running program. It will
        lead to error message: PermissionError: [Errno 13] Permission denied
        """

        self.flask_template = np.asarray(pd.read_csv(self.template_file))

        self.flask_template_shape = self.flask_template.shape

        if self.data_extract:
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

            if self.mode == "Control":
                #"C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\Control_ColonyData_[210,2100,405,1900].pkl"
                a_file = open(self.save_path + "\\Control_ColonyData_{}.pkl".format(str(self.cropping_limits)), "wb")
                pickle.dump(self.ColonyData, a_file)
                a_file.close()
                np.save(self.save_path + "\\Control_countdata_{}.npy".format(str(self.cropping_limits)), count_data)

            elif self.mode == "Open":
                a_file = open(self.save_path + "\\Open_ColonyData_{}.pkl".format(str(self.cropping_limits)), "wb")
                pickle.dump(self.ColonyData, a_file)
                a_file.close()
                np.save(self.save_path + "\\Open_countdata_{}.npy".format(str(self.cropping_limits)), count_data)
            elif self.mode == "GRID Stripes":
                a_file = open(self.save_path + "\\GridStripes_ColonyData_{}.pkl".format(str(self.cropping_limits)), "wb")
                pickle.dump(self.ColonyData, a_file)
                a_file.close()
                np.save(self.save_path + "\\GridStripes_countdata_{}.npy".format(str(self.cropping_limits)), count_data)
            elif self.mode == "GRID Dots":
                a_file = open(self.save_path + "\\GridDots_ColonyData_{}.pkl".format(str(self.cropping_limits)), "wb")
                pickle.dump(self.ColonyData, a_file)
                a_file.close()
                np.save(self.save_path + "\\GridDots_countdata_{}.npy".format(str(self.cropping_limits)), count_data)

        if not self.data_extract:
            #extracting previously saved data
            if self.mode == "Control":
                #"C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\Control_ColonyData_[210,2100,405,1900].pkl"
                a_file = open(self.save_path + "\\Control_ColonyData_{}.pkl".format(str(self.cropping_limits)), "rb")
                self.ColonyData = pickle.load(a_file)
                a_file.close()
                count_data = np.load(self.save_path + "\\Control_countdata_{}.npy".format(str(self.cropping_limits)))

            elif self.mode == "Open":
                a_file = open(self.save_path + "\\Open_ColonyData_{}.pkl".format(str(self.cropping_limits)), "rb")
                self.ColonyData = pickle.load(a_file)
                a_file.close()
                count_data = np.load(self.save_path + "\\Open_countdata_{}.npy".format(self.cropping_limits))
            elif self.mode == "GRID Stripes":
                a_file = open(self.save_path + "\\GridStripes_ColonyData_{}.pkl".format(str(self.cropping_limits)), "rb")
                self.ColonyData = pickle.load(a_file)
                a_file.close()
                count_data = np.load(self.save_path + "\\GridStripes_countdata_{}.npy".format(str(self.cropping_limits)))
            elif self.mode == "GRID Dots":
                a_file = open(self.save_path + "\\GridDots_ColonyData_{}.pkl".format(str(self.cropping_limits)), "rb")
                self.ColonyData = pickle.load(a_file)
                a_file.close()
                count_data = np.load(self.save_path + "\\GridDots_countdata_{}.npy".format(str(self.cropping_limits)))


        return self.ColonyData, count_data


    def Colonymap(self):
        """
        Based on colony coordinates, and mask size, it creates a matrix with colony centroids
        as ones, and none colony centroids as 0.
        """

        #converting to float for rescaling
        self.flask_image = self.flask_template.astype(float)

        print(self.flask_image.shape)



        #Creating empty matrix to be filled with cell colony centroids
        self.colony_map = np.zeros((len(self.time), len(self.dose), len(self.position),  self.flask_image.shape[0], self.flask_image.shape[1]))

        print(self.colony_map.shape)

        for i, date in enumerate(self.time): #looping over experiments
            for j, dose in enumerate(self.dose): #looping over doses
                for k, pos in enumerate(self.position): #looping over pos A B C D
                    #getting x and y coordinate we must int because round returns float
                    self.x_coor = np.array([int(l) for l in round(self.ColonyData[date][dose][pos]["Centroid y-Coordinate (px)"])])
                    self.y_coor = np.array([int(l) for l in round(self.ColonyData[date][dose][pos]["Centroid x-Coordinate (px)"])])
                    #Setting pixels where we have colony = 1
                    for m, x in enumerate(self.x_coor):
                        self.colony_map[i,j,k,x,self.y_coor[m]] = 1


        """
        Matrices are indexed (row,column) = (x,y).
        But in an image we have (y,x) because y represents the height of the image,
        while x represents the width of the image
        """
        print(self.mode)
        print("-----------------")
        print("Colonymap  of shape {} is created".format(self.colony_map.shape))
        return self.colony_map


    def registration(self):
        """
        We want to register the EBT3 dose map to the cell flasks. All EBT3 images were
        registered to EBT3_Open_310821_Xray220kV_5Gy1_001.tif, so we only need
        to register this image.
        Because of the difference in resolution, we upscale the image from 300 to
        1200 dpi. And pad to adjust for difference in FOV when scanning the films.
        """

        if self.mode == "Open":

            #we only need to find tmat for first image of folder to cell flasks, because all other images are already registered to this
            tmp_image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Open\\EBT3_Open_310821_Xray220kV_5Gy1_001.tif", -1)

            #edge is removed, because it affects the registration.
            tmp_image = tmp_image[10:722,10:497]

            #we use the gray channel of the images
            film_image = 0.299*tmp_image[:,:,0] + 0.587*tmp_image[:,:,1] + 0.114*tmp_image[:,:,2]

            #Upscaling image to match cell flask of 1200 dpi


            film_image = tf.rescale(film_image, 4, order =  3)



            #print(image_height1[-1],image_height2[-1])


            #converting the EBT3 image to a mask.

            film_image[film_image >= 3e4] = 0
            film_image[np.logical_and(0 < film_image, film_image < 3e4)] = 1#2**16

            #finding the difference in shape
            shape_diff = (self.flask_image.shape[0] - film_image.shape[0], self.flask_image.shape[1] - film_image.shape[1])

            #pad to achieve equal shapes
            film_image_pad = pad(film_image, shape_diff)

            mean_dose_map = np.load(self.dose_map_path)

            # #checking consequences of upscaling
            # plt.subplot(121)
            # image_height1 =  np.arange(0,600-35,1)             #np.arange(0,len(mean_dose_map),1)
            # plt.plot(image_height1, dose_profile2(image_height1,mean_dose_map[35:600,260:290]))
            # tmp1 = np.abs(dose_profile2(image_height1,mean_dose_map[35:600,260:290]) - 5)/5                                      #np.sqrt((dose_profile2(image_height1,mean_dose_map[35:600,260:290]) - 5)**2)
            # MSE = np.mean(tmp1)
            #
            dose_map_scaled = tf.rescale(mean_dose_map, 4, order = 3)
            # plt.subplot(122)
            # image_height2 = np.arange(0,2400-140,1)    #np.arange(0,len(dose_map_scaled),1)
            # plt.plot(image_height2, dose_profile2(image_height2,dose_map_scaled[35*4:600*4,260*4:290*4]))
            #
            # tmp2 = np.abs(dose_profile2(image_height2,dose_map_scaled[35*4:600*4,260*4:290*4]) - 5)/5                                            #np.sqrt((dose_profile2(image_height2,dose_map_scaled[35*4:600*4,260*4:290*4]) - 5)**2)
            # MSE = np.mean(tmp2)
            # statistic, p_value = ttest_ind_from_stats(mean1=np.mean(tmp1), std1=np.std(tmp1), nobs1=len(tmp1), \
            #                    mean2=np.mean(tmp2), std2=np.std(tmp2), nobs2=len(tmp2), \
            #                    equal_var=False)
            # print("t-statistic      p-value")
            # print(statistic,p_value)
            # plt.show()

            dose_map_pad = pad(dose_map_scaled, shape_diff)




            sr = StackReg(StackReg.RIGID_BODY)
            tmat = sr.register(self.flask_image, film_image_pad)

            self.film_image_reg = tf.warp(film_image_pad,tmat,order = 3)
            self.dose_map_reg = tf.warp(dose_map_pad, tmat, order = 3)

            print(self.dose_map_reg.shape)

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
            plt.imshow(self.dose_map_reg, cmap = "magma")
            plt.imshow(self.film_image_reg, alpha = 0.8)
            plt.imshow(self.flask_image, alpha = 0.7)

            plt.close()

            """Mean dose map plot"""
            fig,ax = plt.subplots(figsize = (9,8))
            dose = ax.imshow(self.dose_map_reg, cmap = "viridis")
            plt.tight_layout()
            cbar = fig.colorbar(dose)
            cbar.ax.tick_params(labelsize=14)
            fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\mean_film_dose_map_{}.png".format(self.mode), dpi = 300)
            plt.show()
            #this is used for dose profiles and 1D survival analysis
            #np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_film_dose_map_reg_{}.npy".format(self.mode), self.dose_map_reg)
            #this is used for 2D analysis


            #np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_film_dose_map_reg_2D{}.npy".format(self.mode), self.dose_map_reg)

            return self.dose_map_reg

        elif "GRID" in self.mode.split(" "):
            if self.mode == "GRID Stripes":
                seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\18112019\\GRID Stripes\\A549-1811-05-gridS-A-SegMask.csv")).astype(float)
                tmp_image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Measurements\\Grid_Stripes\\EBT3_Stripes_310821_Xray220kV_5Gy1_001.tif", -1)
            elif self.mode == "GRID Dots":
                seg_mask = np.asarray(pd.read_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Segmentation Results - 15.11.2021\\20112019\\GRID Dots\\A549-2011-10-gridC-A-SegMask.csv")).astype(float)
                idx1 = seg_mask == 1
                idx0 = seg_mask == 0
                seg_mask[idx1] = 0
                seg_mask[idx0] = 1
                tmp_image = cv2.imread("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\131021\\Measurements\\EBT3_Holes_131021_Xray220kV_5Gy1_001.tif", -1)

            """
            All GRID Dots images are registered to the first image in the folder.
            A mean dose map were created by converting optical density of the images to dose.
            Then a mean dose map was created from all these images.
            So when registering the dose map to cell segmentation mask, we dont need to register the dose map directly.
            What we do is registering the first image in the GRID Dots folder to the segmentation
            mask, and then use the transformation matrix tmat on the mean dose map.
            """

            #edge is removed, because it affects the registration.
            tmp_image = tmp_image[10:722,10:497]

            #we use the gray channel of the images
            film_image = 0.299*tmp_image[:,:,0] + 0.587*tmp_image[:,:,1] + 0.114*tmp_image[:,:,2]

            #Upscaling image to match cell flask of 1200 dpi
            film_image = tf.rescale(film_image, 4)

            if self.mode == "GRID Stripes":

                #converting the EBT3 image to a mask.
                film_image[film_image >= 4e4] = 0
                film_image[np.logical_and(0 < film_image, film_image < 2.8e4)] = 0
                film_image[np.logical_and(2.8e4 < film_image, film_image < 4e4)] = 1


            elif self.mode == "GRID Dots":
                film_image[np.logical_and(0 < film_image, film_image < 2.5e4 )] = 1
                film_image[film_image >= 2.5e4]  = 0



            #finding the difference in shape
            shape_diff = (self.flask_image.shape[0] - film_image.shape[0], self.flask_image.shape[1] - film_image.shape[1])

            #pad to achieve equal shapes
            film_image_pad = pad(film_image, shape_diff)

            """
            Now we load in mean dose map, which is a mean of 16 EBT3 films, which
            all have been registered to EBT3_Open_310821_Xray220kV_5Gy1_001.tif.
            So we can apply the registration matrix tmat, on the mean dose map.
            """

            mean_dose_map = np.load(self.dose_map_path)

            # plt.imshow(mean_dose_map, cmap = "viridis")
            # plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\dose_map_cell_flask_reg.png", dpi = 1200)
            # plt.show()

            print(mean_dose_map.shape)


            dose_map_scaled = tf.rescale(mean_dose_map, 4)

            self.dose_map_pad = pad(dose_map_scaled, shape_diff)


            sr = StackReg(StackReg.RIGID_BODY)
            tmat = sr.register(seg_mask, film_image_pad.astype(float))
            if self.mode == "GRID Stripes":
                print(tmat)
                tmat[1,2]  -= 50
            elif self.mode == "GRID Dots":
                """
                Tuning not necessary for hole GRID
                """
                tmat[1,2] -= 30
                #tmat[0,2] -= 30
                print(tmat)
            self.film_image_reg = tf.warp(film_image_pad,tmat,order = 3)
            self.dose_map_reg = tf.warp(self.dose_map_pad,tmat,order = 3)



            # self.cropping_limits[0]:self.cropping_limits[1],self.cropping_limits[2]:self.cropping_limits[3]


            """plt.subplot(121)
            plt.title("Dosimetry Film")
            plt.imshow(film_image_pad, cmap = "hot")

            plt.subplot(122)
            plt.title("Segmentation mask")
            plt.imshow(seg_mask,cmap = "hot")
            #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\image_processing_registration.png",bbox_inches = "tight", pad_inches = 0.2,dpi = 1200)
            plt.show()"""

            fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12,12))
            ax = ax.flatten()
            ax[0].set_title("Binary dose film")
            ax[0].imshow(film_image_pad)
            ax[1].set_title("Dose map and segmentation mask\nbefore registration")
            ax[1].imshow(self.dose_map_pad)
            ax[1].imshow(seg_mask, alpha = 0.8, cmap = "viridis")
            ax[2].set_title("Dose map and segmentation mask\nafter registration")
            ax[2].imshow(self.dose_map_reg,cmap = "hot")
            #ax[2].imshow(self.film_image_reg)
            ax[2].imshow(seg_mask,alpha = 0.8, cmap = "viridis")
            # fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\131021\\registration_dotted_grid.png", dpi = 1200)
            # fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\registration_striped_grid.png", dpi = 1200)

            plt.close()

            """Mean dose map plot"""
            fig,ax = plt.subplots(figsize = (9,8))
            dose = ax.imshow(self.dose_map_reg, cmap = "viridis")
            plt.tight_layout()
            cbar = fig.colorbar(dose)
            cbar.ax.tick_params(labelsize=14)
            if self.mode == "GRID Stripes":
                fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\mean_film_dose_map_{}.png".format(self.mode), dpi = 300)
            else:
                fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\131021\\mean_film_dose_map_{}.png".format(self.mode), dpi = 300)
            plt.show()


            #Need this for further analysis
            #np.save("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\mean_film_dose_map\\mean_film_dose_map_reg_{}.npy".format(self.mode), self.dose_map_reg)


            """
            All EBT3 films have been registered to image 0. All cell flasks have
            been registered to flask 0. Therefore, we only need to register our image 0 to
            one of the flask templates, and use the transformation matrix on the other cell flasks.
            """

            print("-----------------")
            print("Registration complete")


            return self.dose_map_reg


    def Quadrat(self, survival_viz_path = None):
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
        #self.colony_map = self.colony_map[:,:,:,200:2250,300:1750] #shape (2150,1450)

        """
        Making sure cropping_limits is divisible by kernel size, to fit all quadrats inside image
        """

        """correct_shape = [(self.cropping_limits[1]-self.cropping_limits[0])//self.kernel_size * self.kernel_size, (self.cropping_limits[3]-self.cropping_limits[2])//self.kernel_size*self.kernel_size]
        shape_diff = ((self.cropping_limits[1]-self.cropping_limits[0]) - correct_shape[0], (self.cropping_limits[3]-self.cropping_limits[2])-correct_shape[1])
        print(shape_diff)
        print(correct_shape)
        self.cropping_limits"""


        self.colony_map = self.colony_map[:,:,:,self.cropping_limits[0]:self.cropping_limits[1],self.cropping_limits[2]:self.cropping_limits[3]] #shape (2150,1450)

        if self.mode == "GRID Stripes":
            """
            Visualisation
            """
            print(self.kernel_size)
            # survival_viz(self.colony_map, self.kernel_size, self.dose_map_reg, self.cropping_limits, Stripes = True)
            # survival_viz(self.colony_map, self.kernel_size, self.dose_map_reg, self.cropping_limits, Stripes = True)

            # plt.savefig(survival_viz_path,dpi = 300, pad_inches = 1)
            # plt.show()
            plt.title("registered dose map")
            plt.imshow(self.dose_map_reg[self.cropping_limits[0]:self.cropping_limits[1], self.cropping_limits[2]:self.cropping_limits[3]])
            plt.close()
        elif self.mode == "GRID Dots":
            # survival_viz(self.colony_map, self.kernel_size, self.dose_map_reg, self.cropping_limits, Stripes = False)
            # plt.savefig(survival_viz_path,dpi = 300, pad_inches = 1)
            plt.close()
            plt.imshow(self.dose_map_reg[self.cropping_limits[0]:self.cropping_limits[1], self.cropping_limits[2]:self.cropping_limits[3]])
            plt.close()
        #the dose map need to match colony map
        new_mask_shape = (self.colony_map.shape[3]//self.kernel_size * self.kernel_size, self.colony_map.shape[4]//self.kernel_size * self.kernel_size)
        shape_diff = (self.colony_map.shape[3] - new_mask_shape[0], self.colony_map.shape[4] - new_mask_shape[1])

        if not self.mode == "Control":
            self.dose_map = self.dose_map_reg[self.cropping_limits[0]:self.cropping_limits[1], self.cropping_limits[2]:self.cropping_limits[3]]
            # plt.imshow(self.dose_map)
            # plt.close()
            self.dose_map = crop(self.dose_map, shape_diff)[0,0,0]

        assert shape_diff[0] >= 0 and shape_diff[1] >= 0, "Shape difference between scaled colony map and new_mask_shape is negative"

        self.colony_map = crop(self.colony_map, shape_diff)
        print(self.colony_map.shape)
        #The crop function needs shape (a,b,c,d,e)


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

        # if self.mode == "GRID Stripes":
        #     tmp = np.zeros((self.flask_image.shape[0], self.flask_image.shape[1]))
        #
        #     tmp[::]
        #     #grid = self.count_mat
        #     #plt.imshow(self.count_mat)

        if not self.mode == "Control":
            return self.count_mat, self.dose_map
        else:
            return self.count_mat

    def logistic(self, dose):
        """
        This function does average pooling using a 3x3 kernel with stride 3, to get
        the average dose within 3x3 kernels in the dose map. This way we know which dose the pooled survival
        counts (count mat) recieved, and we can plot them together.
        """
        #if a colony has been counted, set it to one.
        avg_pooling = nn.AvgPool2d(kernel_size = self.kernel_size, stride = self.kernel_size)
        self.pooled_dose = avg_pooling(torch.tensor(self.dose_map).unsqueeze(0))[0]


        if dose == 2:
            dose_map = np.ravel(self.pooled_dose)*2/5
            survival = np.reshape(self.count_mat[:,0,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3] * self.count_mat.shape[4]))
        elif dose == 5:
            dose_map = np.ravel(self.pooled_dose)
            survival = np.reshape(self.count_mat[:,1,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3] * self.count_mat.shape[4]))
        elif dose == 10:
            dose_map = np.ravel(self.pooled_dose)*2
            survival = np.reshape(self.count_mat[:,2,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3] * self.count_mat.shape[4]))

        print(self.pooled_dose)
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


    def SC(self, dose):

        avg_pooling = nn.AvgPool2d(kernel_size = self.kernel_size, stride = self.kernel_size)
        #need this for nearest peak function
        self.pooled_dose = avg_pooling(torch.tensor(self.dose_map).unsqueeze(0))[0]

        if dose == 2:
            pooled_dose_2 = avg_pooling(torch.tensor(self.dose_map*2/5).unsqueeze(0))[0]
            dose_map = pooled_dose_2
            survival = np.reshape(self.count_mat[:,0,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3], self.count_mat.shape[4]))
        elif dose == 5:
            pooled_dose_5 = avg_pooling(torch.tensor(self.dose_map).unsqueeze(0))[0]
            dose_map = pooled_dose_5
            survival = np.reshape(self.count_mat[:,1,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3], self.count_mat.shape[4]))
        elif dose == 10:
            pooled_dose_10 = avg_pooling(torch.tensor(self.dose_map*10/5).unsqueeze(0))[0]
            dose_map = pooled_dose_10
            survival = np.reshape(self.count_mat[:,2,:], (self.count_mat.shape[0],
                                  self.count_mat.shape[2], self.count_mat.shape[3], self.count_mat.shape[4]))



        return np.asarray(dose_map), survival


    def nearest_peak(self, cropping_limits):
        """
        This function finds the minimum distance from a quadrat center to a peak in pixels.
        It needs an individual cropping limit because some peaks are outside the orignal cropping window.
        I.e., some quadrats are far away from a peak within our cropping window, but if you zoom out a little,
        you might find a peak that's very close to this quadrat.
        """
        #dose_map = self.dose_map_reg[self.cropping_limits[0]:self.cropping_limits[1], self.cropping_limits[2]:self.cropping_limits[3]]
        #dose_map = self.dose_map_pad[self.cropping_limits[0]:self.cropping_limits[1], self.cropping_limits[2]:self.cropping_limits[3]]

        #dose_map = self.dose_map_reg[200:2700, 500:1500]

        print(self.dose_map_reg.shape)


        dose_map = self.dose_map_reg[cropping_limits[0]:cropping_limits[1], cropping_limits[2]:cropping_limits[3]]
        # dose_map = self.dose_map_reg[10:700,10:500]


        #plt.imshow(self.dose_map_reg, alpha = 0.8)

        print(self.kernel_size)
        print(dose_map.shape)


        x = np.arange(0,dose_map.shape[0],1)  #height in space
        y = np.arange(0,dose_map.shape[1],1)  #width in space


        fig,ax = plt.subplots()
        ax.set_yticks(x[self.kernel_size::self.kernel_size])
        # ax.set_yticklabels(["{:.3f}".format(y[i]/47) if i % 2 != 0 else "" for i in range(kernel_size,len(y),kernel_size)],fontsize = 6)
        ax.set_yticklabels(["{:.1f}".format(x[i]) for i in range(self.kernel_size,len(x),self.kernel_size)],fontsize = 6)

        ax.set_xticks(y[self.kernel_size::self.kernel_size])
        ax.set_xticklabels(["{:.1f}".format(y[i]) for i in range(self.kernel_size,len(y),self.kernel_size)], fontsize = 6, rotation = 60)

        ax.imshow(dose_map)
        #ax.grid(True, color = "r")



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
        d80 = np.max(dose_map)*0.85  #dividing by 0.8 because OD is opposite to dose
        #d80_idx  = np.abs(dose_map-d80).argmin()

        #d80 = dose_map[d80_idx//dose_map.shape[1], d80_idx%dose_map.shape[1]]


        #d95 = mean_dose_map[d95_idx//mean_dose_map.shape[1], d95_idx%mean_dose_map.shape[1]]

        """
        Using contour lines to identify beginning and end of peak
        """
        print(len(x), len(y))

        isodose = ax.contour(y, x, dose_map, levels = [d80], colors  = "blue") #y represents

        """
        Getting index values from contour lines to find their position
        """
        lines = []
        for line in isodose.collections[0].get_paths():
            if line.vertices.shape[0] > 500: #less than hundred points is not a dose peak edge
                lines.append(line.vertices) #vertices is (column,row)

        dist = np.zeros(np.array(self.pooled_dose.shape))

        print(dist.shape)
        print(dose_map.shape)


        """
        We jump from quadrat centre to quadrat centre to find the smallest distance to a peak
        """
        odd_i = 1
        """
        We do not go the full dimensions of the dose map, because it will surpass
        the dimensions of our pooled dose map, which is based on different cropping limit.
        """

        """
        For a reason I haven't figured out, the i - kernel_size//2 * odd_i doesnt
        work for kernel_size%2 == 0, we therefore use
        i//kernel_size instead
        """

        rest_i = 0
        print(self.pooled_dose.shape[0]*self.kernel_size)
        for i in range(self.kernel_size//2, self.pooled_dose.shape[0]*self.kernel_size , self.kernel_size):
            if self.kernel_size%2 == 0:
                x = 1
                print((i//self.kernel_size))
            else:
                x = 1
                print((i - self.kernel_size//2*odd_i)/self.pooled_dose.shape[0])

            #print(i - kernel_size//2*odd - rest)
            odd_j = 1
            for j in range(self.kernel_size//2,self.pooled_dose.shape[1]*self.kernel_size, self.kernel_size):  # - kernel size to get right amount of
                min_d = 1e6                                 #not possible distance
                #centre of the kernel we are comparing with
                centre = [i + self.kernel_size/2-self.kernel_size//2, j + self.kernel_size/2-self.kernel_size//2] # [x-axis, y-axis]
                for line in lines:
                    x = line[:,1] #as vertices is (column, row) we need to get index 1
                    y = line[:,0]
                    d = np.sqrt((x -centre[0])**2 + (y-centre[1])**2)


                    #plt.scatter(y,x)
                    tmp = np.min(d)
                    #print(tmp)
                    if tmp < min_d:
                        min_d = tmp
                        idx_tmp = np.argwhere(d == min_d)
                        if len(idx_tmp) > 1:
                            #print(idx_tmp)
                            # np.ravel([np.round(x[np.argwhere(d == min_d)]),y[np.argwhere(d == min_d)]])
                            min_d_idx = np.ravel([np.round(x[idx_tmp[1]]),np.round(y[idx_tmp[1]])])
                            #print(idx_tmp[1],idx_tmp[0], x[idx_tmp[0]],x[idx_tmp[1]], y[idx_tmp[0]],y[idx_tmp[1]])
                        else:
                            min_d_idx = np.ravel([np.round(x[idx_tmp]),np.round(y[idx_tmp])])


                # plt.plot([centre[1], min_d_idx[1]],[centre[0],min_d_idx[0]])
                #scatter for some reason wants the x and y axis values. Not x as rows
                plt.scatter(j + self.kernel_size/2 - self.kernel_size//2, i + self.kernel_size/2-self.kernel_size//2 ,s = 5)

                #if the quadrat is located within a peak, then the distance is 0
                if dose_map[i,j] > d80: #assumes only 5 Gy irradiated films
                    if self.kernel_size%2 == 0:
                        dist[i//self.kernel_size, j//self.kernel_size] = 0
                    else:
                        dist[i - self.kernel_size//2*odd_i,j - self.kernel_size//2*odd_j] = 0
                else:
                    if self.kernel_size % 2 == 0:
                        dist[i//self.kernel_size, j//self.kernel_size] = min_d
                        plt.plot([centre[1], min_d_idx[1]],[centre[0],min_d_idx[0]])
                    else:
                        dist[i - self.kernel_size//2*odd_i,j - self.kernel_size//2*odd_j] = min_d
                        plt.plot([centre[1], min_d_idx[1]],[centre[0],min_d_idx[0]])

                odd_j += 2


            odd_i += 2

                # dx = np.subtract(lines, )

        plt.imshow(self.dose_map_reg[self.cropping_limits[0]:self.cropping_limits[1], self.cropping_limits[2]:self.cropping_limits[3]], alpha = 0.8, cmap = "hot")


        #plt.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\Thesis\\figures\\nearest_peak_{}mm.png".format(int(self.kernel_size/47)), bbox_inches = "tight", pad_inches = 0.1, dpi = 1200)
        plt.close()
        """if self.kernel_size == 188:
            print(dist)"""
        return dist


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

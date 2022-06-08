import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.optimize import least_squares
from scipy.stats import f_oneway
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
# from parallelized_kde import parallelized_kde
import random
import matplotlib.patches as patches
from image_reg_split import image_reg
from itertools import combinations


class film_calibration:
    def __init__(self, image_folder,background_folder, control_folder, test_image,\
                 background_image, control_image,  reg_path, measurement_crop = None,  calibration_mode = True,
                 image_registration = False, registration_save = False, grid = False, open = False):
        self.image_folder = image_folder
        self.background_folder = background_folder
        self.control_folder = control_folder
        self.test_image = test_image
        self.background_image = background_image
        self.control_image = control_image
        self.measurement_crop = measurement_crop
        self.calibration_mode = calibration_mode
        self.image_registration = image_registration
        self.registration_save = registration_save
        self.reg_path = reg_path
        self.grid = grid
        self.open = open

    """def sort_key(self, string):
      tmp = re.search(r'_(\d{1,2}|\d.\d)(?:Gy)\d_',string)
      if tmp:
          return float(tmp.group(1))
      else:
          return -1"""
    def image_acquisition(self):
        """
        This function extracts the images from the right folder, and returns the
        intensity values.
        """
        image_files = np.asarray([file for file in os.listdir(self.image_folder)])

        """
        Checking if scans of same film are statisticly significant
        We see that most scans are not signifi different, but we suspect that the large samplesize 94X94,
        decreases the p-value, and migth pick up on very small differences.
        """
        plt.style.use("seaborn")
        if self.calibration_mode:
            ROI_size = 4
            ROI_pixel_size = round(ROI_size*11.81)
            counter = 0
            test_images = []
            filenames = {}
            p_values = np.zeros(len(image_files)//4)
            maxdiff = 0
            #for plotting percentage difference for each film
            film_axis = np.arange(1,len(image_files)//4 + 1,1)
            print(film_axis)
            fig, ax = plt.subplots(figsize = (10,5))
            for i, filename in enumerate(image_files):
                #print(filename)
                tmp = cv2.imread(os.path.join(self.image_folder,filename), -1)
                gray_tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                rows = gray_tmp.shape[0]
                columns = gray_tmp.shape[1]
                gray_tmp = gray_tmp[rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
                            columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size]
                test_images.append(gray_tmp)
                counter += 1
                if counter % 4 == 0:
                    diff = np.zeros(6)
                    mean_intensity = np.zeros(6)
                    #finding difference between mean pixel value of all scans
                    for j,img in enumerate(list(combinations([0,1,2,3],2))):
                        diff[j] = np.abs(np.mean(test_images[img[0]]) - np.mean(test_images[img[1]]))
                        mean_intensity[j] = (np.mean(test_images[img[0]]) + np.mean(test_images[img[1]]))/2
                        if diff[j] > maxdiff:
                            maxdiff = diff[j]
                            maxdiff_img = filename
                            max_mean_intensity = (np.mean(test_images[img[0]]) + np.mean(test_images[img[1]]))/2
                    ax.plot(film_axis[i//4], np.mean(diff/mean_intensity)*100, "*")
                    test_images = []
                    #ANOVA
                    """f = f_oneway(np.ravel(test_images[0]),np.ravel(test_images[1]),np.ravel(test_images[2]),np.ravel(test_images[3]))
                    counter == 1
                    #print("film {}      p-value {:.5}".format(filename.split("_")[4], f[1]))

                    p_values[i//4] = f[1]
                    if p_values[i//4] < 0.5:
                        #finding maximum difference
                        for img in list(combinations([0,1,2,3],2)):
                            diff = np.abs(np.mean(test_images[img[0]]) - np.mean(test_images[img[1]]))
                            if diff > maxdiff:
                                maxdiff = diff
                                maxdiff_img = filename
                                mean_intensity = (np.mean(test_images[img[0]]) + np.mean(test_images[img[1]]))/2

                        filenames[filename] = (filename,i//4,f[1])
                        plt.suptitle(r"$\mu_1 = {:.5}$ $\mu_2 = {:.5}$ $\mu_3 = {:.5}$ $\mu_4 = {:.5}$".format(np.mean(test_images[0]), np.mean(test_images[1]),np.mean(test_images[2]),np.mean(test_images[3])))
                        plt.subplot(2,3,1)
                        plt.imshow(test_images[0])
                        plt.subplot(2,3,2)
                        plt.plot(np.ravel(test_images[0]),"*")
                        plt.subplot(2,3,3)
                        plt.plot(np.ravel(test_images[1]),"*")
                        plt.subplot(2,3,4)
                        plt.plot(np.ravel(test_images[2]),"*")
                        plt.subplot(2,3,5)
                        plt.plot(np.ravel(test_images[3]),"*")
                        plt.close()"""
            ax.set_title("Mean RPD between 4 scans for all images")
            ax.set_xlabel("# Film")
            ax.set_ylabel(r"$\frac{|I_x - I_y|}{(I_x + I_y)/2} \cdot 100\%$", fontsize = 14,labelpad = 40, rotation =0)
            ax.tick_params(axis='both', which='major', labelsize=7)
            fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\percentage_diff_scans.png", bbox_inches = "tight", pad_inches = 0.1, dpi = 300)
            plt.close()
                    # test_images = []
                    #print("---------------")

            print(" max percentage difference")
            print(maxdiff/max_mean_intensity)
            fig, ax = plt.subplots(figsize = (12,12))
            ax.plot(p_values, "*")
            ax.set_xlabel("a.u.")
            ax.set_ylabel("p-value")
            ax.hlines(np.mean(p_values),0,len(p_values))
            ax.hlines(0.5, 0, len(p_values))
            ax.annotate(r"$\alpha = 0.05$", (0.9,0.1), fontsize=15)
            ax.annotate(r"$\bar{x}_{p_{value}}$",(0.6,0.45), fontsize = 15)
            for i in filenames:
                ax.annotate(filenames[i][0].split("_")[4], (filenames[i][1],filenames[i][2]))
                # plt.annotate(filenames[i], )
            significant = len(p_values[p_values < 0.05])/len(p_values)
            non_significant = len(p_values[p_values > 0.05])/len(p_values)
            ax.set_title("{:.3}% significantly different {:.3}% not significantly different ".format(significant*100,non_significant*100))
            #fig.savefig("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\EBT3 dosimetry\\310821\\ANOVA_test_scans.png", bbox_inches = "tight", pad_inches = 0.2, dpi = 1200)
            plt.close()

            #print("p-value: {:.5}".format(f[1]))


        background_files = np.asarray([file for file in os.listdir(self.background_folder)])
        control_files = np.asarray([file for file in os.listdir(self.control_folder)])
        """finding the index where the filename contains 001. We wish to separate
           the first second third and fourth scan, because the film might be
           affected by the scanning.
        """

        first = [i for i, s in enumerate(image_files) if "001" in s]
        second = [i for i, s in enumerate(image_files) if "002" in s]
        third = [i for i, s in enumerate(image_files) if "003" in s]
        fourth = [i for i, s in enumerate(image_files) if "004" in s]

        background = [file for file in background_files]

        control = [i for i, s in enumerate(control_files) if "001" in s and "00Gy" in s]
        first_control = control_files[control]

        self.num_films = (len(first))
        self.num_background = len(background)
        self.num_control = len(control)
        global first_img, second_img, third_img, fourth_img

        if self.calibration_mode:
            first_img = image_files[first]
            second_img = image_files[second]
            third_img = image_files[third]
            fourth_img = image_files[fourth]

        else:
            first_img = sorted(image_files[first],key = len)
            second_img = sorted(image_files[second],key = len)
            third_img = sorted(image_files[second],key = len)
            fourth_img = sorted(image_files[second],key = len)


        """
        Important note:
        The filenames are sorted alphabetically. Therefore the doses are sorted
        as follows:
        [0.1, 0.2, 0.5, 10, 1, 2, 5].
        We therefore need to account for this, so that the image intensity values
        correspond to the right dose. I.E. if the film is very dark, it probably didn't get
        0.1 Gy.
        """
        self.sorted_files = np.append([first_img], [second_img, third_img, fourth_img])

        self.img_shape = (cv2.imread(os.path.join(self.image_folder,self.test_image))).shape



        #noise_img = np.zeros((4,self.img_shape[0],self.img_shape[1],self.img_shape[2]))


        """key_list = []
        for string in sorted_files:
            tmp = re.search(r'_(\d{1,2}|\d.\d)(?:Gy)\d_',string)
            if tmp != None:
                key_list.append(tmp.group(1))
            else:

        key_list.sort(key=float)
        print(len(key_list))"""


        """
        On the 24.8.21 we did calibration of dosimetry films. We did 0.1,0.2,0.5,1,2,5,10 Gy
        using 4 films for each dose. We then scanned the film 4 times in the epson scanner. The final three
        characters of the filename is 00x. Because the film might be affected by the scanning, we split them
        appart.
        The images matrix, will in this case, have shape (132,507,484,3). There are 132 images in total.
        The first 33 images will have the name 001, the next 33 images will have the name 002 and so on.
        The pixel size of the image i 507 x 484. And it has 3 channels: red, green, blue.
        """

        self.images = np.zeros((len(self.sorted_files),self.img_shape[0], self.img_shape[1],self.img_shape[2]))

        print(self.images.shape)


        if self.calibration_mode:
            self.background_img = np.zeros((self.num_background,self.img_shape[0], self.img_shape[1],self.img_shape[2]))
            self.control_img = np.zeros((self.num_control,self.img_shape[0], self.img_shape[1],self.img_shape[2]))

        else:
            """
            Midlertidig løsning. Kommer til å legge inn slik at control og background
            får sin egen mappe, som må sendes inn til film_calibration.
            """

            self.background_shape = (cv2.imread(os.path.join(self.background_folder,self.background_image))).shape
            self.control_shape = (cv2.imread(os.path.join(self.control_folder,self.control_image))).shape
            self.background_img = np.zeros((self.num_background, self.background_shape[0], self.background_shape[1], self.background_shape[2]))
            self.control_img = np.zeros((self.num_control, self.control_shape[0], self.control_shape[1], self.control_shape[2]))


        """
        the control images are only 507 x 484.
        """
        for i,filename in enumerate(self.sorted_files):
            self.images[i] = cv2.imread(os.path.join(self.image_folder,filename), -1)
        for i,filename in enumerate(background_files):
            self.background_img[i] = cv2.imread(os.path.join(self.background_folder,filename),-1)
        for i,filename in enumerate(first_control):

            rows = self.img_shape[0]
            columns = self.img_shape[1]
            ROI_pixel_size = int(4*11.8)
            self.control_img[i] = cv2.imread(os.path.join(self.control_folder,filename),-1)
            #print(np.mean(self.control_img[i,rows//2-ROI_pixel_size:rows//2+ROI_pixel_size, \
            #                                   columns//2 - ROI_pixel_size:columns//2 + ROI_pixel_size, 2]))

        print(len(first_control))
        #mean value of control is correct here!

        """
        Checking difference in response between scans; did not work
        """
        """test = self.images.reshape(4,7,8,self.img_shape[0],self.img_shape[1],self.img_shape[2])
        first_scan = test[0]
        second_scan = test[1]
        third_scan = test[2]
        fourth_scan = test[3]

        dose_axis = ["0.1","0.2","0.5","10","1","2","5"]
        mean_first = np.mean(first_scan, axis = (2,3,4))
        mean_second = np.mean(second_scan, axis = (2,3,4))
        mean_third = np.mean(third_scan, axis = (2,3,4))
        mean_fourth = np.mean(fourth_scan, axis = (2,3,4))
        for i in range(mean_first.shape[0]):  #looping over all doses and checking if the scans are significantly different
            for j in range(mean_first.shape[1]):
                f = f_oneway(mean_first[i,j], mean_second[i,j], mean_third[i,j], mean_fourth[i,j])
            print("dose: {}         p_value: {:.5}".format(dose_axis[i], f[1]))"""
    def color_channel_extraction(self, image, num_images, ROI_size, image_type, channel = "RED"):

        """
        This function simply extract a centrally placed square ROI from the images acquired
        in the function above. The specific color channels are then extracted, so
        the intensity values can be used to find net Optical Density (OD).
        """

        """When extracting images, we need to register them in relation to a chosen reference image. This is to mitigate
        scanning variations."""

        BLUE_chan = image[:num_images,:,:,0]
        GREEN_chan = image[:num_images,:,:,1]
        RED_chan = image[:num_images,:,:,2]

        GREY_chan = 0.299*RED_chan + 0.587*GREEN_chan + 0.114*BLUE_chan
        #GREY_chan = np.mean(image[:num_images], axis = 3)
        rows = self.img_shape[0]
        columns = self.img_shape[1]



        if self.calibration_mode:
            if self.image_registration and self.registration_save:
                BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan, self.calibration_mode, image_type)
                np.save(self.reg_path + "\\BLUE_calib_{}".format(image_type), BLUE_chan)
                np.save(self.reg_path + "\\GREEN_calib_{}".format(image_type), GREEN_chan)
                np.save(self.reg_path + "\\RED_calib_{}".format(image_type), RED_chan)
                np.save(self.reg_path + "\\GREY_calib_{}".format(image_type), GREY_chan)
            elif self.image_registration and not self.registration_save:
                print("Registered but not saved")
                #intensity value is correct before entering registration

                BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, GREEN_chan, RED_chan,GREY_chan, self.calibration_mode,image_type)


            else:
                BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                RED_chan_reg = np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))

                print("BLUE channel shape")
                print(BLUE_chan_reg.shape)

                zero_PV = [25,26,27,28,29,30,31,42,43,44,45,48,55] #,25,28,31,43]

            """The images have resolution 300 dpi, which is 300 pixels/inch.
            There is 25.4 mm in an inch. Therefore we have 300/25.4 = 11.81 pixels
            per mm. To get 2mm in pixels. We therefore multiply it with 11.81"""
            self.ROI_pixel_size = round(ROI_size*11.81)
        #Splitting each channel for calibration
            BLUE_chan_ROI = BLUE_chan_reg[:num_images, rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size]

            GREEN_chan_ROI = GREEN_chan_reg[:num_images,rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size]
            RED_chan_ROI = RED_chan_reg[:num_images,rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size]

            GREY_chan_ROI = GREY_chan_reg[:num_images,rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size]

            print("image type {}".format(image_type))
            print(BLUE_chan_reg.shape)
            print(np.mean(RED_chan_ROI))

            """
            zero_PV = [25,26,27,28,29,30,31,42,43,44,45,48,55] #,25,28,31,43]

            for i in zero_PV:
                fig = plt.figure()
                st = fig.suptitle("image {}".format(i), fontsize="x-large")
                ax1 = fig.add_subplot(2,2,1)
                ax1.set_title("BLUE")
                ax1.imshow(BLUE_chan_ROI[i])

                ax2 = fig.add_subplot(2,2,2)
                ax2.set_title("GREEN")
                ax2.imshow(GREEN_chan_ROI[i])

                ax3 = fig.add_subplot(2,2,3)
                ax3.set_title("RED")
                ax3.imshow(RED_chan_ROI[i])

                ax4 = fig.add_subplot(2,2,4)
                ax4.set_title("GREY")
                ax4.imshow(GREY_chan_ROI[i])

                plt.show()"""
            print("\n color channel extraction for {} complete".format(image_type))
            return BLUE_chan_ROI, GREEN_chan_ROI, RED_chan_ROI, GREY_chan_ROI

        else:
            if self.image_registration:
                if self.open:
                    if image_type == "image":
                        """
                        We use the same background and control images, so there is no need
                        to register twice.
                        """
                        BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan, self.calibration_mode)
                        print(BLUE_chan_reg.shape)
                        print("\n\n\n\n hey \n\n\n\n")
                        if self.registration_save:
                            np.save(self.reg_path + "\\BLUE_open{}".format(image_type), BLUE_chan)
                            np.save(self.reg_path + "\\GREEN_open{}".format(image_type), GREEN_chan)
                            np.save(self.reg_path + "\\RED_open{}".format(image_type), RED_chan)
                            np.save(self.reg_path + "\\GREY_open{}".format(image_type), GREY_chan)
                    else:
                        #if not image type == image, then it's background, and these are the same as for calibration
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                        RED_chan_reg = np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))
                elif self.grid:
                    if image_type == "image":
                        BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, GREEN_chan, RED_chan, GREY_chan, self.calibration_mode, image_type)
                        print(BLUE_chan_reg.shape)
                        print("blue chan reg\n\n\n\n")
                        if self.registration_save:
                            np.save(self.reg_path + "\\BLUE_grid_{}_2".format(image_type), BLUE_chan) #_2 because I was afraid to remove the previous registration
                            np.save(self.reg_path + "\\GREEN_grid_{}_2".format(image_type), GREEN_chan)
                            np.save(self.reg_path + "\\RED_grid_{}_2".format(image_type), RED_chan)
                            np.save(self.reg_path + "\\GREY_grid_{}_2".format(image_type), GREY_chan)
                    else:
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                        RED_chan_reg = np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))


            else:
                if self.open:
                    if image_type == "image":
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_open_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_open_{}.npy".format(image_type))
                        RED_chan_reg = np.load(self.reg_path + "\\RED_open_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_open_{}.npy".format(image_type))
                    else:
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                        RED_chan_reg =np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))
                elif self.grid:
                    if image_type == "image":
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_grid_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_grid_{}.npy".format(image_type))
                        RED_chan_reg = np.load(self.reg_path + "\\RED_grid_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_grid_{}.npy".format(image_type))

                    else:
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                        RED_chan_reg =np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))
            """
            We want to change the shape of the measurement films, because of the
            hars edges, and the narrow end of the film, where intensity is
            approximately 1.
            """
            #self.measurement_shape = (rows//2 + 200 - (rows//2 - 150), columns//2 + 125 - (columns//2 - 125))
            #print(self.measurement_shape)
            #self.measurement_shape = (650-50,260-210)
            self.measurement_shape = (self.measurement_crop[1] - self.measurement_crop[0],
                                      self.measurement_crop[3] - self.measurement_crop[2])
            print(self.measurement_shape)

            plt.imshow(RED_chan_reg[0,self.measurement_crop[0]:self.measurement_crop[1],
                                self.measurement_crop[2]:self.measurement_crop[3]], cmap = "viridis")
            plt.close()

            #RED_chan_reg[RED_chan_reg == 0] = 1e-4
            print("\n color channel extraction and image registration complete")
            print(RED_chan_reg.shape)
            if self.grid:
                plt.imshow(RED_chan_reg[0,self.measurement_crop[0]:self.measurement_crop[1],
                                    self.measurement_crop[2]:self.measurement_crop[3]], cmap = "viridis")
                plt.close()

            return RED_chan_reg[:num_images, self.measurement_crop[0]:self.measurement_crop[1],
                                self.measurement_crop[2]:self.measurement_crop[3]]
            #return RED_chan_reg[:num_images, 50:650,210:260]

    def netOD_calculation(self, background_images,control_images,images, films_per_dose, plot = False, plot_title = None):
        channel = "BLUE"
        """
        This is where the netOD per irradiated film is measured. For each dosepoint
        [0.1,0.2,0.5,1,2,5,10] we have 8 dosimetry films.

        We find the weighted average pixel value for the background (black) and control film.
        Then we use these to find the netOD of each irradiated film.
        """
        #Finding PV background and control

        mean_bckg_PV = np.zeros(len(background_images))
        sigma_bckg_PV = np.zeros(len(background_images))
        mean_ctrl_PV = np.zeros(len(control_images))
        sigma_ctrl_PV = np.zeros(len(control_images))
        #PV_weight = np.zeros(len(self.background_img))
        for i in range(len(control_images)):
            if i < len(background_images):
                mean_bckg_PV[i] = np.mean(background_images[i])
                sigma_bckg_PV[i] = np.std(background_images[i])/np.sqrt(background_images.shape[1]*background_images.shape[2]) #standard error

            mean_ctrl_PV[i] = np.mean(control_images[i])
            sigma_ctrl_PV[i] = np.std(control_images[i])/np.sqrt(control_images.shape[1]*control_images.shape[2]) #standard error

        PV_bckg_weight = (1/sigma_bckg_PV)**2/(np.sum((1/sigma_bckg_PV)**2))
        PV_ctrl_weight = (1/sigma_ctrl_PV)**2/(np.sum((1/sigma_ctrl_PV)**2))

        PV_bckg = np.average(mean_bckg_PV, weights = PV_bckg_weight)
        PV_ctrl = np.average(mean_ctrl_PV, weights = PV_ctrl_weight)

        # sigma_bckg_PV = np.std(mean_bckg_PV)/np.sqrt(len(background_images))
        # sigma_ctrl_PV = np.std(mean_ctrl_PV)/np.sqrt(len(control_images))

        """finding mean standard error"""
        print(background_images.shape, control_images.shape)
        #ROI_bckg = background_images.reshape(background_images.shape[0], background_images.shape[1]*background_images.shape[2])
        #ROI_ctrl = control_images.reshape(control_images.shape[0], control_images.shape[1]*control_images.shape[2])

        #stderr_bckg = np.sqrt(np.sum(np.var(ROI_bckg, axis = 0)))/background_images.shape[0]#(background_images.shape[1]*background_images.shape[2])))  #combining standard error of all films
        #stderr_ctrl = np.sqrt(np.sum(np.var(ROI_ctrl, axis = 0)))/control_images.shape[0]#(control_images.shape[1]*control_images.shape[2])))

        # sigma_bckg_PV = np.sqrt(np.sum(sigma_bckg_PV**2)/len(background_images)) #mean standard error of background image
        # sigma_ctrl_PV = np.sqrt(np.sum(sigma_ctrl_PV**2)/len(control_images))

        # mean_sigma_bckg_PV = np.sqrt(len(sigma_bckg_PV)/(np.sum(1/sigma_bckg_PV**2)))  #from devic
        # mean_sigma_ctrl_PV = np.sqrt(len(sigma_ctrl_PV)/(np.sum(1/sigma_ctrl_PV**2)))

        mean_sigma_bckg_PV = 1/np.sqrt(np.sum(1/sigma_bckg_PV**2))  #from devic
        mean_sigma_ctrl_PV = 1/np.sqrt(np.sum(1/sigma_ctrl_PV**2))


        # tmp1 = np.sum(PV_bckg_weight*mean_bckg_PV**2)/np.sum(PV_bckg_weight) - PV_bckg**2
        # tmp2 = np.sum(PV_bckg_weight**2)/((np.sum(PV_bckg_weight))**2 - np.sum(PV_bckg_weight**2))
        # mean_sigma_bckg_PV = np.sqrt(tmp1*tmp2)

        print(mean_sigma_bckg_PV, mean_sigma_ctrl_PV, len(sigma_bckg_PV), len(sigma_ctrl_PV), PV_bckg, PV_ctrl)
        # print(sigma_bckg_PV, sigma_ctrl_PV)


        # sigma_bckg_PV = np.std(mean_bckg_PV)/np.sqrt(len(background_images))
        # sigma_ctrl_PV = np.std(mean_ctrl_PV)/np.sqrt(len(control_images))


        print("PV_bckg  PV_ctrl")
        print(PV_bckg,PV_ctrl)

        #maybe I should just: sigma_bckg_PV = np.std(mean_bckg_PV) ??

        #Finding netOD in the irradiated films

        #We will make a net OD for each film. Every 8th film will be a new dose.

        """
        In calibration mode, we wish to find the weighted average netOD of each
        film. When we're not in calibration mode, we want to find netOD in
        each pixel.
        """
        if plot:
            """
            Plotting mean pixel values within an ROI scaled using the average control PV
            """
            plt.style.use("seaborn")
            plt.title(plot_title)
            plt.xlabel("Dose [Gy]", fontsize = 15)
            plt.ylabel(r"$\frac{PV_{irradiated}}{PV_{ctrl}}$", fontsize = 20)
            dose_axis = np.array([0.1,0.2,0.5,10,1,2,5])
            dose = dose_axis[0]
            color = ["b","g","r","c","m","y","black"]
            c = color[0]
        if self.calibration_mode:
            netOD = np.zeros(len(images))
            sigma_img_PV = np.zeros(len(images)) #vet ikke hva jeg skal gjøre med denne enda
            PV_img = np.zeros(len(images))
            #finding netOD for each film
            idx = 0
            for i in range(0,len(images)):
                # mean_img_PV = np.mean(images[i])
                PV_img[i] = np.mean(images[i]) #finding mean pixel value within ROI
                sigma_img_PV[i] = np.std(images[i]) #finding std from the pixel values
                if plot:
                    #plt.errorbar(dose,mean_img_PV/PV_ctrl, fmt = "o", yerr = sigma_img_PV/PV_ctrl, c = c, markersize = 5) errorbar not necessary
                    plt.plot(dose, np.mean(PV_img[i])/PV_ctrl, "o", c = c, markersize = 6)
                    if (i + 1) % 8 == 0 and i+1 != len(images):

                        idx += 1
                        #print(idx,i+1)
                        dose = dose_axis[idx]
                        c = color[idx]

                #print(mean_img_PV,i)
                #sigma_img_PV[idx] = np.std(images[i])
                tmp = (PV_ctrl-PV_bckg)/(PV_img[i]-PV_bckg)


                netOD[i] = max([0,np.log10(tmp)])
                """if channel == "BLUE":
                    print((PV_ctrl-PV_bckg)/(PV_img[i]-PV_bckg))
                    if netOD[i] == 0:
                        print("---------")
                        print(PV_img[i], PV_bckg)
                        print("---------")"""


            plt.close()
            print("error shapes")
            print(sigma_ctrl_PV.shape,mean_sigma_bckg_PV.shape,sigma_img_PV.shape)
            return netOD, PV_img, mean_sigma_ctrl_PV, mean_sigma_bckg_PV, sigma_img_PV, PV_ctrl, PV_bckg
            #return netOD, PV_img, sigma_cont_PV, sigma_bckg_PV, sigma_img_PV, PV_ctrl, PV_bckg

        else:
            PV_img = np.zeros(images.shape)
            netOD = np.zeros((len(images),self.measurement_shape[0],self.measurement_shape[1]))
            print(netOD.shape)
            print(PV_img.shape)
            print("fsbjknjksgf")
            sigma_img_PV = np.zeros(len(images)) #vet ikke hva jeg skal gjøre med denne enda
            low = 0
            high = 0
            for i in range(0,len(images)):
                PV_img[i] = images[i]
                sigma_img_PV[i] = np.std(images[i])
                """
                if len(np.argwhere(img_PV == 0)[:,0]) != 0:
                    plt.imshow(img_PV)
                    plt.colorbar()
                    plt.show()
                    print(self.sorted_files[i])
                """
                #all negative values are set to 0
                diff = PV_img[i]-PV_bckg
                diff[diff < 0] = 1e14
                """
                After registration, images will be padded with 0. this will cause
                the sum img_PV-PV_bckg < 0, making the negative sums really large,
                causes the log to become negative. These values will be clipped to 0
                """
                img_OD = np.clip(np.log10((PV_ctrl-PV_bckg)/(diff)),0,66e4)
                netOD[i] = img_OD

            print(netOD.shape)
            # plt.imshow(netOD[0])
            # plt.show()
            print("\n netOD calculation is complete")
            return netOD, PV_img, mean_sigma_ctrl_PV, mean_sigma_bckg_PV, sigma_img_PV, PV_ctrl, PV_bckg


    def calibrate(self, ROI_size, films_per_dose, channel = "RED"):
        """
        This function calibrates the dosimetry films, by splitting the images
        into background (black), control (0 Gy) and images (irradiated). Then it finds the average
        pixel value of the image of interest, and computed the netOD.
        The function might be turned into a loop for convenience
        """

        """
        image types is used to give the images their right name
        """
        image_types = ["background","control","image"]
        plot_title = ["BLUE", "GREEN", "RED", "GRAY"]
        if self.calibration_mode:
            BLUE_chan_ROI, GREEN_chan_ROI, RED_chan_ROI, GREY_chan_ROI = self.color_channel_extraction(self.images, self.num_films, ROI_size, image_types[2])
            print("RED channel mean intensity value")

            BLUE_bckg, GREEN_bckg, RED_bckg, GREY_bckg = self.color_channel_extraction(self.background_img, self.num_background, ROI_size, image_types[0])
            print("RED channel mean intensity value backgroud")

            BLUE_cont, GREEN_cont, RED_cont, GREY_cont = self.color_channel_extraction(self.control_img, self.num_control, ROI_size, image_types[1])
            print("RED channel mean intensity value control")

            ROI = [BLUE_chan_ROI, GREEN_chan_ROI, RED_chan_ROI, GREY_chan_ROI]
            bckg = [BLUE_bckg, GREEN_bckg, RED_bckg, GREY_bckg]
            ctrl = [BLUE_cont, GREEN_cont, RED_cont, GREY_cont]
            """
            Finding the netOD for the 0 Gy films to be appended on the other netOD arrays
            """
            #defining standard deviation of control images ROI
            #4 color channels, 8 control images and 4 background images

            self.PV_ctrl = np.zeros(4)
            self.PV_bckg = np.zeros(4)
            # self.sigma_ctrl_PV = np.zeros((4,8))
            # self.sigma_bckg_PV = np.zeros((4,4))
            self.sigma_ctrl_PV = np.zeros(4)
            self.sigma_bckg_PV = np.zeros(4)
            self.sigma_img_PV = np.zeros((4,self.num_films + 8))
            self.dOD = np.zeros((4,8))
            self.netOD = np.zeros((4, self.num_films + 8))
            self.PV = np.zeros((4, self.num_films + 8))
            color = ["BLUE","GREEN","RED","GREY"]
            for i in range(len(self.netOD)):
                ROI_ = ROI[i]
                bckg_ = bckg[i]
                ctrl_ = ctrl[i]
                print(color[i])
                self.netOD[i,:films_per_dose], self.PV[i,:films_per_dose],\
                self.sigma_ctrl_PV[i], self.sigma_bckg_PV[i],self.sigma_img_PV[i,:films_per_dose], self.PV_ctrl[i],\
                self.PV_bckg[i] = self.netOD_calculation(bckg_, ctrl_, ctrl_, films_per_dose)
                """--------------------------------------------------------------------"""
                self.netOD[i,films_per_dose:self.num_films + films_per_dose], self.PV[i,films_per_dose:self.num_films + films_per_dose],\
                _, _, self.sigma_img_PV[i,films_per_dose:self.num_films+films_per_dose], _,\
                _ = self.netOD_calculation(bckg_,ctrl_,ROI_, films_per_dose, plot = True, plot_title = plot_title[i])
                """--------------------------------------------------------------------"""
                #Estimating uncertainty in OD calculations for all color channels
            #pooled_sigma_ctrl = np.array(np.sqrt(np.sum([self.sigma_ctrl_PV[i]**2 for i in range(self.sigma_ctrl_PV.shape[0])], axis = 1)/self.sigma_ctrl_PV.shape[1]))
            #pooled_sigma_bckg = np.array(np.sqrt(np.sum([self.sigma_bckg_PV[i]**2 for i in range(self.sigma_bckg_PV.shape[0])], axis = 1)/self.sigma_bckg_PV.shape[1]))

            #print("Ultimate shape master")
            #print(pooled_sigma_ctrl.shape,pooled_sigma_bckg.shape,self.PV_ctrl.shape ,self.PV_bckg.shape,self.PV.shape, self.sigma_img_PV.T.shape)

            """
            Alternative method of finding sigma ctrl/bckg
            """
            tmp1 = (self.sigma_ctrl_PV/(self.PV_ctrl - self.PV_bckg))**2
            tmp2 = (self.sigma_img_PV.T/(self.PV.T - self.PV_bckg))**2
            tmp3 = ((self.PV_ctrl - self.PV_bckg)/((self.PV.T - self.PV_bckg)*(self.PV_ctrl - self.PV_bckg)))**2 * self.sigma_bckg_PV**2


            #tmp1 = (pooled_sigma_ctrl/(self.PV_ctrl - self.PV_bckg))**2
            #tmp2 = (self.sigma_img_PV.T/(self.PV.T - self.PV_bckg))**2
            #tmp3 = ((self.PV_ctrl - self.PV_bckg)/((self.PV.T - self.PV_bckg)*(self.PV_ctrl - self.PV_bckg)))**2 * pooled_sigma_bckg**2
            # self.dOD = 1/np.log(10) * np.sqrt((pooled_sigma_ctrl/(self.PV_ctrl - self.PV_bckg))**2 +\
            #                                   (self.sigma_img_PV/(self.PV.T - self.PV_bckg))**2 +\
            #                                   ((self.PV_ctrl - self.PV_bckg)/((self.PV.T - self.PV_bckg)*(self.PV_ctrl - self.PV_bckg)))**2 * self.sigma_bckg_PV**2)
            print(tmp1.shape,tmp2.shape,tmp3.shape)
            self.dOD = 1/np.log(10)*np.sqrt(tmp1 + tmp2 + tmp3).T
            print(self.dOD.shape)
            #First we measure netOD for 0 Gy before measuring for the remaining dose points
            """self.netOD[0,:films_per_dose], self.PV[0,:films_per_dose], self.sigma_ctrl_PV[0], self.sigma_bckg_PV[0],self.sigma_img_PV[0,:films_per_dose], self.PV_ctrl[0], self.PV_bckg[0] = self.netOD_calculation(BLUE_bckg, BLUE_cont, BLUE_cont, films_per_dose)
            self.netOD[1,:films_per_dose], self.PV[0,:films_per_dose], self.sigma_ctrl_PV[1], self.sigma_bckg_PV[1],self.sigma_img_PV[1,:films_per_dose], self.PV_ctrl[1], self.PV_bckg[1] = self.netOD_calculation(GREEN_bckg, GREEN_cont, GREEN_cont, films_per_dose)
            self.netOD[2,:films_per_dose], self.PV[0,:films_per_dose], self.sigma_ctrl_PV[2], self.sigma_bckg_PV[2],self.sigma_img_PV[2,:films_per_dose], self.PV_ctrl[2], self.PV_bckg[2] = self.netOD_calculation(RED_bckg, RED_cont, RED_cont, films_per_dose)
            self.netOD[3,:films_per_dose], self.PV[0,:films_per_dose], self.sigma_ctrl_PV[3], self.sigma_bckg_PV[3],self.sigma_img_PV[3,:films_per_dose], self.PV_ctrl[3], self.PV_bckg[3] = self.netOD_calculation(GREY_bckg, GREY_cont, GREY_cont, films_per_dose)


            self.netOD[0,films_per_dose:self.num_films + films_per_dose], self.PV[0,films_per_dose:self.num_films + films_per_dose], _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+films_per_dose], _, _ = self.netOD_calculation(BLUE_bckg,BLUE_cont,BLUE_chan_ROI, films_per_dose, plot = True, plot_title = "BLUE")
            self.netOD[1,films_per_dose:self.num_films + films_per_dose], self.PV[0,films_per_dose:self.num_films + films_per_dose], _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+films_per_dose], _, _ = self.netOD_calculation(GREEN_bckg,GREEN_cont,GREEN_chan_ROI, films_per_dose,plot = True, plot_title = "GREEN")
            self.netOD[2,films_per_dose:self.num_films + films_per_dose], self.PV[0,films_per_dose:self.num_films + films_per_dose], _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+films_per_dose], _, _ = self.netOD_calculation(RED_bckg,RED_cont,RED_chan_ROI, films_per_dose,plot = True, plot_title = "RED")
            self.netOD[3,films_per_dose:self.num_films + films_per_dose], self.PV[0,films_per_dose:self.num_films + films_per_dose], _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+films_per_dose], _, _ = self.netOD_calculation(GREY_bckg,GREY_cont,GREY_chan_ROI, films_per_dose,plot = True, plot_title = "GRAY")"""

            """
            Reshaping netOD for easier plotting. We have 4 color channels, 8 doses
            0, 0.1, 0.2 etc. and 8 films per dose.
            """
            self.netOD = self.netOD.reshape((4,8,8))
            self.PV = self.PV.reshape((4,8,8))
            self.dOD = self.dOD.reshape((4,8,8))

            return self.netOD,self.sigma_img_PV, self.sigma_bckg_PV, self.sigma_ctrl_PV, self.dOD

        else:


            ROI = self.color_channel_extraction(self.images, self.num_films, ROI_size, image_types[2])
            bckg = self.color_channel_extraction(self.background_img, self.num_background, ROI_size, image_types[0])
            cont = self.color_channel_extraction(self.control_img, self.num_control, ROI_size, image_types[1])

            self.netOD, self.PV, self.sigma_ctrl_PV, self.sigma_bckg_PV, self.sigma_img_PV, self.PV_ctrl, self.PV_bckg = self.netOD_calculation(bckg, cont, ROI, films_per_dose)

            tmp1 = (self.sigma_ctrl_PV/(self.PV_ctrl - self.PV_bckg))**2
            # tmp2 = (self.sigma_img_PV.T/(self.PV.T - self.PV_bckg))**2  #Skipping this because we assume no uncertainty in PV for measurement films

            tmp3 = ((self.PV_ctrl - self.PV_bckg)/((self.PV.T - self.PV_bckg)*(self.PV_ctrl - self.PV_bckg)))**2 * self.sigma_bckg_PV**2

            print(tmp1.shape,tmp3.shape)
            self.dOD = 1/np.log(10)*np.sqrt(tmp1  + tmp3).T
            print(self.dOD.shape)
            print("measurement films netOD shape")
            print(self.netOD.shape)
            return self.netOD,self.sigma_img_PV, self.sigma_bckg_PV, self.sigma_ctrl_PV, self.dOD

    def netOD_split(self, doses, bandwidth, bandwidth_stepsize, channel = "RED", no_split = False):
        self.low_response_OD = []
        self.high_response_OD = []
        self.low_res_dose = []
        self.high_res_dose = []
        self.dOD_low = []
        self.dOD_high = []
        self.no_split = np.zeros((self.netOD.shape[1],self.netOD.shape[2]))

        """Making a list of bandwidths used in the kernel Density algorithm.
        This makes sure that we only have one local minima. We choose values below and
        above this local minima, so we can separate low and high response EBT3 films.
        For now we only do this for the RED channel."""

        if self.calibration_mode:
            print("calibration mode activated")
            for i in range(self.netOD.shape[1]):
                print("Film {}/{}".format(i+1,self.netOD.shape[1]))
                """
                Looping over doses, while splitting netOD into low and high respons
                """
                if channel == "RED":
                    print("splitting RED channel OD")
                    OD = self.netOD[2,i,:].reshape(-1,1)
                elif channel == "BLUE":
                    print("splitting BLUE channel OD")
                    OD = self.netOD[0,i,:].reshape(-1,1)
                elif channel == "GREEN":
                    print("splitting GREEN channel OD")
                    OD = self.netOD[1,i,:].reshape(-1,1)
                elif channel == "GREY":
                    print("splitting GREY channel OD")
                    OD = self.netOD[3,i,:].reshape(-1,1)

                mi = []
                iter = 0
                while len(mi) == 0 or len(mi) > 1:
                    #print(iter)
                    #print(bandwidth[i])
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth[i]).fit(OD)
                    s = np.linspace(0,max(OD),100)
                    kde_scores = kde.score_samples(s)
                    #Henter lokal minima for red channel
                    mi = argrelextrema(kde_scores, np.less)[0]
                    if no_split:
                        if iter > 1000:
                            print("No split response in 25k iterations")
                            print(doses[i])
                            self.no_split[i] = OD[:,0]
                            break
                    if len(mi) == 0:
                        #print("Bandwidth too large")
                        if bandwidth[i] < bandwidth_stepsize:
                            self.no_split[i] = OD[:,0]
                            break
                        else:
                            bandwidth[i] -= bandwidth_stepsize
                    elif 1 < len(mi):
                        #print("Bandwidth too small")
                        bandwidth[i] += bandwidth_stepsize
                    elif len(mi) == 1:
                        #print("Bandwidth of {} perfect".format(bandwidth[i]))
                        # plt.plot(s,kde_scores)
                        # plt.plot(s[mi[0]],kde_scores[mi[0]],"*")
                        # plt.show()


                        low_res_idx = np.argwhere(OD < s[mi[0]])
                        high_res_idx = np.argwhere(OD > s[mi[0]])

                        #double checking that we have split only necessary for 1310
                        if no_split:
                            if np.abs(np.max(OD[low_res_idx[:,0]]) -  np.min(OD[high_res_idx[:,0]])) < 2*np.std(OD):
                                print("what")
                                self.no_split[i] = OD[:,0]
                                break


                        self.low_response_OD = np.append(self.low_response_OD, OD[low_res_idx[:,0]])
                        self.high_response_OD = np.append(self.high_response_OD, OD[high_res_idx[:,0]])

                        self.low_res_dose = np.append(self.low_res_dose, [doses[i] for j in range(len(low_res_idx))], axis = 0)
                        self.high_res_dose = np.append(self.high_res_dose, [doses[i] for j in range(len(high_res_idx))], axis = 0)

                        #self.dOD_low = np.append(self.dOD_low, self.dOD[i])
                        #self.dOD_high

                        sort_idx_low = np.argsort(self.low_res_dose)
                        sort_idx_high = np.argsort(self.high_res_dose)

                    iter += 1

            self.low_response_OD = self.low_response_OD[sort_idx_low]
            self.high_response_OD = self.high_response_OD[sort_idx_high]
            self.low_res_dose = self.low_res_dose[sort_idx_low]
            self.high_res_dose = self.high_res_dose[sort_idx_high]

            print("\n netOD splitting is complete")
            if no_split:
                if len(self.no_split[self.no_split > 0]) != 0:
                    return np.array(self.low_response_OD), np.array(self.high_response_OD), np.array(self.low_res_dose), np.array(self.high_res_dose), bandwidth, np.array(self.no_split)
                else:
                    return self.low_response_OD, self.high_response_OD, self.low_res_dose, self.high_res_dose, bandwidth
            else:
                return self.low_response_OD, self.high_response_OD, self.low_res_dose, self.high_res_dose, bandwidth

        else:
            print("not calibration mode")
            mi = []
            print(len(mi))
            iter = 0
            # print(0 == len(mi)  < 2)
            while len(mi) == 0 or len(mi) > 1:
                print("Iteration : {}".format(iter))
                mean_netOD = np.array([np.mean(self.netOD[i]) for i in range(self.netOD.shape[0])]).reshape(-1,1)
                kde = KernelDensity(kernel = "gaussian", bandwidth = bandwidth).fit(mean_netOD)
                s = np.linspace(np.min(mean_netOD), np.max(mean_netOD),1000).reshape(-1,1)
                kde_scores = kde.score_samples(s)
                mi = argrelextrema(kde_scores, np.less)[0]
                #plt.plot(s,kde_scores)
                #plt.plot(s[mi[0]],kde_scores[mi[0]],"*")
                #plt.show()
                if len(mi) == 0:
                    print("Bandwidth too large")
                    bandwidth -= 0.0001
                elif 1 < len(mi):
                    print("Bandwidth too small")
                    bandwidth += 0.0001
                elif len(mi) == 1:
                    print("Bandwidth of {} perfect".format(bandwidth))
                    plt.plot(s,kde_scores)
                    plt.plot(s[mi[0]],kde_scores[mi[0]],"*")
                    plt.close()
                iter += 1

            #getting indexes for low and high response measurement films, which we need for dose error
            self.low_img_idx = np.argwhere(mean_netOD < s[mi[0]])[:,0]
            self.high_img_idx = np.argwhere(mean_netOD > s[mi[0]])[:,0]
            #print(low_img,high_img)
            return self.low_img_idx, self.high_img_idx


    """
    Fitting using scipy least squares instead of scipy curve fit
    """


    def EBT_model(self,netOD,params,model_type):
        """
        This is the model we wish to fit to the netOD.
        a, b, and n are the possible fitting parameters in this order.
        """
        if model_type == 1:
            #devic 2004
            return params[0]*netOD + params[1]*netOD**params[2]
        elif model_type ==2:
            #gafchromic
            return params[0] + params[1]/(netOD - params[2])
        elif model_type ==3:
            #devic 2018
            return params[0]*netOD/(1 + params[1]*netOD)

    def RSS_func(self,params, netOD, y, model_type):
        if model_type == 1:
            #devic 2004
            return (params[0]*netOD + params[1]*netOD**params[2]) - y
        elif model_type == 2:
            #gafchromic
            return (params[0] + params[1]/(netOD - params[2])) - y
        elif model_type ==3:
            #devic 2018
            return (params[0]*netOD/(1 + params[1]*netOD)) - y


    def EBT_fit(self, doses, num_fitting_params, low_response_OD = None, high_response_OD = None,
                low_res_dose = None, high_res_dose = None,model_type = 1, OD = None):
        """
        For each color channel we fit the EBT model of choice.
        We choose to fit all the films, and not take the mean. To preserve as
        much information as possible.
        We therefore stack the dose and unravel the netOD, and pass them
        through the scipy.optimize curve fit function. If the doses are:
        [0.1,0.2,0.5], with 2 films each. Their netOD is e.g.:

            0.1 Gy          0.2 Gy       0.5 Gy
        [[0.01, 0.03] , [0.04, 0.01], [0.03,0.08]]

        By stacking the doses we obtain a new dose array:

        [0.1 , 0.1 , 0.2, 0.2, 0.5, 0.5]

        and by unraveling netOD we get:

        [0.01, 0.03 , 0.04, 0.01, 0.03,0.08]

        And this is what we enter into the curve fit function.
        """

        """
        Now we have the correct unravelled netOD vector we can fit. But we need the corresponding doses.
        """

        """
        Ta med denne senere eventuelt
        """
        #self.std_err = np.zeros(self.netOD.shape[0])

        if model_type == 1:
            # x0 = np.array([0.5,0.5,0.5])
            x0 = np.array([1,1,1])
        if model_type == 2:
            x0 = np.array([-1,1,1])

        if model_type == 3:
            x0 = np.array([1,1])

        if np.all(OD) != None:

            """
            If OD != None then no low or high OD is found
            """
            print(np.shape(OD),np.shape(doses))

            fit = least_squares(self.RSS_func, x0, args = (OD,doses,model_type), method = "lm")
            self.fitting_param = fit.x
            k = 3
            df = len(OD)- k - 1
            hessian_approx_inv = np.linalg.inv(fit.jac.T.dot(fit.jac)) #follows H^-1 approx J^TJ
            std_err_res = np.sqrt(np.sum(fit.fun**2)/df)**2
            self.param_cov = std_err_res * hessian_approx_inv



            return self.fitting_param, np.diag(self.param_cov)

        else:
            """
            Testing how robust the least squares is
            """
            """x0 = x0
            f = open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Fitting results LM\\fitting_params_.txt", "a")
            for i in range(10):
                fit_low = least_squares(self.RSS_func, x0,
                                    args = (self.low_response_OD,self.low_res_dose, model_type),
                                    method = "lm")
                fit_high = least_squares(self.RSS_func, x0,
                                    args = (self.high_response_OD,self.high_res_dose,model_type),
                                    method = "lm")
                # x0 = [1,1,-1]
                x0 = np.array([np.random.randn(1),np.random.randn(1),abs(np.random.randn(1))]).flatten()
                self.fitting_param_low = fit_low.x
                self.fitting_param_high = fit_high.x
                self.residual_low = np.sum(fit_low.fun**2)/len(fit_low.fun)
                self.residual_high = np.sum(fit_high.fun**2)/len(fit_high.fun)

                fitting_param_low_round = [round(i,3) for i in self.fitting_param_low]
                fitting_param_high_round = [round(i,3) for i in self.fitting_param_high]
                residual_low = round(self.residual_low,3)
                residual_high = round(self.residual_high,3)
                if i == 0:
                    f.write("Calibration curve results\n")
                    f.write("Fitting Params & Residuals\n")
                f.write("----------------------------\n")
                f.write("Starting point     {}\n".format([round(i,3) for i in x0]))
                f.write("LOW                     HIGH\n") #3 tabs
                f.write("{} - {}\n".format(fitting_param_low_round, fitting_param_high_round))
                f.write("{} - {}\n".format(residual_low,residual_high))
            f.close()"""




            #if len(self.no_split[self.no_split != 0]) == 0:
            if np.all(low_response_OD) == None:
                fit_low = least_squares(self.RSS_func, x0,
                                    args = (self.low_response_OD,self.low_res_dose, model_type),
                                    method = "lm")
                fit_high = least_squares(self.RSS_func, x0,
                                    args = (self.high_response_OD,self.high_res_dose,model_type),
                                    method = "lm")
            else:
                fit_low = least_squares(self.RSS_func, x0,
                                    args = (low_response_OD,low_res_dose, model_type),
                                    method = "lm")
                fit_high = least_squares(self.RSS_func, x0,
                                    args = (high_response_OD,high_res_dose,model_type),
                                    method = "lm")
            no_split = False
            if no_split:

                print("no-splits encountered")
                idx, occurences = np.unique(np.argwhere(self.no_split != 0)[:,0], return_counts = True)
                print(occurences)
                no_split_dose = np.repeat(doses[idx], len(self.no_split))
                no_split_OD = self.no_split[self.no_split != 0]
                num_no_split = len(no_split_dose)
                """
                We add no split OD, do a fit then find which category (low or high) gives
                the no split OD smallest residual. This is the category we'll choose
                """
                print(len(np.append(self.low_response_OD,no_split_OD)),
                      len(np.append(self.low_res_dose,no_split_dose)))
                fit_low = least_squares(self.RSS_func, x0,
                                    args = (np.append(self.low_response_OD,no_split_OD),
                                    np.append(self.low_res_dose,no_split_dose), model_type),
                                    method = "lm")
                fit_high = least_squares(self.RSS_func, x0,
                                    args = (np.append(self.high_response_OD,no_split_OD),
                                    np.append(self.high_res_dose,no_split_dose),model_type),
                                    method = "lm")
                #what is the residual for our non split points
                #finding which
                fit_MSE_low = np.mean((fit_low.fun[-len(idx)*8:]**2).reshape(len(idx),8), axis = 1)
                fit_MSE_high = np.mean((fit_high.fun[-len(idx)*8:]**2).reshape(len(idx),8), axis = 1)

                """
                We see that all no splits fall under low response category
                """


                #0 == low response, 1 == high response
                """
                Find which index in [MSE_low, MSE_high] is smallest. If 0 is smallest, then
                the dose has a low response, and opposite if 1
                """
                response = np.argmin([fit_MSE_low, fit_MSE_high], axis = 0)
                #using boolean index
                low_response = response == 0
                high_response = response == 1

                #forcing 10 Gy to be included
                high_response[1] = True
                high_response[0]  =True




                #reshape to have control over which doses we are looking at
                no_split_OD = no_split_OD.reshape(len(idx),8)
                no_split_dose = no_split_dose.reshape(len(idx),8)



                """
                Now we only add the doses with low or high response
                """

                new_low_response_OD = np.append(self.low_response_OD,no_split_OD[low_response])
                new_high_response_OD = np.append(self.high_response_OD,no_split_OD[high_response])
                new_low_response_dose = np.append(self.low_res_dose,no_split_dose[low_response])
                new_high_response_dose = np.append(self.high_res_dose,no_split_dose[high_response])


                fit_low = least_squares(self.RSS_func, x0,
                                    args = (new_low_response_OD,
                                    new_low_response_dose, model_type),
                                    method = "lm")
                fit_high = least_squares(self.RSS_func, x0,
                                    args = (new_high_response_OD,
                                    new_high_response_dose,model_type),
                                    method = "lm")

            """
            We want the variance of each parameter, so we need the covariance matrix.
            The covariance matrix can be found using the formula:
            cov = sigma_res^2 * H^-1.
            In LM, the hessian is approximated to be J^TJ (gauss newton). Because we only get the optimal Jacobian
            at convergence, we need to find J^TJ.
            The standard deviation of the residuals can be found like this
            sigma_res = sqrt(sum((y_hat - y)**2)/df), where df is degrees of freedom,
            which is n-k-1, for n datapoints and k parameters
            """
            k = 3
            df_low = len(self.low_response_OD)- k - 1
            hessian_approx_inv_low = np.linalg.inv(fit_low.jac.T.dot(fit_low.jac)) #follows H^-1 approx J^TJ
            std_err_res_low = np.sqrt(np.sum(fit_low.fun**2)/df_low)**2
            param_cov_low = std_err_res_low * hessian_approx_inv_low

            df_high = len(self.high_response_OD) - k - 1
            hessian_approx_inv_high = np.linalg.inv(fit_high.jac.T.dot(fit_high.jac)) #follows H^-1 approx J^TJ
            std_err_res_high = np.sqrt(np.sum(fit_high.fun**2)/df_high)**2
            param_cov_high = std_err_res_high * hessian_approx_inv_high


            self.fitting_param_low = fit_low.x
            self.fitting_param_high = fit_high.x
            print(np.sum(fit_low.fun**2), np.sum(fit_high.fun**2))
            print(df_low,df_high)

            print("Optimality")
            print(fit_low.optimality)

            print(np.sqrt(np.sum(fit_low.fun**2)/(len(fit_low.fun)-1))) #MSE

            #self.residual_low = fit_low.cost
            #self.residual_high = fit_high.cost
            #if len(self.no_split[self.no_split != 0]) == 0:
            if not no_split:
                return self.fitting_param_low, self.fitting_param_high, np.diag(param_cov_low), np.diag(param_cov_high), fit_low, fit_high#, self.residual_low, self.residual_high
            else:
                return self.fitting_param_low, self.fitting_param_high, \
                np.diag(param_cov_low), np.diag(param_cov_high), fit_low, fit_high,\
                new_low_response_OD, new_high_response_OD, new_low_response_dose, new_high_response_dose

        """if OD != None:
             popt, pcov = curve_fit(model, OD, doses)
             self.fitting_param = popt
             return self.fitting_param
        else:
            low_popt, low_pcov = curve_fit(model, self.low_response_OD, self.low_res_dose)
            high_popt, high_pcov = curve_fit(model, self.high_response_OD, self.high_res_dose)

            self.fitting_param_low = low_popt
            self.fitting_param_high = high_popt
            return self.fitting_param_low, self.fitting_param_high"""




        print("\n fitting is complete")




if __name__ == "__main__":
    folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"
    background_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Background"
    test_image = "EBT3_Calib_310821_Xray220kV_00Gy1_001.tif"
    film_calib = film_calibration(folder, test_image)

    images = film_calib.image_acquisition()

    films_per_dose = 8
    ROI_size = 2 #mm

    film_calib.calibrate(ROI_size, films_per_dose)

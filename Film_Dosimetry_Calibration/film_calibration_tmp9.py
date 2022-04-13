import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.optimize import least_squares
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
# from parallelized_kde import parallelized_kde
import random
from image_reg_split import image_reg


class film_calibration:
    def __init__(self, image_folder,background_folder, control_folder, test_image,\
                 background_image, control_image,  reg_path, measurement_crop = None,  calibration_mode = True,
                 image_registration = False, registration_save = True, grid = False, open = False):
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
            #print(self.images[i].shape)
        for i,filename in enumerate(background_files):
            self.background_img[i] = cv2.imread(os.path.join(self.background_folder,filename),-1)
        for i,filename in enumerate(first_control):
            self.control_img[i] = cv2.imread(os.path.join(self.control_folder,filename),-1)
            """
            tmp = cv2.imread(os.path.join(self.folder,filename),cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb)
            plt.colorbar()
            plt.show()"""

    def color_channel_extraction(self, image, num_images, ROI_size, image_type):

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
                BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, RED_chan, GREEN_chan, GREY_chan, self.calibration_mode)
                np.save(self.reg_path + "\\BLUE_calib_{}".format(image_type), BLUE_chan)
                np.save(self.reg_path + "\\GREEN_calib_{}".format(image_type), GREEN_chan)
                np.save(self.reg_path + "\\RED_calib_{}".format(image_type), RED_chan)
                np.save(self.reg_path + "\\GREY_calib_{}".format(image_type), GREY_chan)
            elif self.image_registration and not self.registration_save:
                print("Registered but not saved")
                BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, RED_chan, GREEN_chan, GREY_chan, self.calibration_mode)
            else:
                BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                RED_chan_reg = np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))

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
                        BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, RED_chan, GREEN_chan, GREY_chan, self.calibration_mode)
                        np.save(self.reg_path + "\\BLUE_open{}".format(image_type), BLUE_chan)
                        np.save(self.reg_path + "\\GREEN_open{}".format(image_type), GREEN_chan)
                        np.save(self.reg_path + "\\RED_open{}".format(image_type), RED_chan)
                        np.save(self.reg_path + "\\GREY_open{}".format(image_type), GREY_chan)
                    else:
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                        RED_chan_reg =np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))
                elif self.grid:
                    if image_type == "image":
                        BLUE_chan_reg, GREEN_chan_reg, RED_chan_reg, GREY_chan_reg = image_reg(BLUE_chan, RED_chan, GREEN_chan, GREY_chan, self.calibration_mode)
                        np.save(self.reg_path + "\\BLUE_grid_{}".format(image_type), BLUE_chan)
                        np.save(self.reg_path + "\\GREEN_grid_{}".format(image_type), GREEN_chan)
                        np.save(self.reg_path + "\\RED_grid_{}".format(image_type), RED_chan)
                        np.save(self.reg_path + "\\GREY_grid_{}".format(image_type), GREY_chan)
                    else:
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_calib_{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_calib_{}.npy".format(image_type))
                        RED_chan_reg =np.load(self.reg_path + "\\RED_calib_{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_calib_{}.npy".format(image_type))


            else:
                if self.open:
                    if image_type == "image":
                        BLUE_chan_reg = np.load(self.reg_path + "\\BLUE_open{}.npy".format(image_type))
                        GREEN_chan_reg = np.load(self.reg_path + "\\GREEN_open{}.npy".format(image_type))
                        RED_chan_reg = np.load(self.reg_path + "\\RED_open{}.npy".format(image_type))
                        GREY_chan_reg = np.load(self.reg_path + "\\GREY_open{}.npy".format(image_type))
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
            RED_chan_reg[RED_chan_reg == 0] = 1e-4
            print("\n color channel extraction and image registration complete")
            print(RED_chan_reg.shape)
            return RED_chan_reg[:num_images, self.measurement_crop[0]:self.measurement_crop[1],
                                self.measurement_crop[2]:self.measurement_crop[3]]
            #return RED_chan_reg[:num_images, 50:650,210:260]

    def netOD_calculation(self, background_images,control_images,images, films_per_dose, plot = False, plot_title = None):
        #print(images.shape)
        """
        This is where the netOD per irradiated film is measured. For each dosepoint
        [0.1,0.2,0.5,1,2,5,10] we have 8 dosimetry films.

        We find the weighted average pixel value for the background (black) and control film.
        Then we use these to find the netOD of each irradiated film.
        """
        #Finding PV background and control

        mean_bckg_PV = np.zeros(len(background_images))
        sigma_bckg_PV = np.zeros(len(background_images))
        mean_cont_PV = np.zeros(len(control_images))
        sigma_cont_PV = np.zeros(len(control_images))
        #PV_weight = np.zeros(len(self.background_img))
        for i in range(len(control_images)):
            if i < len(background_images):
                mean_bckg_PV[i] = np.mean(background_images[i])
                sigma_bckg_PV[i] = np.std(background_images[i])

            mean_cont_PV[i] = np.mean(control_images[i])
            sigma_cont_PV[i] = np.std(control_images[i])

        PV_bckg_weight = (1/sigma_bckg_PV)**2/(np.sum((1/sigma_bckg_PV)**2))
        PV_cont_weight = (1/sigma_cont_PV)**2/(np.sum((1/sigma_cont_PV)**2))

        PV_bckg = np.average(mean_bckg_PV, weights = PV_bckg_weight)
        PV_cont = np.average(mean_cont_PV, weights = PV_cont_weight)


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
            #finding netOD for each film
            idx = 0
            for i in range(0,len(images)):
                mean_img_PV = np.mean(images[i]) #finding mean pixel value within ROI
                sigma_img_PV[i] = np.std(images[i]) #finding std from the pixel values
                if plot:
                    #plt.errorbar(dose,mean_img_PV/PV_cont, fmt = "o", yerr = sigma_img_PV/PV_cont, c = c, markersize = 5) errorbar not necessary
                    plt.plot(dose, mean_img_PV/PV_cont, "o", c = c, markersize = 6)
                    if (i + 1) % 8 == 0 and i+1 != len(images):

                        idx += 1
                        #print(idx,i+1)
                        dose = dose_axis[idx]
                        c = color[idx]

                #print(mean_img_PV,i)
                #sigma_img_PV[idx] = np.std(images[i])
                netOD[i] = max([0,np.log10((PV_cont-PV_bckg)/(mean_img_PV-PV_bckg))])
            plt.close()
            print(sigma_cont_PV.shape,sigma_bckg_PV.shape,sigma_img_PV.shape)
            return netOD, sigma_cont_PV, sigma_bckg_PV, sigma_img_PV

        else:
            netOD = np.zeros((len(images),self.measurement_shape[0],self.measurement_shape[1]))
            sigma_img_PV = np.zeros(films_per_dose) #vet ikke hva jeg skal gjøre med denne enda
            low = 0
            high = 0
            for i in range(0,len(images)):
                img_PV = images[i]
                """
                if len(np.argwhere(img_PV == 0)[:,0]) != 0:
                    plt.imshow(img_PV)
                    plt.colorbar()
                    plt.show()
                    print(self.sorted_files[i])
                """
                #all negative values are set to 0
                diff = img_PV-PV_bckg
                diff[diff < 0] = 1e14
                """
                After registration, images will be padded with 0. this will cause
                the sum img_PV-PV_bckg < 0, making the negative sums really large,
                causes the log to become negative. These values will be clipped to 0
                """
                img_OD = np.clip(np.log10((PV_cont-PV_bckg)/(diff)),0,66e4)
                netOD[i] = img_OD
            print(netOD.shape)
            # plt.imshow(netOD[0])
            # plt.show()
            print("\n netOD calculation is complete")
            return netOD


    def calibrate(self, ROI_size, films_per_dose, channel = "RED"):
        """
        This function calibrates the dosimetry films, by splitting the images
        into background (black), control (0 Gy) and images (irradiated). Then it finds the average
        pixel value of the image of interest, and computed the netOD.
        """

        """
        image types is used to give the images their right name
        """
        image_types = ["background","control","image"]
        if self.calibration_mode:
            BLUE_chan_ROI, GREEN_chan_ROI, RED_chan_ROI, GREY_chan_ROI = self.color_channel_extraction(self.images, self.num_films, ROI_size, image_types[2])
            print(BLUE_chan_ROI.shape)

            BLUE_bckg, GREEN_bckg, RED_bckg, GREY_bckg = self.color_channel_extraction(self.background_img, self.num_background, ROI_size, image_types[0])

            BLUE_cont, GREEN_cont, RED_cont, GREY_cont = self.color_channel_extraction(self.control_img, self.num_control, ROI_size, image_types[1])

            """
            Finding the netOD for the 0 Gy films to be appended on the other netOD arrays
            """
            #defining standard deviation of control images ROI
            #4 color channels, 4 control images
            self.sigma_ctrl_PV = np.zeros((4,8))
            self.sigma_bckg_PV = np.zeros((4,4))
            self.sigma_img_PV = np.zeros((4,self.num_films + 8))
            zero_Gy_blue, self.sigma_ctrl_PV[0], self.sigma_bckg_PV[0], self.sigma_img_PV[0,:films_per_dose] = self.netOD_calculation(BLUE_bckg, BLUE_cont, BLUE_cont, films_per_dose)
            zero_Gy_green, self.sigma_ctrl_PV[1], self.sigma_bckg_PV[1],self.sigma_img_PV[1,:films_per_dose] = self.netOD_calculation(GREEN_bckg, GREEN_cont, GREEN_cont, films_per_dose)
            zero_Gy_red, self.sigma_ctrl_PV[2], self.sigma_bckg_PV[2],self.sigma_img_PV[2,:films_per_dose] = self.netOD_calculation(RED_bckg, RED_cont, RED_cont, films_per_dose)
            zero_Gy_grey, self.sigma_ctrl_PV[3], self.sigma_bckg_PV[3],self.sigma_img_PV[3,:films_per_dose] = self.netOD_calculation(GREY_bckg, GREY_cont, GREY_cont, films_per_dose)

            self.netOD = np.zeros((4, self.num_films + 8))

            tmp_OD_blue, _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+8] = self.netOD_calculation(BLUE_bckg,BLUE_cont,BLUE_chan_ROI, films_per_dose, plot = True, plot_title = "BLUE")
            tmp_OD_green, _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+8] = self.netOD_calculation(GREEN_bckg,GREEN_cont,GREEN_chan_ROI, films_per_dose,plot = True, plot_title = "GREEN")
            tmp_OD_red, _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+8] = self.netOD_calculation(RED_bckg,RED_cont,RED_chan_ROI, films_per_dose,plot = True, plot_title = "RED")
            tmp_OD_grey, _, _, self.sigma_img_PV[0,films_per_dose:self.num_films+8] = self.netOD_calculation(GREY_bckg,GREY_cont,GREY_chan_ROI, films_per_dose,plot = True, plot_title = "GRAY")

            self.netOD[0] = np.insert(tmp_OD_blue, 0, zero_Gy_blue)
            self.netOD[1] = np.insert(tmp_OD_green, 0, zero_Gy_green)
            self.netOD[2] = np.insert(tmp_OD_red, 0, zero_Gy_red)
            self.netOD[3] = np.insert(tmp_OD_grey, 0, zero_Gy_grey)

            """
            Reshaping netOD for easier plotting. We have 4 color channels, 8 doses
            0, 0.1, 0.2 etc. and 8 films per dose.
            """
            self.netOD = self.netOD.reshape((4,8,8))

            return self.netOD

        else:

            ROI = self.color_channel_extraction(self.images, self.num_films, ROI_size, image_types[2])
            plt.imshow(ROI[0])
            plt.close()
            bckg = self.color_channel_extraction(self.background_img, self.num_background, ROI_size, image_types[0])
            cont = self.color_channel_extraction(self.control_img, self.num_control, ROI_size, image_types[1])
            self.netOD = self.netOD_calculation(bckg, cont, ROI, films_per_dose)
            print("wazzuppp")
            print(self.netOD.shape)
            return self.netOD

    def netOD_split(self, doses, bandwidth, channel = "RED"):
        self.low_response_OD = []
        self.high_response_OD = []
        self.low_res_dose = []
        self.high_res_dose = []

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
                    print(iter)
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth[i]).fit(OD)
                    s = np.linspace(0,max(OD),100)
                    kde_scores = kde.score_samples(s)
                    #Henter lokal minima for red channel
                    mi = argrelextrema(kde_scores, np.less)[0]

                    if len(mi) == 0:
                        print("Bandwidth too large")
                        bandwidth[i] -= 0.0001
                    elif 1 < len(mi):
                        print("Bandwidth too small")
                        bandwidth[i] += 0.0001
                    elif len(mi) == 1:
                        print("Bandwidth of {} perfect".format(bandwidth[i]))
                        #plt.plot(s,kde_scores)
                        #plt.plot(s[mi[0]],kde_scores[mi[0]],"*")
                        #plt.show()
                    iter += 1

                low_res_idx = np.argwhere(OD < s[mi[0]])
                high_res_idx = np.argwhere(OD > s[mi[0]])


                self.low_response_OD = np.append(self.low_response_OD, OD[low_res_idx[:,0]])
                self.high_response_OD = np.append(self.high_response_OD, OD[high_res_idx[:,0]])

                self.low_res_dose = np.append(self.low_res_dose, [doses[i] for j in range(len(low_res_idx))], axis = 0)
                self.high_res_dose = np.append(self.high_res_dose, [doses[i] for j in range(len(high_res_idx))], axis = 0)

                sort_idx_low = np.argsort(self.low_res_dose)
                sort_idx_high = np.argsort(self.high_res_dose)


            self.low_response_OD = self.low_response_OD[sort_idx_low]
            self.high_response_OD = self.high_response_OD[sort_idx_high]
            self.low_res_dose = self.low_res_dose[sort_idx_low]
            self.high_res_dose = self.high_res_dose[sort_idx_high]

            print("\n netOD splitting is complete")
            return self.low_response_OD, self.high_response_OD, self.low_res_dose, self.high_res_dose, bandwidth

        else:

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

            low_img = np.argwhere(mean_netOD < s[mi[0]])[:,0]
            high_img = np.argwhere(mean_netOD > s[mi[0]])[:,0]
            #print(low_img,high_img)
            return low_img, high_img


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


    def EBT_fit(self, doses, num_fitting_params, model_type, OD = None):
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
            x0 = np.array([-1,1,1])

        if model_type == 2:
            x0 = np.array([1,1,1])

        if model_type == 3:
            x0 = np.array([1,1])

        if OD != None:
            fit = least_squares(self.RSS_func, x0, args = (OD,doses,model_type), method = "lm")
            self.fitting_param = fit.x
            self.residual = fit.fun

            return self.fitting_param, self.residual

        else:
            """
            Testing how robust the least squares is
            """
            """x0 = x0
            f = open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Fitting results LM\\fitting_params.txt", "a")
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
                self.residual_low = fit_low.cost
                self.residual_high = fit_high.cost

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
            fit_low = least_squares(self.RSS_func, x0,
                                args = (self.low_response_OD,self.low_res_dose, model_type),
                                method = "lm")
            fit_high = least_squares(self.RSS_func, x0,
                                args = (self.high_response_OD,self.high_res_dose,model_type),
                                method = "lm")
            self.fitting_param_low = fit_low.x
            self.fitting_param_high = fit_high.x
            self.residual_low = fit_low.cost
            self.residual_high = fit_high.cost
            return self.fitting_param_low, self.fitting_param_high, self.residual_low, self.residual_high
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

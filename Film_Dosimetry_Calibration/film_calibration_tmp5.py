import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


class film_calibration:
    def __init__(self, folder, test_image, calibration_mode = True):
        self.folder = folder
        self.test_image = test_image
        self.calibration_mode = calibration_mode

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

        files = np.asarray([file for file in os.listdir(self.folder)])
        """finding the index where the filename contains 001. We wish to separate
           the first second third and fourth scan, because the film might be
           affected by the scanning.
        """
        first = [i for i, s in enumerate(files) if "001" in s and "00Gy" not in s]
        second = [i for i, s in enumerate(files) if "002" in s and "00Gy" not in s]
        third = [i for i, s in enumerate(files) if "003" in s and "00Gy" not in s]
        fourth = [i for i, s in enumerate(files) if "004" in s and "00Gy" not in s]

        background = [i for i, s in enumerate(files) if "black" in s]
        background_files = files[background]

        control = [i for i, s in enumerate(files) if "00Gy" in s and "001" in s]
        control_files = files[control]

        global first_img, second_img, third_img, fourth_img

        if self.calibration_mode:
            first_img = files[first]
            self.num_films = (len(first_img))
            second_img = files[second]
            third_img = files[third]
            fourth_img = files[fourth]

        else:
            first_img = sorted(files[first],key = len)
            self.num_films = (len(first_img))
            second_img = sorted(files[second],key = len)
            third_img = sorted(files[second],key = len)
            fourth_img = sorted(files[second],key = len)


        """
        Important note:
        The filenames are sorted alphabetically. Therefore the doses are sorted
        as follows:
        [0.1, 0.2, 0.5, 10, 1, 2, 5].
        We therefore need to account for this, so that the image intensity values
        correspond to the right dose. I.E. if the film is very dark, it probably didn't get
        0.1 Gy.
        """
        sorted_files = np.append([first_img], [second_img, third_img, fourth_img])
        print(sorted_files)

        self.img_shape = (cv2.imread(os.path.join(self.folder,self.test_image))).shape


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

        self.images = np.zeros((len(sorted_files),self.img_shape[0], self.img_shape[1],self.img_shape[2]))
        if self.calibration_mode:
            self.background_img = np.zeros((len(background_files),self.img_shape[0], self.img_shape[1],self.img_shape[2]))
            self.control_img = np.zeros((len(control_files),self.img_shape[0], self.img_shape[1],self.img_shape[2]))
        else:
            """
            Midlertidig løsning. Kommer til å legge inn slik at control og background
            får sin egen mappe, som må sendes inn til film_calibration.
            """
            self.background_img = np.zeros((4,507,484,3))
            self.control_img = np.zeros((8,507,484,3))

        """
        the control images are only 507 x 484.
        """

        for i,filename in enumerate(sorted_files):
            self.images[i] = cv2.imread(os.path.join(self.folder,filename),-1)
        for i,filename in enumerate(background_files):
            self.background_img[i] = cv2.imread(os.path.join(self.folder,filename),-1)
        for i,filename in enumerate(control_files):
            self.control_img[i] = cv2.imread(os.path.join(self.folder,filename),-1)



    def color_channel_extraction(self, image, num_images, ROI_size):

        """
        This function simply extract a centrally placed square ROI from the images acquired
        in the function above. The specific color channels are then extracted, so
        the intensity values can be used to find net Optical Density (OD).
        """
        rows = self.img_shape[0]
        columns = self.img_shape[1]
        self.ROI_pixel_size = ROI_size

        if self.calibration_mode:
        #Splitting each channel for calibration
            BLUE_chan_ROI = image[:num_images, rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size,0]
            GREEN_chan_ROI = image[:num_images,rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size,1]
            RED_chan_ROI = image[:num_images,rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size,2]
            GREY_chan_ROI = np.mean(image[:num_images,rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                    columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size],axis = 3)

            return BLUE_chan_ROI, GREEN_chan_ROI, RED_chan_ROI, GREY_chan_ROI

        else:
            RED_chan_ROI = image[:num_images,rows//2-self.ROI_pixel_size:rows//2+self.ROI_pixel_size, \
                                columns//2 - self.ROI_pixel_size:columns//2 + self.ROI_pixel_size,2]
            return RED_chan_ROI

    def netOD_calculation(self, background_images,control_images,images, films_per_dose):
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
        netOD = np.zeros(len(images))
        sigma_img_PV = np.zeros(films_per_dose) #vet ikke hva jeg skal gjøre med denne enda


        #finding netOD for each film
        for i in range(0,len(images)):
            mean_img_PV = np.mean(images[i])
            #sigma_img_PV[idx] = np.std(images[i])
            netOD[i] = max([0,np.log10((PV_cont-PV_bckg)/(mean_img_PV-PV_bckg))])

        return netOD


    def calibrate(self, ROI_size, films_per_dose):
        """
        This function calibrates the dosimetry films, by splitting the images
        into background (black), control (0 Gy) and images (irradiated). Then it finds the average
        pixel value of the image of interest, and computed the netOD.
        """

        if self.calibration_mode:
            BLUE_chan_ROI, GREEN_chan_ROI, RED_chan_ROI, GREY_chan_ROI = self.color_channel_extraction(self.images, self.num_films, ROI_size)

            BLUE_bckg, GREEN_bckg, RED_bckg, GREY_bckg = self.color_channel_extraction(self.background_img, len(self.background_img), ROI_size)

            BLUE_cont, GREEN_cont, RED_cont, GREY_cont = self.color_channel_extraction(self.control_img, len(self.control_img), ROI_size)

            """
            Finding the netOD for the 0 Gy films to be appended on the other netOD arrays
            """
            zero_Gy_blue = self.netOD_calculation(BLUE_bckg, BLUE_cont, BLUE_cont, films_per_dose)
            zero_Gy_green = self.netOD_calculation(GREEN_bckg, GREEN_cont, GREEN_cont, films_per_dose)
            zero_Gy_red = self.netOD_calculation(RED_bckg, RED_cont, RED_cont, films_per_dose)
            zero_Gy_grey = self.netOD_calculation(GREY_bckg, GREY_cont, GREY_cont, films_per_dose)


            self.netOD = np.zeros((4, self.num_films + len(zero_Gy_blue)))

            self.netOD[0] = np.insert(self.netOD_calculation(BLUE_bckg,BLUE_cont,BLUE_chan_ROI, films_per_dose), 0, zero_Gy_blue)
            self.netOD[1] = np.insert(self.netOD_calculation(GREEN_bckg,GREEN_cont,GREEN_chan_ROI, films_per_dose), 0, zero_Gy_green)
            self.netOD[2] = np.insert(self.netOD_calculation(RED_bckg,RED_cont,RED_chan_ROI, films_per_dose), 0, zero_Gy_red)
            self.netOD[3] = np.insert(self.netOD_calculation(GREY_bckg,GREY_cont,GREY_chan_ROI, films_per_dose), 0, zero_Gy_grey)

            """
            Reshaping netOD for easier plotting. We have 4 color channels, 8 doses
            0, 0.1, 0.2 etc. and 8 films per dose.
            """
            self.netOD = self.netOD.reshape((4,8,8))

            return self.netOD

        else:

            RED_chan_ROI = self.color_channel_extraction(self.images, self.num_films, ROI_size)
            RED_bckg = self.color_channel_extraction(self.background_img, len(self.background_img), ROI_size)
            RED_cont = self.color_channel_extraction(self.control_img, len(self.control_img), ROI_size)
            self.netOD = self.netOD_calculation(RED_bckg, RED_cont, RED_chan_ROI, films_per_dose)
            self.netOD = self.netOD.reshape((1,-1)) #reshaping to keep the same splitting for loop

            return self.netOD

    def netOD_split(self, doses, bandwidth):
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
                """
                Splitting netOD into high and low respons
                """

                OD = self.netOD[2,i,:].reshape(-1,1)

                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth[i]).fit(OD)
                s = np.linspace(0,max(OD),100)
                kde_scores = kde.score_samples(s)
                #Henter lokal minima for red channel
                mi = argrelextrema(kde_scores, np.less)[0]
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

            return self.low_response_OD, self.high_response_OD, self.low_res_dose, self.high_res_dose

        else:
            OD = self.netOD.reshape(-1,1)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(OD)
            s = np.linspace(0,max(OD),100)
            kde_scores = kde.score_samples(s)
            #Henter lokal minima for red channel
            mi = argrelextrema(kde_scores, np.less)[0]
            print(mi)
            low_res_idx = np.argwhere(OD < s[mi[0]])
            high_res_idx = np.argwhere(OD > s[mi[0]])
            self.low_response_OD = np.append(self.low_response_OD, OD[low_res_idx[:,0]])
            self.high_response_OD = np.append(self.high_response_OD, OD[high_res_idx[:,0]])

            return self.low_response_OD, self.high_response_OD

    def EBT_model(self,netOD,a,b,n):
        """
        This is the model we wish to fit to the netOD.
        a, b, n is the fitting parameters
        """
        return a*netOD + b*netOD**n

    def EBT_fit(self, doses, num_fitting_params):
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


        low_popt, low_pcov = curve_fit(self.EBT_model, self.low_response_OD, self.low_res_dose)
        high_popt, high_pcov = curve_fit(self.EBT_model, self.high_response_OD, self.high_res_dose)

        self.fitting_param_low = low_popt
        self.fitting_param_high = high_popt


        #print(np.sqrt(np.diag(pcov)))

        return self.fitting_param_low, self.fitting_param_high



if __name__ == "__main__":
    folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"
    background_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Background"
    test_image = "EBT3_Calib_310821_Xray220kV_00Gy1_001.tif"
    film_calib = film_calibration(folder, test_image)

    images = film_calib.image_acquisition()

    films_per_dose = 8
    ROI_size = 2 #mm

    film_calib.calibrate(ROI_size, films_per_dose)

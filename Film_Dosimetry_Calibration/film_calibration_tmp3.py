import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.optimize import curve_fit

class film_calibration:
    def __init__(self, folder, test_image):
        self.folder = folder
        self.test_image = test_image

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
        first_img = files[first]
        self.num_films = (len(first_img))

        second = [i for i, s in enumerate(files) if "002" in s and "00Gy" not in s ]
        second_img = files[second]

        third = [i for i, s in enumerate(files) if "003" in s and "00Gy" not in s]
        third_img = files[third]

        fourth = [i for i, s in enumerate(files) if "004" in s and "00Gy" not in s]
        fourth_img = files[fourth]

        background = [i for i, s in enumerate(files) if "black" in s and "00Gy" not in s]
        background_files = files[background]

        control = [i for i, s in enumerate(files) if "00Gy" in s and "001" in s]
        control_files = files[control]


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


        self.img_shape = (cv2.imread(os.path.join(self.folder,self.test_image))).shape


        noise_img = np.zeros((4,self.img_shape[0],self.img_shape[1],self.img_shape[2]))




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
        self.background_img = np.zeros((len(background_files),self.img_shape[0], self.img_shape[1],self.img_shape[2]))
        self.control_img = np.zeros((len(control_files),self.img_shape[0], self.img_shape[1],self.img_shape[2]))


        for i,filename in enumerate(sorted_files):
            self.images[i] = cv2.imread(os.path.join(self.folder,filename),-1)
        for i,filename in enumerate(background_files):
            self.background_img[i] = cv2.imread(os.path.join(self.folder,filename),-1)
        for i,filename in enumerate(control_files):
            self.control_img[i] = cv2.imread(os.path.join(self.folder,filename),-1)



    def color_channel_extraction(self, image, num_images):

        """
        This function simply extract a centrally placed square ROI from the images acquired
        in the function above. The specific color channels are then extracted, so
        the intensity values can be used to find net Optical Density (OD).
        """
        rows = self.img_shape[0]
        columns = self.img_shape[1]


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
            netOD[i] = np.log10((PV_cont-PV_bckg)/(mean_img_PV-PV_bckg))

        return netOD


    def calibrate(self, ROI_size, films_per_dose):
        """
        This function calibrates the dosimetry films, by splitting the images
        into background (black), control (0 Gy) and images (irradiated). Then it finds the average
        pixel value of the image of interest, and computed the netOD.
        """
        #ROI to number of pixels
        self.ROI_pixel_size = round(ROI_size * 3.7795275591) #3.8 is mm to pixel conversion factor
        rows = self.img_shape[0]
        columns = self.img_shape[1]

        BLUE_chan_ROI, GREEN_chan_ROI, RED_chan_ROI, GREY_chan_ROI = self.color_channel_extraction(self.images, self.num_films)

        BLUE_bckg, GREEN_bckg, RED_bckg, GREY_bckg = self.color_channel_extraction(self.background_img, len(self.background_img))

        BLUE_cont, GREEN_cont, RED_cont, GREY_cont = self.color_channel_extraction(self.control_img, len(self.control_img))


        self.netOD = np.zeros((4, self.num_films))

        self.netOD[0] = self.netOD_calculation(BLUE_bckg,BLUE_cont,BLUE_chan_ROI, films_per_dose)
        self.netOD[1] = self.netOD_calculation(GREEN_bckg,GREEN_cont,GREEN_chan_ROI, films_per_dose)
        self.netOD[2] = self.netOD_calculation(RED_bckg,RED_cont,RED_chan_ROI, films_per_dose)
        self.netOD[3] = self.netOD_calculation(GREY_bckg,GREY_cont,GREY_chan_ROI, films_per_dose)


        #reshape so that every dose has their 8 netOD from the 8 films
        self.netOD = self.netOD.reshape((4,7,8))




        """
        netOD has shape 4, 7, 56
        4 channels and 56 images. Every 8 images is a new dose. Therefore we reshape the netOD to be 7,8

        remember to get the netOD for control images
        """

        return self.netOD


    def EBT_model(self,doses,a,b,n):
        """
        This is the model we wish to fit to the netOD.
        a, b, n is the fitting parameters
        """
        return a + b*doses**n

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

        self.fitting_param = np.zeros((self.netOD.shape[0],num_fitting_params))
        self.std_err = np.zeros(self.netOD.shape[0])

        """
        Here we make the stacked dose array, in our case we have 8 films per dose.
        """
        dose_axis_octo = []
        for i in (doses):   #looping over doses
        #Adding dose i 8 times to dose_axis_octo, so the curve fit is correct.
            for j in range(self.netOD.shape[2]):
                dose_axis_octo = np.append(dose_axis_octo, i)

        #For each color, we see which n in the EBT model gives the best fit.
        for i in range(self.netOD.shape[0]):
            popt, pcov = curve_fit(self.EBT_model, dose_axis_octo , np.ravel(self.netOD[i]))
            self.fitting_param[i] = popt
            self.std_err[i] = np.sqrt(np.diag(pcov))

        return self.fitting_param, self.std_err




if __name__ == "__main__":
    folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Calibration"
    background_folder = "C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\310821\\Background"
    test_image = "EBT3_Calib_310821_Xray220kV_00Gy1_001.tif"
    film_calib = film_calibration(folder, test_image)

    images = film_calib.image_acquisition()

    films_per_dose = 8
    ROI_size = 2 #mm

    film_calib.calibrate(ROI_size, films_per_dose)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from scipy.optimize import curve_fit
from patsy import dmatrices
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import cv2
from itertools import repeat
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import t,chi2
from datetime import date


def mean_dose(lower_lim, upper_lim, doses):
    """
    Assumes doses with shape (m,n)
    """
    idx = np.argwhere(np.logical_and(lower_lim < doses, doses < upper_lim))
    mean = 0
    std = 0
    for i in range(doses.shape[0]):
        idx_tmp = idx[np.argwhere(idx[:,0] == 0)[:,0]]
        mean += np.mean(doses[i,idx_tmp[:,1]])
        std += np.std(doses[i,idx_tmp[:,1]])
    mean /= doses.shape[0]
    std /= doses.shape[0]
    return mean, std


def mean_survival(X, SC, rounding, method):
    if len(X.shape) < 2:
        dose_axis = X
    else:
        dose_axis = X[:,1]
    """Rounding of to neares 0.5 demands that we round of to one less decimal first"""
    rounding -= 1
    # dose_categories = np.unique(np.round(np.unique(X[:,1]),rounding))
    """Allocating each dose into the closest category"""
    allocated_doses = np.round(dose_axis*2, rounding)/2
    dose_categories = np.unique(allocated_doses)
    # print(dose_categories)
    mean_SC = []
    SC_std = []
    dose_std = []
    if len(dose_categories)  < 2:
        mean_SC.append(np.mean(SC))
        SC_std.append(np.std(SC)/np.sqrt(len(SC)))
        dose_std.append(np.std(dose_axis))
    else:
        for i in range(0,len(dose_categories)):

            #idx = np.argwhere(np.logical_and(dose_categories[i] <= dose_axis, dose_axis < dose_categories[i+1]))
            idx = np.argwhere(dose_categories[i] == allocated_doses)
            print("Number of {:.2f} Gy {}".format(dose_categories[i],len(idx)))
            #print(len(idx), dose_categories[i])
            #plt.plot(dose_categories[i], 20-count,"*")
            if len(idx) != 0:
                dose_std.append(np.std(dose_axis[idx[:,0]]))
                if dose_categories[i] == 4.5:
                    print(idx.shape)
                    print(idx)
                    print(dose_axis[idx[:,0]])
                #print(dose_categories[i], dose_categories[i+1])
                mean_SC.append(np.mean(SC[idx[:,0]]))
                SC_std.append(np.std(SC[idx[:,0]])/np.sqrt(len(SC[idx[:,0]])))
    print(np.shape(dose_std))
    return np.array(mean_SC),np.array(SC_std), dose_categories, dose_std

def poisson_regression(respond_variables, X , num_regressors, plot_title, save_path, legend, SC_lengths, kernel_size, save_results = False):
    #making design matrix. The intercept is 1
    #the survival to ctrl
    """
    This function takes dose and survival data, and uses Poisson regression to estimate
    average survival for given dose.
    We need to make a design matrix with the different parameters which are
    dose dose^2 and the g factor (area fraction)
    Design matrix looks like this

    [1,x00,x01,x02,x03
     1,x10,x11,x12,x13
     ...
     ...
     1,xn0,xn1,xn2,xn3]
    """
    #we interpolate the first parameter to fit a line to the poisson regression model
    #X[:,1] is doses

    colors = ["b","g","r","grey","m","y","black","saddlebrown"]

    #X_train, X_test, y_train, y_test = train_test_split(X, respond_variables,train_size = 0.8)
    # poisson_training_results = sm.GLM(respond_variables, X, family=sm.families.Poisson()).fit()
    model = sm.GLM(respond_variables, X, family=sm.families.Poisson())
    poisson_training_results = model.fit(full_output = True)

    pvalue = 1-chi2.cdf(poisson_training_results.pearson_chi2, X.shape[0] - num_regressors - 1)


    print("AIC for {} regressors".format(num_regressors))
    print(poisson_training_results.aic)
    print("Log likelihood")
    print(poisson_training_results.llf)







    if save_results == True:
        # f = open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\regression results\\GLM_results_OPEN&STRIPES&DOTS.txt", "a")
        f = open("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\2D analysis\\Poisson\\regression results\\GLM_results_test.txt", "a")

        f.write("\nnum regressors\t\t\tkernel size\t\t\tdate\t\t\tAIC\t\t\tGOF p-value\t# datapoints")
        f.write("\n{}\t\t\t\t{}\t\t\t\t{}\t\t{:.5f}\t\t{:.3f}\t\t{}".format(num_regressors,  kernel_size, date.today(), poisson_training_results.aic, pvalue, len(respond_variables)))
        f.close()

    summary = poisson_training_results.summary()
    summary2 = summary.as_csv()

    print(summary2)

    fitting_params = poisson_training_results.params
    if num_regressors == 5:
        fit_label = r"Fit: $ {:.3f} {:+.3f}D {:+.3f} D^2 {:+.3f} g {:+.3f} (1-g) {:+.3f} l$".format(fitting_params[0], fitting_params[1],fitting_params[2], fitting_params[3],  fitting_params[4], fitting_params[5])
    elif num_regressors == 4:
        fit_label = r"Fit: $ {:.3f} {:+.3f}D {:+.3f} D^2 {:+.3f} g {:+.3f} (1-g)$".format(fitting_params[0], fitting_params[1],fitting_params[2], fitting_params[3],  fitting_params[4])
    elif num_regressors == 3:
        fit_label = r"Fit: $ {:.3f} {:+.3f}D {:+.3f} D^2 {:+.3f} l$".format(fitting_params[0], fitting_params[1],fitting_params[2], fitting_params[3])
    elif num_regressors == 2:
        fit_label = r"Fit: $ {:+.3f} {:+.3f}D {:+.3f} D^2$".format(fitting_params[0], fitting_params[1],fitting_params[2])

    if save_results == True:
        beginningtex = """\\documentclass{report}
        \\usepackage{booktabs}
        \\begin{document}"""
        endtex = "\end{document}"

        f = open(save_path, 'w')
        f.write(beginningtex)
        f.write(poisson_training_results.summary().as_latex())
        f.write(endtex)
        f.close()
        #df.to_csv("C:\\Users\\jacob\\OneDrive\\Documents\\Skole\\Master\\data\\Survival Analysis Data\\GLM_results_39mm_OPEN.csv")
    #run test data through the model.
    poisson_predictions = poisson_training_results.get_prediction(X)

    #summary_frame() returns a pandas DataFrame
    predictions_summary_frame = poisson_predictions.summary_frame()



    """
    plot true vs predicted
    """
    predicted_counts = predictions_summary_frame['mean']

    mean_predicted_SC = mean_survival(X, predicted_counts, 1, None)
    print("predicted mean survival")
    #print(predicted_SC)
    #we sort the doses to get correct axis
    #dose_axis, correct_counts = zip(*sorted(zip(X_test[:,0], y_test)))
    #_,predicted_counts = zip(*sorted(zip(X_test[:,0], predicted_counts)))

    print(len(respond_variables))

    print(np.sum(SC_lengths))

    dose_axis = X[:,1]


    """2D plotting"""
    fig,ax = plt.subplots(figsize = (10,8))
    #ax.set_title(plot_title)
    ax.set_xlabel("Dose [Gy]", fontsize = 12)
    ax.set_ylabel("SC", fontsize = 12)


    # ax.plot(X[SC_lengths[3]:SC_lengths[4] - 1,3], respond_variables[SC_lengths[3]:SC_lengths[4] - 1], 'o', label='Observed', color = colors[0], markersize = 3)
    # ax.plot(X[SC_lengths[3]:SC_lengths[4] - 1,3], predicted_counts[SC_lengths[3]:SC_lengths[4] - 1], "^", label = "Predicted", color = colors[0], markersize = 4)
    # ax.legend()
    # plt.show()
    for i in range(len(SC_lengths)-1):
        print(SC_lengths[i], SC_lengths[i+1])
        ax.plot(dose_axis[SC_lengths[i]:SC_lengths[i+1] - 1], respond_variables[SC_lengths[i]:SC_lengths[i+1] - 1], 'o', label='Observed' + legend[i], color = colors[i], markersize = 3)
        ax.plot(dose_axis[SC_lengths[i]:SC_lengths[i+1] - 1], predicted_counts[SC_lengths[i]:SC_lengths[i+1] - 1], "^", label = "Predicted" + legend[i], color = colors[i], markersize = 4)
    ax.legend(fontsize = 14, markerscale = 2)

    """
    3D plotting
    """

    # from mpl_toolkits import mplot3d
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #Z = respond_variables[SC_lengths[3]:SC_lengths[4]-1]

    #ax.scatter3D(X[SC_lengths[3]:SC_lengths[4]-1,1], X[SC_lengths[3]:SC_lengths[4]-1,3], Z, c=Z, cmap='Greens');
    #plt.show()

    # dose_axis = X[:,1]
    # plt.plot(dose_axis, predicted_counts, "bo", label=fit_label)
    # plt.plot(dose_axis, respond_variables, 'ro', label='Correct survival')
    # plt.title(plot_title)
    # plt.xlabel("Dose [Gy]")
    # plt.ylabel("SC")
    # plt.legend()
    #plt.show()
    return poisson_training_results, mean_predicted_SC, summary2

def data_stacking_2(grid, *args):
    """
    We stack all survival data with dose except 0 Gy ctrl, this will be added manually
    if needed
    """
    if grid == False:
        survival_2, survival_5, dose2, dose5 = args
        SC = np.concatenate((np.ravel(survival_2),
                             np.ravel(survival_5)))
        doses = np.array([dose2,dose5])
    if grid == True:
        survival_2, survival_5, survival_10, dose2, dose5, dose10 = args
        SC = np.concatenate((np.ravel(survival_2),
                             np.ravel(survival_5), np.ravel(survival_10)))
        doses = np.array([dose2,dose5,dose10])
    tot_dose_axis = np.ravel(np.repeat(doses, survival_2.shape[0]*survival_2.shape[1], axis = 0))
    return SC, tot_dose_axis

    return mean_dose, std_dose

def dose_fit_error(OD, dOD,dparam,param):
    da,db,dn = np.sqrt(dparam)
    a,b,n = param

    print(da,db,dn,a,b,n)
    return np.sqrt(OD**2 * da**2 + (OD**n)**2 * db**2 + (a + b*n * OD**(n-1))**2 * dOD**2 + \
                  (b*np.log(OD)*OD**n)**2 * dn**2)
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)

import numpy as np
from scipy.optimize import curve_fit

def logLQ(d,alpha, beta):
    return -(alpha*d + beta*d**2)

def fit(model, x, y):
    popt, pcov = curve_fit(model, x, y)
    return popt

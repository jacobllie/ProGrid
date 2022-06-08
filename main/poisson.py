import numpy as np
from math import factorial

def poisson(lambda_, k):
    return lambda_**k * np.exp(-lambda_)/[factorial(i) for i in (k)]

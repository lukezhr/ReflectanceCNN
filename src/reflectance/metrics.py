import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from .helper import params_ranges

# Metric functions
# ________________________________________________________________________________
# For a group of data
def mse(predicted, true):
    return mean_squared_error(true, predicted)

def rmse(predictions, targets):
    """ Calculate root mean squared error. """
    return sqrt(mean_squared_error(predictions, targets))

def adjusted_r2(r2, n, p):
    """ Calculate adjusted R-squared. """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def mase(predictions, targets):
    """ Calculate mean absolute scaled error. """
    return mean_absolute_error(predictions, targets) / mean_absolute_error(targets[1:], targets[:-1])

# For a single parameter
def se(predicted, true):
    """ Calculate squared error. """
    return (predicted - true) ** 2

def ae(predicted, true):
    """ Calculate absolute error. """
    return np.abs(predicted - true)

def pe(predicted, true):
    """ Calculate percentage error. """
    return np.abs((predicted - true) / true) * 100 if true != 0 else np.abs((predicted - true) / (true + 1e-6)) * 100

def ne(predicted, true, max_params, min_params, parameter_name):
    """ Calculate normalized error. """
    ranges = params_ranges(max_params, min_params)
    return ae(predicted, true) / ranges[parameter_name]
# ________________________________________________________________________________
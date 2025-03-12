"""
Physics-based components for thin film characterization.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import List, Tuple, Optional
from reflectance.cnn_structure import ReflectanceCNN
import time

# Local physics imports
from reflectance.ThinFilmClasses import ThinFilmLayer, ThinFilmLayerTL, ThinFilmSystem
from reflectance.TaucLorentz import TL_nk

def data_prep(wavelengths, reflectances, n_data=1000, min_wavelength=440, max_wavelength=800):
    """
    Interpolate reflectance data to a specified number of points between given wavelength limits.
    This version supports both single reflectance vectors (1D array) and multiple reflectance columns (2D array).

    Parameters:
    wavelengths (array-like): The original wavelengths array.
    reflectances (array-like): The reflectance values corresponding to the wavelengths.
                              This can be a 1D array for single reflectance or 2D array for multiple sets.
    min_wavelength (int): The minimum wavelength for interpolation.
    max_wavelength (int): The maximum wavelength for interpolation.
    n_data (int): The number of points to interpolate.

    Returns:
    np.array: New wavelengths array.
    np.array: Interpolated reflectance values, matching the input reflectance dimensionality (1D or 2D).
    """
    # Generate the new wavelengths array
    print("wavelegnths shape: ", wavelengths.shape)
    new_wavelengths = np.linspace(min_wavelength, max_wavelength, n_data)
    print("new_wavelengths shape: ", new_wavelengths.shape)

    # Check if reflectances is a 1D array and reshape it to 2D if necessary
    if reflectances.ndim == 1:
        reflectances = reflectances.reshape(-1, 1)

    # Initialize an array to store the interpolated reflectances
    new_reflectances = np.zeros((n_data, reflectances.shape[1]))
    
    print("wavelegnths shape: ", wavelengths.shape)
    print("reflectances shape: ", reflectances.shape)
    print("new_wavelengths shape: ", new_wavelengths.shape)
    print("new_reflectances shape: ", new_reflectances.shape)

    # Perform the linear interpolation for each column of reflectances
    for i in range(reflectances.shape[1]):
        new_reflectances[:, i] = np.interp(new_wavelengths, wavelengths, reflectances[:, i])
    print("new_reflectances shape: ", new_reflectances.shape)

    # If the original reflectances were a 1D array, reshape the result back to 1D
    if new_reflectances.shape[1] == 1:
        new_reflectances = new_reflectances.ravel()
        print("new_reflectances shape: ", new_reflectances.shape)

    return new_wavelengths, new_reflectances



# Denormalize the parameters
def denormalize(samples, max_params, min_params):
    epsilon = 1e-8
    return samples * (max_params - min_params + epsilon) + min_params


def predict(data_reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data, thickness_adjustment=0, plot=False):
    energy = 1239.84193 / data_wavelength
    new_wavelengths, reflectance = data_prep(
        data_wavelength, data_reflectance, n_data=n_data)
    reflectance = torch.from_numpy(reflectance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reflectance = reflectance.to(device)

    # Add two dimensions to the reflectance tensor to match the input shape of the model (batch_size, channels, length)
    reflectance = reflectance.unsqueeze(0).unsqueeze(0)

    # Predict the parameters
    model.double()  # Convert the model to double precision
    model.eval()
    start = time.time()
    with torch.no_grad():
        params = model(reflectance).cpu().numpy()[0]
        params_before_denormalization = params.copy()
        params = denormalize(params, max_params, min_params).tolist()
        # # Adjust the thickness shift (for model_ultimate_0)
        params[-1] = params[-1] + thickness_adjustment
        # Take the absolute value of the parameters, in case of negative values
        params = [abs(param) for param in params]
        print(params)
    end = time.time()
    print(f"Prediction time: {end - start} seconds")

    # Calculate n, k
    _, _, n, k = TL_nk(data_wavelength, params)
    n = np.squeeze(n)
    k = np.squeeze(k)

    multilayer.layers[1].param = params[:-1]
    multilayer.layers[1].thickness = params[-1]
    multilayer.layers[1].n = n
    multilayer.layers[1].k = k
    R_cal, _, _ = multilayer.calculate_RTA(data_wavelength)

    if plot:
        # Plot n, k
        plt.figure()
        plt.plot(energy, n, label="n", c="b")
        plt.title("n")
        plt.xlabel("Energy (eV)")
        plt.ylabel("n")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(energy, k, label="k", c="b")
        plt.title("k")
        plt.xlabel("Energy (eV)")
        plt.ylabel("k")
        plt.legend()
        plt.show()

        # Plot reflectance
        plt.figure()
        plt.plot(energy, data_reflectance,
                 label="Data Reflectance", alpha=0.5)
        plt.plot(energy, R_cal, label="R_cal", color="r")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Reflectance")
        plt.title("Data Reflectance and R_cal")
        plt.legend()
        plt.show()

    return params, params_before_denormalization, n, k, R_cal


def process_data_from_film_measure(data_path, left, right, uncertainty_threshold=None):
    '''Process data from film measurement.'''
    data, left, right = process_data(
        data_path, left=left, right=right, uncertainty_threshold=uncertainty_threshold)
    data_wavelength = data['wavelength'].to_numpy()
    data_reflectance = data['reflectance'].to_numpy()
    return data_reflectance, data_wavelength


def process_data_across_stripe(data_path, wavelength_path, data_index=None):
    '''Process data for the across stripe test.'''
    data = np.loadtxt(data_path, delimiter=',')
    data = np.clip(data, 0, 1) # Clip the reflectance values to [0, 1]
    data_wavelength = np.loadtxt(wavelength_path, delimiter=',')
    if data_index is not None:
        data_reflectance = data[:, data_index]

    return data_reflectance, data_wavelength


# Construct multilayer
def construct_bi2o3_multilayer(wavelength):
    left, right = 440, 800
    n_points = 10
    params = [1, 1, 1, 1, 1]  # placehoding values
    thickness = 400  # placeholding value
    air = ThinFilmLayer("air", 1, 0, left, right)
    layer1 = ThinFilmLayerTL(thickness, params, wavelength)
    layer2 = ThinFilmLayer("sio2", 200, n_points, left, right)
    substrate = ThinFilmLayer("c-Si", 1, 0, left, right)
    multilayer = ThinFilmSystem([air, layer1, layer2, substrate])
    return multilayer


def optimize_TL(params, multilayer, data_wavelength, data_reflectance, layer_index=1, ftol=10e-5):
    
    # # Choose a part of the data for optimization
    # def data_prep_optimize_TL(reflectance, wavelength, left_opt=450, right_opt=800):
    #     indices = np.where((wavelength >= left_opt) & (wavelength <= right_opt))
    #     data_wavelength_opt = wavelength[indices]
    #     data_reflectance_opt = reflectance[indices]
    #     return data_reflectance_opt, data_wavelength_opt

    # Define the cost function
    def cost_function(params):
        multilayer.layers[layer_index].update(
            params[:-1], params[-1], data_wavelength)
        R, _, _ = multilayer.calculate_RTA(data_wavelength)
        # data_reflectance_cropped, _ = data_prep_optimize_TL(data_reflectance, data_wavelength)
        # R_cropped, _ = data_prep_optimize_TL(R, data_wavelength)
        # return R_cropped - data_reflectance_cropped
        return R - data_reflectance

    # Set the lower and upper bounds for the parameters
    lower_bounds = [1e-6, 1e-6, 0,
                    0, 1, 50]  # A, E0, G, Eg, e_inf, thickness
    upper_bounds = [100, 10, 14.0932,
                    4, 10, 150]

    # optimize using least_squares
    start = time.time()
    print('Optimizing...')
    # result = least_squares(cost_function, params, method='lm', ftol=ftol)
    result = least_squares(cost_function, params, bounds=(
        lower_bounds, upper_bounds), ftol=ftol)
    end = time.time()
    print(f"Optimization time: {end - start} seconds")
    params_opt = result.x
    print(f"Optimized params: {params_opt}")
    multilayer.layers[layer_index].update(
        params_opt[:-1], params_opt[-1], data_wavelength)
    R_cal_opt, _, _ = multilayer.calculate_RTA(data_wavelength)

    return R_cal_opt, params_opt, multilayer

def optimization(params, multilayer, data_wavelength, data_reflectance):
    """ 
    Perform optimization using TL parameters and update multilayer structure. 
    
    Args:
        params (np.array): Initial parameters for optimization.
        multilayer (ThinFilmSystem): Multilayer system to optimize.
        data_wavelength (np.array): Wavelength data for optimization.
        data_reflectance (np.array): Reflectance data for optimization.
    
    Returns:
        tuple: (R_cal_opt, params_opt, n_opt, k_opt, updated_multilayer)
    """
    R_cal_opt, params_opt, multilayer = optimize_TL(params, multilayer, data_wavelength, data_reflectance)
    _, _, n_opt, k_opt = TL_nk(data_wavelength, params_opt)
    return R_cal_opt, params_opt, n_opt, k_opt, multilayer
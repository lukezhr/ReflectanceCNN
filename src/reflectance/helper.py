import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from .model_eval_pkg import construct_bi2o3_multilayer

# Helper functions
# ________________________________________________________________________________
def load_parameters(file_path):
    """ Load parameters from CSV file. """
    data = pd.read_csv(file_path, delimiter=",", header=None)
    print(f"Data loaded from {file_path} contains {data.shape[0]} rows.")
    print('data.shape:', data.shape)
    return data.values

def load_reflectance_and_wavelength(file_path):
    """ Load reflectance and wavelength data from CSV file. """
    data = np.loadtxt(file_path, delimiter=",")
    return data

def load_max_min_params(model_name):
    """ Load max and min parameters from CSV files. """
    max_params = np.loadtxt(f"{model_name}/max_params.csv", delimiter=",")
    min_params = np.loadtxt(f"{model_name}/min_params.csv", delimiter=",")
    return max_params, min_params

def normalize(samples, max_params, min_params):
    epsilon = 1e-8
    return (samples - min_params) / (max_params - min_params + epsilon)

def params_ranges(max_params, min_params):
    """ Get the range of parameters. """
    A_range = abs(max_params[0] - min_params[0])
    E0_range = abs(max_params[1] - min_params[1])
    G_range = abs(max_params[2] - min_params[2])
    Eg_range = abs(max_params[3] - min_params[3])
    e_inf_range = abs(max_params[4] - min_params[4])
    thickness_range = abs(max_params[5] - min_params[5])
    return {'A': A_range, 'E0': E0_range, 'G': G_range, 'Eg': Eg_range, 'e_inf': e_inf_range, 'thickness': thickness_range}

def load_data_model_params(data_path, model_name, mode='single', chosen_index=0, simulate=False, simulated_params_path='parameters.csv', data_wavelength_path='wavelength_cropped.csv', normalized_params_path='Y_test_d1000.csv'):
    """ Load data, model, and parameters. """
    # Load the model and construct multilayer system
    model = torch.load(f'{model_name}.pth', map_location=torch.device('cpu'))

    # Load parameters
    max_params, min_params = load_max_min_params(model_name)

    data_reflectance = load_reflectance_and_wavelength(data_path)
    simulated_params = None
    data_wavelength = None
    normalized_params = None

    if mode == 'single':
        data_reflectance = data_reflectance[chosen_index, :]
        print("data_reflectance.shape (single):", data_reflectance.shape)
        if simulate and simulated_params_path is not None and normalized_params_path is not None:
            data_wavelength = np.linspace(440, 800, data_reflectance.shape[0])
            multilayer = construct_bi2o3_multilayer(data_wavelength)
            simulated_params = load_parameters(simulated_params_path)
            simulated_params = simulated_params[chosen_index, :]
            normalized_params = load_parameters(normalized_params_path)
            normalized_params = normalized_params[chosen_index, :]
        elif simulate == False and data_wavelength_path is not None:
            data_wavelength = load_reflectance_and_wavelength(data_wavelength_path)
            print("data_reflectance.shape (single):", data_reflectance.shape)
            multilayer = construct_bi2o3_multilayer(data_wavelength)
    elif mode == 'dataset':
        if simulate and simulated_params_path is not None and normalized_params_path is not None:
            print("data_reflectance.shape (dataset):", data_reflectance.shape)
            data_wavelength = np.linspace(440, 800, data_reflectance.shape[1])
            multilayer = construct_bi2o3_multilayer(data_wavelength)
            simulated_params = load_parameters(simulated_params_path)
            normalized_params = load_parameters(normalized_params_path)
        elif simulate == False and data_wavelength_path is not None:
            data_wavelength = load_reflectance_and_wavelength(data_wavelength_path)
            print("data_reflectance.shape (dataset):", data_reflectance.shape)
            multilayer = construct_bi2o3_multilayer(data_wavelength)
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'single' or 'dataset'.")

    return data_reflectance, data_wavelength, simulated_params, model, multilayer, max_params, min_params, normalized_params
# ________________________________________________________________________________
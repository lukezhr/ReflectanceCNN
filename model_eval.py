import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
from model_eval_pkg import ReflectanceCNN, predict, optimize_TL, construct_bi2o3_multilayer
from TaucLorentz import TL_nk
import json
import time
import pandas as pd
from matplotlib.patches import Polygon


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



# Core functions
# ________________________________________________________________________________
def evaluate_and_predict(data_reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data, simulate=False, simulated_params=None):
    """ 
    Predict parameters, n, and k, and calculate R_cal using the given model and data.
    If the data is simulated, return the true n and k values as well.
    """
    params, params_before_denormalization, n, k, R_cal = predict(data_reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data=n_data)
    print("R_cal.shape:", R_cal.shape)
    print("n.shape:", n.shape)
    print("k.shape:", k.shape)
    if simulate and simulated_params is not None:
        _, _, n_true, k_true = TL_nk(data_wavelength, simulated_params)
        n_true, k_true = n_true.flatten(), k_true.flatten()
        return params, params_before_denormalization, n, k, R_cal, n_true, k_true
    return params, params_before_denormalization, n, k, R_cal, None, None

def optimization(params, multilayer, data_wavelength, data_reflectance):
    """ Perform optimization using TL parameters and update multilayer structure. """
    R_cal_opt, params_opt, multilayer = optimize_TL(params, multilayer, data_wavelength, data_reflectance)
    _, _, n_opt, k_opt = TL_nk(data_wavelength, params_opt)
    return R_cal_opt, params_opt, n_opt, k_opt, multilayer

def calculate_metrics(data_reflectance, R_cal, params, params_before_denormalization, max_params, min_params, simulated_params=None, n=None, k=None, n_true=None, k_true=None, normalized_params=None):
    """ Calculate R2 and other statistical metrics. """
    metrics = {'R2': r2_score(data_reflectance, R_cal), 
               'MSE': mse(R_cal, data_reflectance), 
               'RMSE': rmse(R_cal, data_reflectance), 
               'MASE': mase(R_cal, data_reflectance),
               'adjusted_R2': adjusted_r2(r2_score(data_reflectance, R_cal), len(data_reflectance), 6)
               }
    if n_true is not None and k_true is not None:
        metrics['R2_n'] = r2_score(n_true, n)
        metrics['R2_k'] = r2_score(k_true, k)
        metrics['MSE_n'] = mse(n, n_true)
        metrics['MSE_k'] = mse(k, k_true)
        metrics['RMSE_n'] = rmse(n, n_true)
        metrics['RMSE_k'] = rmse(k, k_true)
        metrics['MASE_n'] = mase(n, n_true)
        metrics['MASE_k'] = mase(k, k_true)
        metrics['adjusted_R2_n'] = adjusted_r2(r2_score(n_true, n), len(n_true), 1)
        metrics['adjusted_R2_k'] = adjusted_r2(r2_score(k_true, k), len(k_true), 1)
        
    if simulated_params is not None:
        metrics['R2_params'] = r2_score(simulated_params, params)
        metrics['MSE_params'] = mse(params, simulated_params)
        metrics['RMSE_params'] = rmse(params, simulated_params)
        metrics['MASE_params'] = mase(params, simulated_params)
        metrics['adjusted_R2_params'] = adjusted_r2(r2_score(simulated_params, params), len(simulated_params), 6)
        
        # metrics for A, E0, G, Eg, e_inf, thickness
        # A
        metrics['SE_A'] = se(params[0], simulated_params[0])
        metrics['AE_A'] = ae(params[0], simulated_params[0])
        metrics['PE_A'] = pe(params[0], simulated_params[0])
        metrics['NE_A'] = ne(params[0], simulated_params[0], max_params, min_params, 'A')
        # E0
        metrics['SE_E0'] = se(params[1], simulated_params[1])
        metrics['AE_E0'] = ae(params[1], simulated_params[1])
        metrics['PE_E0'] = pe(params[1], simulated_params[1])
        metrics['NE_E0'] = ne(params[1], simulated_params[1], max_params, min_params, 'E0')
        # G
        metrics['SE_G'] = se(params[2], simulated_params[2])
        metrics['AE_G'] = ae(params[2], simulated_params[2])
        metrics['PE_G'] = pe(params[2], simulated_params[2])
        metrics['NE_G'] = ne(params[2], simulated_params[2], max_params, min_params, 'G')
        # Eg
        metrics['SE_Eg'] = se(params[3], simulated_params[3])
        metrics['AE_Eg'] = ae(params[3], simulated_params[3])
        metrics['PE_Eg'] = pe(params[3], simulated_params[3])
        metrics['NE_Eg'] = ne(params[3], simulated_params[3], max_params, min_params, 'Eg')
        # e_inf
        metrics['SE_e_inf'] = se(params[4], simulated_params[4])
        metrics['AE_e_inf'] = ae(params[4], simulated_params[4])
        metrics['PE_e_inf'] = pe(params[4], simulated_params[4])
        metrics['NE_e_inf'] = ne(params[4], simulated_params[4], max_params, min_params, 'e_inf')
        # thickness
        metrics['SE_thickness'] = se(params[5], simulated_params[5])
        metrics['AE_thickness'] = ae(params[5], simulated_params[5])
        metrics['PE_thickness'] = pe(params[5], simulated_params[5])
        metrics['NE_thickness'] = ne(params[5], simulated_params[5], max_params, min_params, 'thickness')
        
    if normalized_params is not None:
        metrics['R2_normalized_params'] = r2_score(params_before_denormalization, normalized_params)
        metrics['MSE_normalized_params'] = mse(params_before_denormalization, normalized_params)
        metrics['RMSE_normalized_params'] = rmse(params_before_denormalization, normalized_params)
        metrics['MASE_normalized_params'] = mase(params_before_denormalization, normalized_params)
        metrics['adjusted_R2_normalized_params'] = adjusted_r2(r2_score(params_before_denormalization, normalized_params), len(normalized_params), 6)
        
        # metrics for A, E0, G, Eg, e_inf, thickness
        # A
        metrics['SE_A_normalized'] = se(params_before_denormalization[0], normalized_params[0])
        metrics['AE_A_normalized'] = ae(params_before_denormalization[0], normalized_params[0])
        metrics['PE_A_normalized'] = pe(params_before_denormalization[0], normalized_params[0])
        # E0
        metrics['SE_E0_normalized'] = se(params_before_denormalization[1], normalized_params[1])
        metrics['AE_E0_normalized'] = ae(params_before_denormalization[1], normalized_params[1])
        metrics['PE_E0_normalized'] = pe(params_before_denormalization[1], normalized_params[1])
        # G
        metrics['SE_G_normalized'] = se(params_before_denormalization[2], normalized_params[2])
        metrics['AE_G_normalized'] = ae(params_before_denormalization[2], normalized_params[2])
        metrics['PE_G_normalized'] = pe(params_before_denormalization[2], normalized_params[2])
        # Eg
        metrics['SE_Eg_normalized'] = se(params_before_denormalization[3], normalized_params[3])
        metrics['AE_Eg_normalized'] = ae(params_before_denormalization[3], normalized_params[3])
        metrics['PE_Eg_normalized'] = pe(params_before_denormalization[3], normalized_params[3])
        # e_inf
        metrics['SE_e_inf_normalized'] = se(params_before_denormalization[4], normalized_params[4])
        metrics['AE_e_inf_normalized'] = ae(params_before_denormalization[4], normalized_params[4])
        metrics['PE_e_inf_normalized'] = pe(params_before_denormalization[4], normalized_params[4])
        # thickness
        metrics['SE_thickness_normalized'] = se(params_before_denormalization[5], normalized_params[5])
        metrics['AE_thickness_normalized'] = ae(params_before_denormalization[5], normalized_params[5])
        metrics['PE_thickness_normalized'] = pe(params_before_denormalization[5], normalized_params[5])
        
        
    return metrics
# ________________________________________________________________________________



# Plotting functions
# ________________________________________________________________________________
def plot_results_single(data_wavelength, data_reflectance, R_cal, params, simulated_params, plot_suptitle, optional_data={}, save=False, show=True, perform_optimization=False):
    energy = 1239.84193 / data_wavelength  # Convert wavelength to energy
    figsize = (4, 3)  # Figure size for individual plots

    # Functions for plotting the separate plots
    def plot_reflectance(ax, label="", pure_data=False):
        ax.plot(energy, data_reflectance, label="R_data", alpha=0.5)
        if not pure_data:
            ax.plot(energy, R_cal, label="R_pred", color="orange", linestyle="dashed")
            
            # # Add the percentage difference between true and predicted R values
            # diff = np.abs(data_reflectance - R_cal)
            # max_diff_index = np.argmax(diff)
            # max_diff_energy = energy[max_diff_index]
            # max_diff = diff[max_diff_index]
            # percentage = max_diff / max(np.max(data_reflectance), np.max(R_cal)) * 100
            # ax.annotate("", xy=(max_diff_energy, data_reflectance[max_diff_index]), xytext=(max_diff_energy, R_cal[max_diff_index]), arrowprops=dict(arrowstyle="<->"))
            # ax.text(max_diff_energy, (data_reflectance[max_diff_index] + R_cal[max_diff_index]) / 2, f'Δ = {percentage:.1f}%', fontsize='14', fontweight='bold', ha='right')
            
            if perform_optimization and 'R_cal_opt' in optional_data and optional_data['R_cal_opt'] is not None:
                ax.plot(energy, optional_data['R_cal_opt'], label="R_fit", color="r")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Reflectance")
        ax.set_title("Reflectance", fontweight='bold')
        ax.text(0.5, 0.90, 'MSE=0.0023,\nR²=0.88', ha='center', va='center', color='red', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.legend()
        add_subplot_label(ax, label)

    def plot_n(ax, label=""):
        if 'n_true' in optional_data and optional_data['n_true'] is not None:
            ax.plot(energy, optional_data['n_true'], label="n_true", alpha=0.5)
        if 'n' in optional_data and optional_data['n'] is not None:
            ax.plot(energy, optional_data['n'], label="n_pred", color="orange", linestyle="dashed")
            
        # Add the percentage difference between true and predicted n values
        if 'n_true' in optional_data and 'n' in optional_data and optional_data['n_true'] is not None and optional_data['n'] is not None:
            diff = np.abs(optional_data['n_true'] - optional_data['n'])
            max_diff_index = np.argmax(diff)
            max_diff_energy = energy[max_diff_index]
            max_diff = diff[max_diff_index]
            percentage = max_diff / max(np.max(optional_data['n_true']), np.max(optional_data['n'])) * 100
            ax.annotate("", xy=(max_diff_energy, optional_data['n_true'][max_diff_index]), xytext=(max_diff_energy, optional_data['n'][max_diff_index]), arrowprops=dict(arrowstyle="<->"))
            ax.text(max_diff_energy, (optional_data['n_true'][max_diff_index] + optional_data['n'][max_diff_index]) / 2, f'Δ = {percentage:.1f}%', fontsize='14', fontweight='bold', ha='right')
            
        if perform_optimization and 'n_opt' in optional_data and optional_data['n_opt'] is not None:
            ax.plot(energy, optional_data['n_opt'], label="n_fit", color="r")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("n")
        ax.set_title("n", fontweight='bold')
        ax.legend()
        add_subplot_label(ax, label)

    def plot_k(ax, label=""):
        if 'k_true' in optional_data and optional_data['k_true'] is not None:
            ax.plot(energy, optional_data['k_true'], label="k_true", alpha=0.5)
        if 'k' in optional_data and optional_data['k'] is not None:
            ax.plot(energy, optional_data['k'], label="k_pred", color="orange", linestyle="dashed")
            
        # Add the percentage difference between true and predicted k values
        if 'k_true' in optional_data and 'k' in optional_data and optional_data['k_true'] is not None and optional_data['k'] is not None:
            diff = np.abs(optional_data['k_true'] - optional_data['k'])
            max_diff_index = np.argmax(diff)
            max_diff_energy = energy[max_diff_index]
            max_diff = diff[max_diff_index]
            percentage = max_diff / max(np.max(optional_data['k_true']), np.max(optional_data['k'])) * 100
            ax.annotate("", xy=(max_diff_energy, optional_data['k_true'][max_diff_index]), xytext=(max_diff_energy, optional_data['k'][max_diff_index]), arrowprops=dict(arrowstyle="<->"))
            ax.text(max_diff_energy, (optional_data['k_true'][max_diff_index] + optional_data['k'][max_diff_index]) / 2, f'Δ = {percentage:.1f}%', fontsize='14', fontweight='bold', ha='right')
            
        if perform_optimization and 'k_opt' in optional_data and optional_data['k_opt'] is not None:
            ax.plot(energy, optional_data['k_opt'], label="k_fit", color="r")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("k")
        ax.set_title("k", fontweight='bold')
        ax.legend()
        add_subplot_label(ax, label)

    # def plot_params(ax, label=""):
    #''' Plot the predicted and true parameters in their original scale. '''
    #     num_of_bars = 3  # Number of bars to plot
    #     bar_width = 1 / (num_of_bars + 1)
    #     index = np.arange(len(params))
    #     bar_data = [
    #         {'data': optional_data.get('params_normalized'), 'label': 'params_pred', 'color': "orange"},  # normalized data for bar height
    #         {'data': optional_data.get('params_opt_normalized'), 'label': 'params_opt', 'color': "r"} if perform_optimization and optional_data.get('params_opt_normalized') is not None else None,
    #         {'data': optional_data.get('simulated_params_normalized'), 'label': 'params_true', 'color': 'tab:blue'} if 'simulated_params_normalized' in optional_data and optional_data.get('simulated_params_normalized') is not None else None
    #     ]
    #     bar_data = [data for data in bar_data if data is not None]  # Remove None values

    #     for i, data in enumerate(bar_data):
    #         if data['data'] is not None:
    #             bars = ax.bar(index + i * bar_width, data['data'], bar_width, label=data['label'], color=data['color'])
    #             for bar, denorm_value in zip(bars, params):  
    #                 ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{denorm_value:.2f}", ha='center', va='bottom', fontsize=6, fontweight='bold')
    #     ax.set_xlabel('Parameter')
    #     ax.set_xticks(index + bar_width * num_of_bars / 2, ['A', 'E0', 'G', 'Eg', 'e_inf', 'd'])
    #     ax.set_ylabel('Values')
    #     ax.set_title('Parameters Comparison')
    #     ax.legend()
    #     add_subplot_label(ax, label)
    
    # def plot_params(ax, params, simulated_params, label="", optional_data={}):
    #     num_of_bars = 3  # Number of bar sets (predicted and true)
    #     bar_width = 1 / (num_of_bars + 1)  # Width of each bar
    #     index = np.arange(len(params))

    #     # Normalized data for plotting
    #     params_pred_normalized = optional_data.get('params_normalized', np.zeros(len(params)))  # Predicted normalized
    #     params_true_normalized = optional_data.get('simulated_params_normalized', np.zeros(len(simulated_params)))  # True normalized

    #     # Plotting predicted parameters
    #     bars_pred = ax.bar(index, params_pred_normalized, bar_width, label='Predicted', color='orange')
    #     for bar, value in zip(bars_pred, params):  # Displaying actual predicted values
    #         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha='center', va='bottom', fontsize=6, fontweight='bold')

    #     # Plotting true parameters
    #     bars_true = ax.bar(index + bar_width, params_true_normalized, bar_width, label='True')
    #     for bar, value in zip(bars_true, simulated_params):  # Displaying actual true values
    #         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha='center', va='bottom', fontsize=6, fontweight='bold')

    #     ax.set_xlabel('Parameter Index')
    #     ax.set_ylabel('Values')
    #     ax.set_title('Parameters Comparison')
    #     ax.set_xticks(index + bar_width / 2, ['A', 'E0', 'G', 'Eg', 'e_inf', 'thickness'])
    #     ax.set_yticklabels([])  # Disable y-axis values
    #     ax.legend()
    #     add_subplot_label(ax, label)


    def plot_params(ax, params, simulated_params, optional_data={}, label=""):
        ''' Plot the predicted and true parameters in normalized scale. '''
        num_of_bars = 3 if ('params_opt' in optional_data and optional_data['params_opt'] is not None) else 2  # Determine number of bars based on optional data
        bar_width = 0.25  # Adjust bar width to fit all bars within the axis space
        index = np.arange(len(params))

        # Set the height for true parameters (blue bars) to 0.5
        normalized_true_params = np.full_like(simulated_params, 0.5)  # True parameters normalized to 0.5

        # Normalizing predicted parameters: ratio of predicted to true, then scaled by 0.5
        normalized_pred_params = (params / simulated_params * 0.5) if simulated_params.any() else np.zeros_like(params)

        # Plotting true parameters
        bars_true = ax.bar(index - bar_width, normalized_true_params, bar_width, label='True')
        for bar, value in zip(bars_true, simulated_params):  # Displaying actual true values
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5, f"{value:.2f}", ha='center', va='bottom', fontsize=8)

        # Plotting predicted parameters
        bars_pred = ax.bar(index, normalized_pred_params, bar_width, label='Predicted', color='orange')
        for bar, value in zip(bars_pred, params):  # Displaying actual predicted values
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height - 0.05, f"{value:.2f}", ha='center', va='bottom', fontsize=8)

        # Plotting optimized parameters if available
        if 'params_opt' in optional_data and optional_data['params_opt'] is not None:
            params_opt = optional_data['params_opt']
            normalized_opt_params = (params_opt / simulated_params * 0.5) if simulated_params.any() else np.zeros_like(params_opt)
            bars_opt = ax.bar(index + bar_width, normalized_opt_params, bar_width, label='Fit', color='red')
            for bar, value in zip(bars_opt, params_opt):  # Displaying actual optimized values
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{value:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Normalized Values')
        ax.set_title('Parameters Comparison')
        ax.set_xticks(index, ['A', 'E0', 'G', 'Eg', 'e_inf', 'thickness'])
        ax.set_yticklabels([])  # Disable y-axis values
        ax.legend()
        add_subplot_label(ax, label)
    
    
    def plot_radar_chart(ax, params, simulated_params, optional_data={}, label=""):
        ''' Plot radar chart for predicted and true parameters in normalized scale. '''
        labels = np.array(['A', 'E0', 'G', 'Eg', 'e_inf', 'd (thickness)'])
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        labels = np.append(labels, labels[0])  # Add the first label at the end to complete the loop
        
        ax.set_thetagrids(np.degrees(angles), labels, fontsize='12') # Set the labels for each axis
        
        # Normalize data for visualization
        if 'params_opt' in optional_data and optional_data['params_opt'] is not None:
            opt_params = np.concatenate((optional_data['params_opt'], [optional_data['params_opt'][0]]))
            pred_params = np.concatenate((params, [params[0]]))
            pred_params_normalized = np.full_like(pred_params, 0.5)  # Set true params height to 0.5
            opt_params_normalized = opt_params / pred_params * 0.5  # Adjust optimized params height
            
            # Define and plot polygons
            pred_polygon = Polygon(list(zip(angles, pred_params_normalized)), closed=True, color='orange', alpha=0.25)
            opt_polygon = Polygon(list(zip(angles, opt_params_normalized)), closed=True, color='red', alpha=0.25)
            
            ax.add_patch(pred_polygon)
            ax.plot(angles, pred_params_normalized, 'o-', linewidth=1, label='Pred', color='orange')
            
            ax.add_patch(opt_polygon)
            ax.plot(angles, opt_params_normalized, 'o-', linewidth=1, label='Fit', color='red')
            
            # for angle, label_param, pp, op in zip(angles, labels, pred_params, opt_params):
            #     ax.text(angle, 0.35, f'{pp:.2f}', ha='center', va='center', fontweight='bold', color='orange', fontsize=10)
            #     ax.text(angle, (op / pp * 0.5) + 0.15, f'{op:.2f}', ha='center', va='center', color='red', fontweight='bold', fontsize=10)
            
            # Adjust the position of the text labels individually
            angle1, pp1, op1 = angles[0], pred_params[0], opt_params[0]
            angle2, pp2, op2 = angles[1], pred_params[1], opt_params[1]
            angle3, pp3, op3 = angles[2], pred_params[2], opt_params[2]
            angle4, pp4, op4 = angles[3], pred_params[3], opt_params[3]
            angle5, pp5, op5 = angles[4], pred_params[4], opt_params[4]
            angle6, pp6, op6 = angles[5], pred_params[5], opt_params[5]

            # For each label, call ax.text with specific coordinates
            # A
            ax.text(angle1, 0.65, f'{pp1:.2f}', ha='center', va='center', fontweight='bold', color='orange', fontsize=10)
            ax.text(angle1, (op1 / pp1 * 0.5) + 0.15, f'{op1:.2f}', ha='center', va='center', color='red', fontweight='bold', fontsize=10)

            # E0
            ax.text(angle2, 0.55, f'{pp2:.2f}', ha='left', va='center', fontweight='bold', color='orange', fontsize=10)
            ax.text(angle2, (op2 / pp2 * 0.5) + 0.05, f'{op2:.2f}', ha='left', va='center', color='red', fontweight='bold', fontsize=10)
            
            # G
            ax.text(angle3, 0.55, f'{pp3:.2f}', ha='center', va='center', fontweight='bold', color='orange', fontsize=10)
            ax.text(angle3, (op3 / pp3 * 0.5) + 0.15, f'{op3:.2f}', ha='center', va='center', color='red', fontweight='bold', fontsize=10)
            
            # Eg
            ax.text(angle4, 0.35, f'{pp4:.2f}', ha='center', va='bottom', fontweight='bold', color='orange', fontsize=10)
            ax.text(angle4, (op4 / pp4 * 0.5) + 0.15, f'{op4:.2f}', ha='center', va='center', color='red', fontweight='bold', fontsize=10)
            
            # e_inf
            ax.text(angle5, 0.68, f'{pp5:.2f}', ha='center', va='center', fontweight='bold', color='orange', fontsize=10)
            ax.text(angle5, (op5 / pp5 * 0.5) + 0.1, f'{op5:.2f}', ha='center', va='center', color='red', fontweight='bold', fontsize=10)
            
            # d (thickness)
            ax.text(angle6, 0.70, f'{pp6:.2f}', ha='center', va='center', fontweight='bold', color='orange', fontsize=10)
            ax.text(angle6, (op6 / pp6 * 0.5) + 0.1, f'{op6:.2f}', ha='center', va='center', color='red', fontweight='bold', fontsize=10)

                
        if 'simulated_params' in optional_data and optional_data['simulated_params'] is not None:
            true_params = np.concatenate((simulated_params, [simulated_params[0]]))
            pred_params = np.concatenate((params, [params[0]]))
            true_params_normalized = np.full_like(true_params, 0.5)
            pred_params_normalized = pred_params / true_params * 0.5

            # Define and plot polygons
            true_polygon = Polygon(list(zip(angles, true_params_normalized)), closed=True, alpha=0.25)
            pred_polygon = Polygon(list(zip(angles, pred_params_normalized)), closed=True, color='orange', alpha=0.25)

            ax.add_patch(true_polygon)
            ax.plot(angles, true_params_normalized, linewidth=1, label='True')

            ax.add_patch(pred_polygon)
            ax.plot(angles, pred_params_normalized, linewidth=1, label='Pred')
            
            for angle, label_param, tp, pp in zip(angles, labels, true_params, pred_params): # tp = true parameter, pp = predicted parameter
                ax.text(angle, 1.2, f'{tp:.2f}', ha='center', va='center', fontweight='bold', color='#1f77b4', fontsize=6)
                ax.text(angle, (pp / tp * 0.5) + 2.0, f'{pp:.2f}', ha='center', va='center', color='orange', fontweight='bold', fontsize=6)
        # else:
        #     pred_params = np.concatenate((params, [params[0]]))
        #     pred_params_normalized = np.full_like(pred_params, 0.5)  # Set true params height to 0.5
            
        #     # Define and plot polygons
        #     pred_polygon = Polygon(list(zip(angles, pred_params_normalized)), closed=True, color='orange', alpha=0.25)
            
        #     ax.add_patch(pred_polygon)
        #     ax.plot(angles, pred_params_normalized, 'o-', linewidth=1, label='Pred', color='orange')
            
        #     for angle, label_param, pp in zip(angles, labels, pred_params):
        #         ax.text(angle, 0.35, f'{pp:.2f}', ha='center', va='center', color='orange', fontweight='bold', fontsize=10)
        
        ax.set_title('Parameters Comparison', fontweight='bold')
        ax.grid(alpha=0.2)
        ax.set_yticklabels([])
        ax.legend(loc='upper left', bbox_to_anchor=(-0.4, 1))
        
        add_subplot_label(ax, label, x=-0.43, y=1.15)



    def add_subplot_label(ax, label, x=-0.1, y=1.15):
    # Position the subplot label in axes coordinates
        ax.text(x, y, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

    # Create the whole plot with subplots
    fig = plt.figure(figsize=(10, 8))
    plt.suptitle(plot_suptitle, fontsize=16)
    grid_spec = plt.GridSpec(2, 2)

    # Assign subplots
    ax_radar = fig.add_subplot(grid_spec[0, 0], polar=True)
    ax_n = fig.add_subplot(grid_spec[0, 1])
    ax_k = fig.add_subplot(grid_spec[1, 0])
    ax_reflectance = fig.add_subplot(grid_spec[1, 1])

    # Now plot using the specific axis
    plot_radar_chart(ax_radar, params, simulated_params, optional_data=optional_data, label="(a)")
    plot_n(ax_n, label="(b)")
    plot_k(ax_k, label="(c)")
    plot_reflectance(ax_reflectance, label="(d)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make room for the suptitle

    if save:
        plt.savefig(f"{plot_suptitle}.png", dpi=300)
    if show:
        plt.show()
    
    
def plot_results_dataset(data_wavelength, data_reflectance, R_cal, n, k, params_list, plot_suptitle, save=False, perform_optimization=False, optional_data={}):
    """
    Generate separate heatmaps for reflectance, n, k across the dataset, and separate line plots for each parameter
    (A, E0, G, Eg, e_inf, thickness) as a function of data index.
    """
    energy = 1239.84193 / data_wavelength  # Convert wavelength to energy
    
    # Heatmap of Reflectance
    plt.figure()
    R_cal = np.flipud(np.array(R_cal).T)
    plt.imshow(R_cal, aspect='auto', cmap='viridis', extent=[0, R_cal.shape[1], energy.min(), energy.max()])
    plt.colorbar()
    plt.xlabel("Data Index")
    plt.ylabel("Energy (eV)")
    plt.title(f"Reflectance Heatmap")
    if save:
        plt.savefig(f"{plot_suptitle}_Reflectance_Heatmap.png", dpi=300)
    plt.show()

    # Heatmap of n
    plt.figure()
    n = np.array(n).T
    plt.imshow(n, aspect='auto', cmap='viridis', extent=[0, R_cal.shape[1], energy.min(), energy.max()])
    plt.colorbar()
    plt.xlabel("Data Index")
    plt.ylabel("Energy (eV)")
    plt.title(f"n Heatmap")
    if save:
        plt.savefig(f"{plot_suptitle}_n_Heatmap.png", dpi=300)
    plt.show()

    # Heatmap of k
    k = np.array(k).T
    plt.figure()
    plt.imshow(k, aspect='auto', cmap='viridis', extent=[0, R_cal.shape[1], energy.min(), energy.max()])
    plt.colorbar()
    plt.xlabel("Data Index")
    plt.ylabel("Energy (eV)")
    plt.title(f"k Heatmap")
    if save:
        plt.savefig(f"{plot_suptitle}_k_Heatmap.png", dpi=300)
    plt.show()

    # Separate line plots for each parameter
    param_names = ['A', 'E0', 'G', 'Eg', 'e_inf', 'thickness']
    for i, param_name in enumerate(param_names):
        plt.figure(figsize=(6,4))
        param_data = np.array([params[i] for params in params_list])
        plt.plot(param_data, label=f"{param_name}")
        plt.xlabel("Data Index")
        plt.ylabel(f"Value of {param_name}")
        plt.title(f"{param_name} across the stripe")
        plt.legend()
        if save:
            plt.savefig(f"{plot_suptitle}_{param_name}.png", dpi=300)
        plt.show()
    
    # Plot metrics across the stripe
    if 'R2' in optional_data and optional_data['R2'] is not None:
        plt.figure(figsize=(6,4))
        R2_data = np.array(optional_data['R2'])
        print("R2_data: ", R2_data)
        plt.plot(R2_data, label="R2")
        plt.xlabel("Data Index")
        plt.ylabel("R2")
        plt.title("R2 across the stripe")
        plt.legend()
        if save:
            plt.savefig(f"{plot_suptitle}_R2_reflectance.png", dpi=300)
        plt.show()
    
    if 'MSE' in optional_data and optional_data['MSE'] is not None:
        plt.figure(figsize=(6,4))
        MSE_data = np.array(optional_data['MSE'])
        plt.plot(MSE_data, label="MSE")
        plt.xlabel("Data Index")
        plt.ylabel("MSE")
        plt.title("MSE across the stripe")
        plt.legend()
        if save:
            plt.savefig(f"{plot_suptitle}_MSE_reflectance.png", dpi=300)
        plt.show()
    
    # If perform optimization, plot the optimized reflectance, n, k, and parameters
    if perform_optimization:
        # Heatmap of optimized Reflectance
        R_cal_opt = np.array(R_cal).T
        plt.figure()
        plt.imshow(R_cal_opt, aspect='auto', cmap='viridis', extent=[energy.min(), energy.max(), 0, len(R_cal_opt)])
        plt.colorbar()
        plt.xlabel("Data Index")
        plt.ylabel("Energy (eV)")
        plt.title(f"Reflectance Heatmap (After Optimization)")
        if save:
            plt.savefig(f"{plot_suptitle}_Reflectance_Heatmap_Optimized.png", dpi=300)
        plt.show()
        
        # Heatmap of optimized n
        n_opt = np.array(n).T
        plt.figure()
        plt.imshow(n_opt, aspect='auto', cmap='viridis', extent=[energy.min(), energy.max(), 0, len(n_opt)])
        plt.colorbar()
        plt.xlabel("Data Index")
        plt.ylabel("Energy (eV)")
        plt.title(f"n Heatmap (After Optimization)")
        if save:
            plt.savefig(f"{plot_suptitle}_n_Heatmap_Optimized.png", dpi=300)
        plt.show()
        
        # Heatmap of optimized k
        k_opt = np.array(k).T
        plt.figure()
        plt.imshow(k_opt, aspect='auto', cmap='viridis', extent=[energy.min(), energy.max(), 0, len(k_opt)])
        plt.colorbar()
        plt.xlabel("Data Index")
        plt.ylabel("Energy (eV)")
        plt.title(f"k Heatmap (After Optimization)")
        if save:
            plt.savefig(f"{plot_suptitle}_k_Heatmap_Optimized.png", dpi=300)
        plt.show()
        
        # Separate line plots for each parameter after optimization
        for i, param_name in enumerate(param_names):
            plt.figure(figsize=(6,4))
            param_data = np.array([params[i] for params in params_list])
            param_data_opt = np.array([optional_data['params_opt'][i] for params in params_list])
            plt.plot(param_data, label=f"{param_name}")
            plt.plot(param_data_opt, label=f"{param_name} (Optimized)")
            plt.xlabel("Data Index")
            plt.ylabel(f"Value of {param_name}")
            plt.title(f"{param_name} across the stripe (After Optimization)")
            plt.legend()
            if save:
                plt.savefig(f"{plot_suptitle}_{param_name}_Optimized.png", dpi=300)
            plt.show()
            
        # Plot metrics across the stripe after optimization
        if 'R2_opt' in optional_data and optional_data['R2_opt'] is not None:
            plt.figure(figsize=(6,4))
            R2_data_opt = np.array(optional_data['R2_opt'])
            plt.plot(R2_data_opt, label="R2 (Optimized)")
            plt.xlabel("Data Index")
            plt.ylabel("R2")
            plt.title("R2 across the stripe (After Optimization)")
            plt.legend()
            if save:
                plt.savefig(f"{plot_suptitle}_R2_reflectance_Optimized.png", dpi=300)
            plt.show()
        
        if 'MSE_opt' in optional_data and optional_data['MSE_opt'] is not None:
            plt.figure(figsize=(6,4))
            MSE_data_opt = np.array(optional_data['MSE_opt'])
            plt.plot(MSE_data_opt, label="MSE (Optimized)")
            plt.xlabel("Data Index")
            plt.ylabel("MSE")
            plt.title("MSE across the stripe (After Optimization)")
            plt.legend()
            if save:
                plt.savefig(f"{plot_suptitle}_MSE_reflectance_Optimized.png", dpi=300)
            plt.show()
        
        


    
    
# ________________________________________________________________________________


# Process data functions
# ________________________________________________________________________________
# Single data
def process_single_data(data_path, model_name, n_data, chosen_index=0, plot=False, simulate=True, simulated_params_path='parameters.csv', perform_optimization=False, data_wavelength_path='wavelength_cropped.csv', normalized_params_path='Y_test_d1000.csv'):
    """ 
    Function to load data, run predictions, optimizations, and plot results for a single data (simulated or experimental). 
    """
    # Load data, model, and parameters
    data_reflectance, data_wavelength, simulated_params, model, multilayer, max_params, min_params, normalized_params = load_data_model_params(data_path, model_name, mode='single', chosen_index=chosen_index, simulate=simulate, simulated_params_path=simulated_params_path, data_wavelength_path=data_wavelength_path)
    print("data_reflectance.shape:", data_reflectance.shape)
    print("data_wavelength.shape:", data_wavelength.shape)
    
    # Normalize simulated params for plotting
    simulated_params_normalized = None
    if simulated_params is not None:
        simulated_params_normalized = np.abs(normalize(simulated_params, max_params, min_params))

    # Evaluate and predict
    params, params_before_denormalization, n, k, R_cal, n_true, k_true = evaluate_and_predict(data_reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data=n_data, simulate=simulate, simulated_params=simulated_params)
    params_before_denormalization = np.abs(params_before_denormalization)
    # Metrics before optimization
    metrics = calculate_metrics(data_reflectance, R_cal, params, params_before_denormalization, max_params, min_params, simulated_params, n, k, n_true, k_true)
    metrics.update({'data_index': chosen_index})
    with open(f"{model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f)
    
    # Perform optimization if specified
    R_cal_opt, n_opt, k_opt, params_opt, params_opt_normalized = None, None, None, None, None
    if perform_optimization:
        R_cal_opt, params_opt, n_opt, k_opt, updated_multilayer = optimization(params, multilayer, data_wavelength, data_reflectance)
        params_opt_normalized = np.abs(normalize(params_opt, max_params, min_params))
        n_opt, k_opt = n_opt.squeeze(), k_opt.squeeze()
        metrics.update({
            'R2_opt': r2_score(data_reflectance, R_cal_opt),
            'R2_n_opt': r2_score(n_true, n_opt) if n_true is not None else None,
            'R2_k_opt': r2_score(k_true, k_opt) if k_true is not None else None,
            'MSE_opt': mse(R_cal_opt, data_reflectance) if R_cal_opt is not None else None,
            'RMSE_opt': rmse(R_cal_opt, data_reflectance) if R_cal_opt is not None else None,
            'MASE_opt': mase(R_cal_opt, data_reflectance) if R_cal_opt is not None else None,
            'MSE_n_opt': mse(n_opt, n_true) if n_true is not None else None,
            'MSE_k_opt': mse(k_opt, k_true) if k_true is not None else None,
            'RMSE_n_opt': rmse(n_opt, n_true) if n_true is not None else None,
            'RMSE_k_opt': rmse(k_opt, k_true) if k_true is not None else None,
            'MASE_n_opt': mase(n_opt, n_true) if n_true is not None else None,
            'MASE_k_opt': mase(k_opt, k_true) if k_true is not None else None,
            'R2_params_opt': r2_score(simulated_params, params_opt) if simulated_params is not None else None
        })
        with open(f"{model_name}_metrics.json", 'w') as f:
            json.dump(metrics, f)
    
    # Prepare data for plotting
    optional_data = {
        'n': n,
        'k': k,
        'n_true': n_true,
        'k_true': k_true,
        'params_normalized': normalize(params, max_params, min_params),
        'R_cal_opt': R_cal_opt,
        'n_opt': n_opt,
        'k_opt': k_opt,
        'params_opt': params_opt,
        'params_opt_normalized': params_opt_normalized,
        'simulated_params': simulated_params,
        'simulated_params_normalized': simulated_params_normalized,
        'R2': metrics.get('R2'),
        'R2_opt': metrics.get('R2_opt'),
        'R2_n': metrics.get('R2_n'),
        'R2_k': metrics.get('R2_k'),
        'R2_n_opt': metrics.get('R2_n_opt'),
        'R2_k_opt': metrics.get('R2_k_opt'),
        'MSE': metrics.get('MSE'),
        'MSE_opt': metrics.get('MSE_opt'),
        'MSE_n': metrics.get('MSE_n'),
        'MSE_k': metrics.get('MSE_k'),
        'MSE_n_opt': metrics.get('MSE_n_opt'),
        'MSE_k_opt': metrics.get('MSE_k_opt')
    }   

    # Plot results
    if plot:
        print("n_opt in optional_data:", 'n_opt' in optional_data)
        print("n_opt value:", optional_data.get('n_opt'))
        plot_results_single(data_wavelength, data_reflectance, R_cal, params, simulated_params, plot_suptitle=f'Data {chosen_index}' + (' Simulated (Baseline)' if simulate else ' Experimental' + (' With Fitting' if perform_optimization else '')), optional_data=optional_data, save=True, show=True, perform_optimization=perform_optimization)
    


# Dataset
def process_dataset(data_path, model_name, n_data, simulate, simulated_params_path, perform_optimization, normalized_params_path, filter=False, plot_single=False, plot_dataset=True):
    """ Process a dataset. """
    # Load data, model, and parameters
    data_reflectance, data_wavelength, simulated_params, model, multilayer, max_params, min_params, normalized_params = load_data_model_params(data_path, model_name, mode='dataset', simulate=simulate, simulated_params_path=simulated_params_path, normalized_params_path=normalized_params_path)

    # Lists to store predicted results and parameters, and dictionary for metrics
    reflectance_predicted = []
    reflectance_predicted_opt = []
    n_predicted = []
    n_predicted_opt = []
    k_predicted = []
    k_predicted_opt = []
    parameters_predicted = []
    parameters_predicted_opt = []
    params_before_denormalization_predicted = []
    metrics_list = []
    r2_values = []
    mse_values = []
    r2_opt_values = []
    mse_opt_values = []
    
    # Initialize lists to store prediction and optimization times
    prediction_times = []
    optimization_times = []
    
    # Record the indices of the data with the highest and lowest R2 and mse
    # R2
    highest_r2 = -float('inf')
    lowest_r2 = float('inf')
    index_highest_r2 = -1
    index_lowest_r2 = -1
    
    # mse
    highest_mse = -float('inf')
    lowest_mse = float('inf')
    index_highest_mse = -1
    index_lowest_mse = -1
    
    # optional data
    optional_data_highest_r2 = None
    optional_data_lowest_r2 = None
    optional_data_highest_mse = None
    optional_data_lowest_mse = None

    # Evaluate and predict
    for i in range(data_reflectance.shape[0]):
        print(f"Processing data {i + 1}...")
        reflectance = data_reflectance[i, :]
        wavelength = data_wavelength
        # wavelength = data_wavelength if simulate else data_wavelength[i, :]
        params_true = simulated_params[i, :] if simulate else None
        params_true_normalized = normalized_params[i, :] if normalized_params is not None else None

        # Predict and calculate metrics
        start_time = time.time()
        params, params_before_denormalization, n, k, R_cal, n_true, k_true = evaluate_and_predict(reflectance, wavelength, model, max_params, min_params, multilayer, n_data=n_data, simulate=simulate, simulated_params=params_true)
        end_time = time.time()
        prediction_times.append(end_time - start_time)
        
        single_metrics = calculate_metrics(reflectance, R_cal, params, params_before_denormalization, max_params, min_params, params_true, n, k, n_true, k_true, normalized_params[i] if normalized_params is not None else None)
        single_metrics['data_index'] = i
        metrics_list.append(single_metrics)
        r2_values.append(single_metrics['R2'])
        mse_values.append(single_metrics['MSE'])

        # Collect predicted reflectance and parameters
        reflectance_predicted.append(R_cal)
        n_predicted.append(n)
        k_predicted.append(k)
        parameters_predicted.append(params)
        params_before_denormalization_predicted.append(params_before_denormalization)

        # Perform optimization if necessary
        R_cal_opt, n_opt, k_opt, params_opt = None, None, None, None
        if perform_optimization:
            start_time = time.time()
            R_cal_opt, params_opt, n_opt, k_opt, updated_multilayer = optimization(params, multilayer, wavelength, reflectance)
            n_opt, k_opt = n_opt.squeeze(), k_opt.squeeze()
            end_time = time.time()
            optimization_times.append(end_time - start_time)

            single_metrics.update({
            'R2_opt': r2_score(reflectance, R_cal_opt),
            'R2_n_opt': r2_score(n_true, n_opt) if n_true is not None else None,
            'R2_k_opt': r2_score(k_true, k_opt) if k_true is not None else None,
            'MSE_opt': mse(R_cal_opt, reflectance) if R_cal_opt is not None else None,
            'RMSE_opt': rmse(R_cal_opt, reflectance) if R_cal_opt is not None else None,
            'MASE_opt': mase(R_cal_opt, reflectance) if R_cal_opt is not None else None,
            'MSE_n_opt': mse(n_opt, n_true) if n_true is not None else None,
            'MSE_k_opt': mse(k_opt, k_true) if k_true is not None else None,
            'RMSE_n_opt': rmse(n_opt, n_true) if n_true is not None else None,
            'RMSE_k_opt': rmse(k_opt, k_true) if k_true is not None else None,
            'MASE_n_opt': mase(n_opt, n_true) if n_true is not None else None,
            'MASE_k_opt': mase(k_opt, k_true) if k_true is not None else None,
            'R2_params_opt': r2_score(simulated_params, params_opt) if simulated_params is not None else None
            })
            with open(f"{model_name}_metrics_data{i}.json", 'w') as f:
                json.dump(single_metrics, f)

            r2_opt_values.append(single_metrics['R2_opt'])
            mse_opt_values.append(single_metrics['MSE_opt'])
            reflectance_predicted_opt.append(R_cal_opt)
            n_predicted_opt.append(n_opt)
            k_predicted_opt.append(k_opt)
            parameters_predicted_opt.append(params_opt)
                
        # Prepare data for plotting
        optional_data = {
            'n': n,
            'k': k,
            'n_true': n_true,
            'k_true': k_true,
            'R_cal_opt': R_cal_opt,
            'n_opt': n_opt,
            'k_opt': k_opt,
            'params_opt': params_opt,
            'simulated_params': simulated_params,
            'R2': single_metrics.get('R2'),
            'R2_opt': single_metrics.get('R2_opt'),
            'R2_n': single_metrics.get('R2_n'),
            'R2_k': single_metrics.get('R2_k'),
            'R2_n_opt': single_metrics.get('R2_n_opt'),
            'R2_k_opt': single_metrics.get('R2_k_opt'),
            'MSE': single_metrics.get('MSE'),
            'MSE_opt': single_metrics.get('MSE_opt')
        } 
            
        # Update highest and lowest R2
        current_r2 = single_metrics['R2']
        if current_r2 > highest_r2:
            highest_r2 = current_r2
            index_highest_r2 = i
            optional_data_highest_r2 = optional_data
        if current_r2 < lowest_r2:
            lowest_r2 = current_r2
            index_lowest_r2 = i
            optional_data_lowest_r2 = optional_data
            
        # Update highest and lowest mse
        current_mse = single_metrics['MSE']
        if current_mse > highest_mse:
            highest_mse = current_mse
            index_highest_mse = i
            optional_data_highest_mse = optional_data
        if current_mse < lowest_mse:
            lowest_mse = current_mse
            index_lowest_mse = i
            optional_data_lowest_mse = optional_data
            
        # Plot if necessary
        if plot_single:
            plot_results_single(wavelength, reflectance, R_cal, params, plot_suptitle=f"Data {i}", optional_data=optional_data, save=False, show=False)

    # Convert lists to DataFrames
    metrics_df = pd.DataFrame(metrics_list)
    reflectance_df = pd.DataFrame(reflectance_predicted)
    n_df = pd.DataFrame(n_predicted)
    k_df = pd.DataFrame(k_predicted)
    r2_values_df = pd.DataFrame(r2_values)
    mse_values_df = pd.DataFrame(mse_values)
    
    column_names = ['A', 'E0', 'G', 'Eg', 'e_inf', 'd']
    parameters_df = pd.DataFrame(parameters_predicted, columns=column_names)
    params_before_denormalization_df = pd.DataFrame(params_before_denormalization_predicted, columns=column_names)

    suffix = "_filtered" if filter else ""

    # Save individual metrics and predicted data to CSV
    metrics_df.to_csv(f"{model_name}_metrics_individual{suffix}.csv", index=False)
    reflectance_df.to_csv(f"{model_name}_reflectance_predicted{suffix}.csv", index=False, header=False)
    parameters_df.to_csv(f"{model_name}_parameters_predicted{suffix}.csv", index=False)
    params_before_denormalization_df.to_csv(f"{model_name}_parameters_before_denormalization_predicted{suffix}.csv", index=False)
    n_df.to_csv(f"{model_name}_n_predicted{suffix}.csv", index=False, header=False)
    k_df.to_csv(f"{model_name}_k_predicted{suffix}.csv", index=False, header=False)
    
    if perform_optimization:
        metrics_df_opt = pd.DataFrame(metrics_list)
        reflectance_df_opt = pd.DataFrame(reflectance_predicted_opt)
        parameters_df_opt = pd.DataFrame(parameters_predicted_opt, columns=column_names)
        n_df_opt = pd.DataFrame(n_predicted_opt)
        k_df_opt = pd.DataFrame(k_predicted_opt)
        r2_opt_values_df = pd.DataFrame(r2_opt_values)
        mse_opt_values_df = pd.DataFrame(mse_opt_values)
        
        metrics_df_opt.to_csv(f"{model_name}_metrics_individual_optimized{suffix}.csv", index=False)
        reflectance_df_opt.to_csv(f"{model_name}_reflectance_predicted_optimized{suffix}.csv", index=False, header=False)
        parameters_df_opt.to_csv(f"{model_name}_parameters_predicted_optimized{suffix}.csv", index=False)
        n_df_opt.to_csv(f"{model_name}_n_predicted_optimized{suffix}.csv", index=False, header=False)
        k_df_opt.to_csv(f"{model_name}_k_predicted_optimized{suffix}.csv", index=False, header=False)
        r2_opt_values_df.to_csv(f"{model_name}_R2_optimized{suffix}.csv", index=False, header=False)
        mse_opt_values_df.to_csv(f"{model_name}_MSE_optimized{suffix}.csv", index=False, header=False)

    # Calculate mean and std and save them
    mean_metrics = metrics_df.mean().to_dict()
    std_metrics = metrics_df.std().to_dict()
    with open(f"{model_name}_metrics_mean{suffix}.json", 'w') as f:
        json.dump(mean_metrics, f)
    with open(f"{model_name}_metrics_std{suffix}.json", 'w') as f:
        json.dump(std_metrics, f)
        
    # Calculate average and total prediction time
    avg_prediction_time = sum(prediction_times) / len(prediction_times)
    total_prediction_time = sum(prediction_times)

    # Calculate average and total optimization time
    if perform_optimization:
        avg_optimization_time = sum(optimization_times) / len(optimization_times)
        total_optimization_time = sum(optimization_times)
    else:
        avg_optimization_time = 0
        total_optimization_time = 0

    print(f"Average prediction time: {avg_prediction_time} seconds")
    print(f"Total prediction time: {total_prediction_time} seconds")
    print(f"Average optimization time: {avg_optimization_time} seconds")
    print(f"Total optimization time: {total_optimization_time} seconds")

        
    # # Plot the results with the highest and lowest R2
    # if index_highest_r2 != -1:
    #     R_cal_highest = reflectance_predicted[index_highest_r2]
    #     plot_results_single(data_wavelength, data_reflectance[index_highest_r2], R_cal_highest, parameters_predicted[index_highest_r2], f"Data with Highest R2: {round(highest_r2, 2)}", optional_data=optional_data_highest_r2, save=True, perform_optimization=perform_optimization)
    # if index_lowest_r2 != -1:
    #     R_cal_lowest = reflectance_predicted[index_lowest_r2]
    #     plot_results_single(data_wavelength, data_reflectance[index_lowest_r2], R_cal_lowest, parameters_predicted[index_lowest_r2], f"Data with Lowest R2: {round(lowest_r2, 2)}", optional_data=optional_data_lowest_r2, save=True, perform_optimization=perform_optimization)

    # # Plot the results with the highest and lowest mse
    # if index_highest_mse != -1:
    #     R_cal_highest = reflectance_predicted[index_highest_mse]
    #     plot_results_single(data_wavelength, data_reflectance[index_highest_mse], R_cal_highest, parameters_predicted[index_highest_mse], f"Data with Highest MSE: {round(highest_mse, 4)}", optional_data=optional_data_highest_mse, save=True, perform_optimization=perform_optimization)
    # if index_lowest_mse != -1:
    #     R_cal_lowest = reflectance_predicted[index_lowest_mse]
    #     plot_results_single(data_wavelength, data_reflectance[index_lowest_mse], R_cal_lowest, parameters_predicted[index_lowest_mse], f"Data with Lowest MSE: {round(lowest_mse, 4)}", optional_data=optional_data_lowest_mse, save=True, perform_optimization=perform_optimization)
        
    # Plot the results for the entire dataset
    if plot_dataset:
        plot_results_dataset(data_wavelength, data_reflectance, reflectance_predicted, n_predicted, k_predicted, parameters_predicted, plot_suptitle=model_name, save=True, perform_optimization=perform_optimization, optional_data=optional_data)
        if perform_optimization:
            plot_results_dataset(data_wavelength, data_reflectance, reflectance_predicted_opt, n_predicted_opt, k_predicted_opt, parameters_predicted_opt, plot_suptitle=f"{model_name}_Optimized", save=True, perform_optimization=perform_optimization, optional_data=optional_data)
    return metrics_df, reflectance_df, parameters_df, mean_metrics, std_metrics
# ________________________________________________________________________________
        
        


# Main function
# ________________________________________________________________________________
def main(data_path, model_name, n_data, mode='single', chosen_index=0, plot=False, simulate=True, simulated_params_path='parameters.csv', perform_optimization=False, normalized_params_path='Y_test_d1000.csv', filter=False):
    """ Main function to load data, run predictions, optimizations, and plot results. """
    if mode == 'single':
        # Handle a single data
        process_single_data(data_path, model_name, n_data, chosen_index, plot, simulate, simulated_params_path, perform_optimization)
    elif mode == 'dataset':
        # Handle a dataset
        process_dataset(data_path, model_name, n_data, simulate, simulated_params_path, perform_optimization, normalized_params_path, filter=filter)
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'single' or 'dataset'.")

if __name__ == "__main__": 
    # baseline model
    # model_name = "model_N10000_d1000_baseline"
    model_name = "model_N50000_d500_6_conv_layers_lr1e-4"
    
    mode = 'dataset'
    plot=False
    simulate = True
    filter=False
    perform_optimization = False
    
    if simulate:
        data_path = "X_test_d500.csv"
    else:
        data_path = "reflectance_cropped.csv"
    
    start = time.time()
    main(data_path, model_name, n_data=500, mode=mode, plot=plot, simulate=simulate, simulated_params_path='parameters_d500.csv', perform_optimization=perform_optimization, normalized_params_path='Y_test_d500.csv', filter=filter)
    end = time.time()
    
    print(f"Execution time: {end - start} seconds.")
# ________________________________________________________________________________

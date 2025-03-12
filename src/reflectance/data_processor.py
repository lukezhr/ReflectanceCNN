import json
import time
import numpy as np
from .evaluation import ModelEvaluationResults
from .model_eval_pkg import optimize_TL, construct_bi2o3_multilayer, predict, optimization
from .TaucLorentz import TL_nk
from .helper import load_data_model_params
from .plotting import Plotter

def evaluate_and_predict(data_reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data, simulate=False, simulated_params=None):
    """ 
    Predict parameters, n, and k, and calculate R_cal using the given model and data.
    If the data is simulated, return the true n and k values as well.
    """
    params, params_before_denormalization, n, k, R_cal = predict(data_reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data=n_data)
    
    n_true, k_true = None, None
    if simulate and simulated_params is not None:
        _, _, n_true, k_true = TL_nk(data_wavelength, simulated_params)
        n_true, k_true = n_true.flatten(), k_true.flatten()
    
    # Create and return a ModelEvaluationResults instance
    return ModelEvaluationResults(
        data_reflectance=data_reflectance,
        R_cal=R_cal,
        params=params,
        params_before_denormalization=params_before_denormalization,
        max_params=max_params,
        min_params=min_params,
        simulated_params=simulated_params,
        n=n,
        k=k,
        n_true=n_true,
        k_true=k_true
    )

def process_single_data(data_path, model_name, n_data, chosen_index=0, plot=False, simulate=True, simulated_params_path='parameters.csv', perform_optimization=False, data_wavelength_path='wavelength_cropped.csv', normalized_params_path='Y_test_d1000.csv'):
    """ 
    Function to load data, run predictions, optimizations, and plot results for a single data (simulated or experimental). 
    """
    # Load data, model, and parameters
    data_reflectance, data_wavelength, simulated_params, model, multilayer, max_params, min_params, normalized_params = load_data_model_params(data_path, model_name, mode='single', chosen_index=chosen_index, simulate=simulate, simulated_params_path=simulated_params_path, data_wavelength_path=data_wavelength_path)
    
    # Evaluate and predict
    results = evaluate_and_predict(data_reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data=n_data, simulate=simulate, simulated_params=simulated_params)
    
    # Save metrics
    results.save_metrics(f"results/metrics/{model_name}_metrics.json")
    
    # Perform optimization if specified
    if perform_optimization:
        R_cal_opt, params_opt, n_opt, k_opt, updated_multilayer = optimization(results.params, multilayer, data_wavelength, data_reflectance)
        results.add_optimization_results(R_cal_opt, params_opt, n_opt, k_opt)
        results.save_metrics(f"results/metrics/{model_name}_metrics.json")
    
    # Plot results
    if plot:
        plotter = Plotter(data_wavelength, data_reflectance, results.R_cal, results.params, simulated_params, optional_data=results.get_plotting_data())
        plotter.plot_single_results(f'Data {chosen_index}' + (' Simulated (Baseline)' if simulate else ' Experimental' + (' With Fitting' if perform_optimization else '')), save=True, show=True)

def process_dataset(data_path, model_name, n_data, simulate, simulated_params_path, perform_optimization, normalized_params_path, filter=False, plot_single=False, plot_dataset=True):
    """ Process a dataset. """
    # Load data, model, and parameters
    data_reflectance, data_wavelength, simulated_params, model, multilayer, max_params, min_params, normalized_params = load_data_model_params(data_path, model_name, mode='dataset', simulate=simulate, simulated_params_path=simulated_params_path, normalized_params_path=normalized_params_path)

    # Lists to store results
    all_results = []
    metrics_list = []
    r2_values = []
    mse_values = []
    r2_opt_values = []
    mse_opt_values = []

    # Evaluate and predict for each data point
    for i in range(data_reflectance.shape[0]):
        print(f"Processing data {i + 1}...")
        reflectance = data_reflectance[i, :]
        params_true = simulated_params[i, :] if simulate else None
        params_true_normalized = normalized_params[i, :] if normalized_params is not None else None

        # Predict and calculate metrics
        results = evaluate_and_predict(reflectance, data_wavelength, model, max_params, min_params, multilayer, n_data=n_data, simulate=simulate, simulated_params=params_true)
        
        # Perform optimization if specified
        if perform_optimization:
            R_cal_opt, params_opt, n_opt, k_opt, updated_multilayer = optimization(results.params, multilayer, data_wavelength, reflectance)
            results.add_optimization_results(R_cal_opt, params_opt, n_opt, k_opt)
        
        # Save results
        all_results.append(results)
        metrics_list.append(results.metrics)
        r2_values.append(results.metrics['R2'])
        mse_values.append(results.metrics['MSE'])
        if perform_optimization:
            r2_opt_values.append(results.metrics['R2_opt'])
            mse_opt_values.append(results.metrics['MSE_opt'])
        
        # Plot single data if specified
        if plot_single:
            plotter = Plotter(data_wavelength, reflectance, results.R_cal, results.params, params_true, optional_data=results.get_plotting_data())
            plotter.plot_single_results(f'Data {i}' + (' Simulated (Baseline)' if simulate else ' Experimental' + (' With Fitting' if perform_optimization else '')), save=True, show=True)
    
    # Plot dataset results if specified
    if plot_dataset:
        plotter = Plotter(data_wavelength, data_reflectance, [r.R_cal for r in all_results], [r.n for r in all_results], [r.k for r in all_results], [r.params for r in all_results], optional_data={'R2': r2_values, 'MSE': mse_values, 'R2_opt': r2_opt_values, 'MSE_opt': mse_opt_values})
        plotter.plot_dataset_results('Dataset Results', save=True, perform_optimization=perform_optimization)
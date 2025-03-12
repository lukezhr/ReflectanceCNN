import json
from .metrics import mse, rmse, adjusted_r2, mase
from .helper import normalize
import numpy as np
from sklearn.metrics import r2_score

class ModelEvaluationResults:
    def __init__(self, data_reflectance, R_cal, params, params_before_denormalization, 
                 max_params, min_params, simulated_params=None, n=None, k=None, 
                 n_true=None, k_true=None, normalized_params=None):
        self.data_reflectance = data_reflectance
        self.R_cal = R_cal
        self.params = params
        self.params_before_denormalization = params_before_denormalization
        self.max_params = max_params
        self.min_params = min_params
        self.simulated_params = simulated_params
        self.n = n
        self.k = k
        self.n_true = n_true
        self.k_true = k_true
        self.normalized_params = normalized_params
        self.metrics = self.calculate_metrics()
        self.optimization_results = None  # To store optimization results

    def calculate_metrics(self):
        """Calculate all metrics and return as a dictionary"""
        metrics = {
            'R2': r2_score(self.data_reflectance, self.R_cal),
            'MSE': mse(self.data_reflectance, self.R_cal),
            'RMSE': rmse(self.data_reflectance, self.R_cal),
            'MASE': mase(self.data_reflectance, self.R_cal),
            'adjusted_R2': adjusted_r2(r2_score(self.data_reflectance, self.R_cal), 
                                     len(self.data_reflectance), 6)
        }

        if self.n_true is not None and self.k_true is not None:
            self._add_nk_metrics(metrics)
            
        if self.simulated_params is not None:
            self._add_params_metrics(metrics)
            
        if self.normalized_params is not None:
            self._add_normalized_params_metrics(metrics)
            
        return metrics

    def _add_nk_metrics(self, metrics):
        """Calculate and store n and k related metrics."""
        if self.n_true is not None and self.k_true is not None:
            metrics.update({
                'R2_n': r2_score(self.n_true, self.n),
                'R2_k': r2_score(self.k_true, self.k),
                'MSE_n': mse(self.n_true, self.n),
                'MSE_k': mse(self.k_true, self.k),
                'RMSE_n': rmse(self.n_true, self.n),
                'RMSE_k': rmse(self.k_true, self.k),
                'MASE_n': mase(self.n_true, self.n),
                'MASE_k': mase(self.k_true, self.k)
            })

    def _add_params_metrics(self, metrics):
        """Calculate and store parameter-related metrics."""
        if self.simulated_params is not None:
            metrics.update({
                'R2_params': r2_score(self.simulated_params, self.params),
                'MSE_params': mse(self.simulated_params, self.params),
                'RMSE_params': rmse(self.simulated_params, self.params),
                'MASE_params': mase(self.simulated_params, self.params)
            })

    def _add_normalized_params_metrics(self, metrics):
        """Calculate and store normalized parameter-related metrics."""
        if self.normalized_params is not None:
            metrics.update({
                'R2_normalized_params': r2_score(self.normalized_params, normalize(self.params, self.max_params, self.min_params)),
                'MSE_normalized_params': mse(self.normalized_params, normalize(self.params, self.max_params, self.min_params)),
                'RMSE_normalized_params': rmse(self.normalized_params, normalize(self.params, self.max_params, self.min_params)),
                'MASE_normalized_params': mase(self.normalized_params, normalize(self.params, self.max_params, self.min_params))
            })

    def add_optimization_results(self, R_cal_opt, params_opt, n_opt, k_opt):
        """Add optimization results and update metrics"""
        self.optimization_results = {
            'R_cal_opt': R_cal_opt,
            'params_opt': params_opt,
            'n_opt': n_opt,
            'k_opt': k_opt,
            'params_opt_normalized': normalize(params_opt, self.max_params, self.min_params)
        }
        
        # Update metrics with optimization results
        self.metrics.update({
            'R2_opt': r2_score(self.data_reflectance, R_cal_opt),
            'R2_n_opt': r2_score(self.n_true, n_opt) if self.n_true is not None else None,
            'R2_k_opt': r2_score(self.k_true, k_opt) if self.k_true is not None else None,
            'MSE_opt': mse(R_cal_opt, self.data_reflectance),
            'RMSE_opt': rmse(R_cal_opt, self.data_reflectance),
            'MASE_opt': mase(R_cal_opt, self.data_reflectance),
            'MSE_n_opt': mse(n_opt, self.n_true) if self.n_true is not None else None,
            'MSE_k_opt': mse(k_opt, self.k_true) if self.k_true is not None else None,
            'RMSE_n_opt': rmse(n_opt, self.n_true) if self.n_true is not None else None,
            'RMSE_k_opt': rmse(k_opt, self.k_true) if self.k_true is not None else None,
            'MASE_n_opt': mase(n_opt, self.n_true) if self.n_true is not None else None,
            'MASE_k_opt': mase(k_opt, self.k_true) if self.k_true is not None else None,
            'R2_params_opt': r2_score(self.simulated_params, params_opt) if self.simulated_params is not None else None
        })

    def get_plotting_data(self):
        """Prepare data for plotting"""
        plotting_data = {
            'n': self.n,
            'k': self.k,
            'n_true': self.n_true,
            'k_true': self.k_true,
            'params_normalized': normalize(self.params, self.max_params, self.min_params),
            'simulated_params': self.simulated_params,
            'simulated_params_normalized': normalize(self.simulated_params, 
                                                   self.max_params, self.min_params) 
                                                   if self.simulated_params is not None else None,
            **self.metrics
        }
        
        if self.optimization_results is not None:
            plotting_data.update(self.optimization_results)
        
        return plotting_data

    def save_metrics(self, file_path):
        """Save metrics to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f)
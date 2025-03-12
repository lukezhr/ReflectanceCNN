import time
from typing import Optional
from reflectance.cnn_structure import ReflectanceCNN
from reflectance.data_processor import process_single_data, process_dataset
from reflectance.helper import load_data_model_params
from reflectance.plotting import Plotter
from reflectance.evaluation import ModelEvaluationResults
from reflectance.model_eval_pkg import predict, optimize_TL, construct_bi2o3_multilayer, optimization

def main(
    data_path: str,
    model_name: str,
    n_data: int,
    mode: str = 'single',
    chosen_index: int = 0,
    plot: bool = False,
    simulate: bool = True,
    simulated_params_path: Optional[str] = None,
    perform_optimization: bool = False,
    normalized_params_path: Optional[str] = None,
    filter: bool = False,
    save_plots: bool = True,
    save_metrics: bool = True
) -> None:
    """
    Main function to load data, run predictions, optimizations, and plot results.

    Args:
        data_path (str): Path to the reflectance data file.
        model_name (str): Name of the model to use.
        n_data (int): Number of data points.
        mode (str): Mode of operation ('single' or 'dataset'). Defaults to 'single'.
        chosen_index (int): Index of the data point to process (for 'single' mode). Defaults to 0.
        plot (bool): Whether to plot results. Defaults to False.
        simulate (bool): Whether the data is simulated. Defaults to True.
        simulated_params_path (Optional[str]): Path to the simulated parameters file. Defaults to None.
        perform_optimization (bool): Whether to perform optimization. Defaults to False.
        normalized_params_path (Optional[str]): Path to the normalized parameters file. Defaults to None.
        filter (bool): Whether to filter the dataset. Defaults to False.
        save_plots (bool): Whether to save plots. Defaults to True.
        save_metrics (bool): Whether to save metrics. Defaults to True.
    """
    if mode == 'single':
        process_single_data(
            data_path=data_path,
            model_name=model_name,
            n_data=n_data,
            chosen_index=chosen_index,
            plot=plot,
            simulate=simulate,
            simulated_params_path=simulated_params_path,
            perform_optimization=perform_optimization,
            data_wavelength_path='data/experimental/wavelength_cropped.csv',
            normalized_params_path=normalized_params_path
        )
    elif mode == 'dataset':
        process_dataset(
            data_path=data_path,
            model_name=model_name,
            n_data=n_data,
            simulate=simulate,
            simulated_params_path=simulated_params_path,
            perform_optimization=perform_optimization,
            normalized_params_path=normalized_params_path,
            filter=filter,
            plot_single=plot,
            plot_dataset=plot
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'single' or 'dataset'")

if __name__ == "__main__":
    # Example usage
    model_name = "training/model_N50000_d500_6_conv_layers_lr1e-3"
    mode = 'dataset'  # 'single' or 'dataset'
    plot = False
    simulate = True
    filter = False
    perform_optimization = False

    if simulate:
        data_path = "data/synthetic/X_test_d500.csv"
    else:
        data_path = "data/experimental/reflectance_cropped.csv"

    start = time.time()
    main(
        data_path=data_path,
        model_name=model_name,
        n_data=500,
        mode=mode,
        plot=plot,
        simulate=simulate,
        simulated_params_path='data/synthetic/parameters_d500.csv',
        perform_optimization=perform_optimization,
        normalized_params_path='data/synthetic/Y_test_d500.csv',
        filter=filter
    )
    end = time.time()

    print(f"Execution time: {end - start} seconds.")
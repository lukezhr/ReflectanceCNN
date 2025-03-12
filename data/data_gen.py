"""
This module contains the functions for generating the synthetic data for the model (both for training and testing).
"""
import os
import time
import logging
import numpy as np
from scipy.stats import uniform
from pyDOE import lhs
from reflectance.TaucLorentz import TL_nk
from reflectance.ThinFilmClasses import ThinFilmLayer, ThinFilmLayerTL, ThinFilmSystem
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_project_root():
    """
    Traverse upward from the directory where this file is located until a directory containing
    'setup.py' is found. This directory is assumed to be the project root.
    """
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.exists(os.path.join(current, 'setup.py')):
            return current
        new_current = os.path.dirname(current)
        if new_current == current:
            # We reached the filesystem root without finding setup.py.
            raise RuntimeError("Project root not found. Make sure 'setup.py' is present in your project root.")
        current = new_current


def sample_params_lhs(n_samples, param_ranges):
    """Generate Latin Hypercube Samples for the given parameter ranges."""
    lhs_samples = lhs(5, samples=n_samples)
    scaled_samples = np.array([uniform(loc=low, scale=high - low).ppf(lhs_samples[:, i])
                               for i, (low, high) in enumerate(param_ranges)]).T
    G_samples = np.array([uniform(loc=0, scale=np.sqrt(2) * e_0).rvs()
                          for e_0 in scaled_samples[:, 1]])
    scaled_samples = np.insert(scaled_samples, 2, G_samples, axis=1)
    return scaled_samples


def normalize(samples):
    """Normalize samples to the range [0, 1]."""
    epsilon = 1e-8 # avoid division by zero
    return (samples - samples.min(axis=0)) / (samples.max(axis=0) - samples.min(axis=0) + epsilon)


def calculate_reflectance(multilayer, params, wavelengths):
    """Calculate reflectance for a given set of parameters."""
    R = []
    for param_set in params:
        multilayer.layers[1].param = param_set[:5]
        multilayer.layers[1].thickness = param_set[5]
        r, _, _ = multilayer.calculate_RTA(wavelengths)
        R.append(r)
    return np.array(R)


def add_noise(data, noise_level):
    """Add Gaussian noise to the data."""
    return np.clip(data + np.random.normal(0, noise_level, data.shape), 0, 1)


def save_data(folder, reflectance, reflectance_noised, parameters, parameters_normalized, n, k, info):
    """Save generated data to the specified folder."""
    os.makedirs(folder, exist_ok=True)
    np.savetxt(os.path.join(folder, "reflectance_LHS.csv"), reflectance, delimiter=",")
    np.savetxt(os.path.join(folder, "reflectance_LHS_noised.csv"), reflectance_noised, delimiter=",")
    np.savetxt(os.path.join(folder, "parameters_LHS.csv"), parameters, delimiter=",")
    np.savetxt(os.path.join(folder, "parameters_LHS_normalized.csv"), parameters_normalized, delimiter=",")
    np.savetxt(os.path.join(folder, "n_LHS.csv"), n, delimiter=",")
    np.savetxt(os.path.join(folder, "k_LHS.csv"), k, delimiter=",")
    with open(os.path.join(folder, "info.txt"), "w") as f:
        f.write(info)
    logging.info(f"Data saved to {folder}")


def run_simulation(n_samples, param_ranges, left, right, n_points, noise_level, data_folder):
    """Run the simulation workflow."""
    wavelengths = np.linspace(left, right, n_points)

    # Sample parameters
    logging.info("Sampling parameters...")
    scaled_samples = sample_params_lhs(n_samples, param_ranges)
    normalized_samples = normalize(scaled_samples)

    # Calculate n and k
    logging.info("Calculating n and k...")
    _, _, n, k = TL_nk(wavelengths, scaled_samples)

    # Setup the multilayer system
    air = ThinFilmLayer("air", 1, 0, left, right)
    layer1 = ThinFilmLayerTL(100, [0, 4, 4, 0, 1], wavelengths)  # Placeholder parameters
    layer2 = ThinFilmLayer("sio2", 200, 10, left, right)
    substrate = ThinFilmLayer("c-Si", 1, 0, left, right)
    multilayer = ThinFilmSystem([air, layer1, layer2, substrate])

    # Calculate reflectance
    logging.info("Calculating reflectance...")
    reflectance = calculate_reflectance(multilayer, scaled_samples, wavelengths)
    reflectance_noised = add_noise(reflectance, noise_level)

    # Save results
    info = f"n_samples: {n_samples}, n_points: {n_points}, noise_level: {noise_level}, param_ranges: {param_ranges}"
    save_data(data_folder, reflectance, reflectance_noised, scaled_samples, normalized_samples, n, k, info)

    # Split data
    logging.info("Splitting data...")
    x_file_path = os.path.join(data_folder, "reflectance_LHS_noised.csv")
    y_file_path = os.path.join(data_folder, "parameters_LHS_normalized.csv")
    split_data(data_folder, x_file_path, y_file_path)
    logging.info("Data split and saved successfully.")


def split_data(output_dir, x_file_path, y_file_path, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Split the data into training, validation, and testing sets and save them.
    """
    # Ensure the split sizes are valid
    if not (0 < train_size < 1 and 0 <= val_size < 1 and 0 <= test_size < 1 and train_size + val_size + test_size == 1):
        raise ValueError(
            "Invalid split sizes. Ensure they sum to 1 and are in the range (0, 1).")

    # Load the datasets
    X = np.loadtxt(x_file_path, delimiter=",")
    Y = np.loadtxt(y_file_path, delimiter=",")

    # First, split the data into training and (temporary) testing datasets
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=1-train_size, random_state=42)

    # Adjust test_size for the secondary split because the remaining dataset is smaller
    secondary_test_size = test_size / (test_size + val_size)

    # Next, split the temporary testing datasets into validation and testing datasets
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=secondary_test_size, random_state=42)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save these datasets to their respective files
    np.savetxt(os.path.join(output_dir, "X_train.csv"), X_train, delimiter=",")
    np.savetxt(os.path.join(output_dir, "Y_train.csv"), Y_train, delimiter=",")
    np.savetxt(os.path.join(output_dir, "X_val.csv"), X_val, delimiter=",")
    np.savetxt(os.path.join(output_dir, "Y_val.csv"), Y_val, delimiter=",")
    np.savetxt(os.path.join(output_dir, "X_test.csv"), X_test, delimiter=",")
    np.savetxt(os.path.join(output_dir, "Y_test.csv"), Y_test, delimiter=",")


if __name__ == "__main__":
    # Simulation parameters
    n_samples = 100 # number of samples. 100 is just to test the code, in practice it should be 50000 to reach a good accuracy
    thickness_ranges = [(10, 150)]
    left, right = 400, 800 # wavelength range
    n_points = 500 # dimension of the data
    noise_level = 0.01

    # Define the data folder
    project_root = get_project_root()
    data_folder = os.path.join(project_root, "data", "synthetic")

    for thickness_range in thickness_ranges:
        param_ranges = [
            (1e-6, 100),  # A
            (3.2, 7),      # E_0
            (0.5, 3.2),     # E_g
            (1, 10),      # e_inf
            thickness_range  # Thickness
        ]
        logging.info(f"Generating data for thickness range: {thickness_range}")
        start_total = time.time()
        run_simulation(n_samples, param_ranges, left, right, n_points, noise_level, data_folder)
        end_total = time.time()
        logging.info(f"Total time: {(end_total - start_total) / 60:.2f} mins")
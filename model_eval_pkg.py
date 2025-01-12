from scipy.optimize import least_squares
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from ThinFilmClasses import ThinFilmLayer, ThinFilmLayerTL, ThinFilmSystem
from TaucLorentz import TL_nk
import matplotlib.pyplot as plt
import time
import pandas as pd


class ReflectanceCNN(nn.Module):
    def __init__(self, input_channels, input_length):
        super(ReflectanceCNN, self).__init__()
        kernel_size = [150, 100, 75, 50, 15, 5]  # 6 conv layers
        # num of filters at each conv layer
        channels = [64, 64, 64, 64, 64, 64]
        # Conv1D + BatchNorm + MaxPool
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=channels[0], kernel_size=kernel_size[0], stride=1, padding=(
            kernel_size[0] - 1) // 2)  # 'same' padding
        self.bn1 = nn.BatchNorm1d(num_features=channels[0])
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1],
                               kernel_size=kernel_size[1], stride=1, padding=(kernel_size[1] - 1) // 2)
        self.bn2 = nn.BatchNorm1d(num_features=channels[1])
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2],
                               kernel_size=kernel_size[2], stride=1, padding=(kernel_size[2] - 1) // 2)
        self.bn3 = nn.BatchNorm1d(num_features=channels[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3],
                               kernel_size=kernel_size[3], stride=1, padding=(kernel_size[3] - 1) // 2)
        self.bn4 = nn.BatchNorm1d(num_features=channels[3])
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.conv5 = nn.Conv1d(in_channels=channels[3], out_channels=channels[4],
                               kernel_size=kernel_size[4], stride=1, padding=(kernel_size[4] - 1) // 2)
        self.bn5 = nn.BatchNorm1d(num_features=channels[4])
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.conv6 = nn.Conv1d(in_channels=channels[4], out_channels=channels[5],
                               kernel_size=kernel_size[5], stride=1, padding=(kernel_size[5] - 1) // 2)
        self.bn6 = nn.BatchNorm1d(num_features=channels[5])
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)
        
        self.conv7 = nn.Conv1d(in_channels=channels[5], out_channels=channels[6], kernel_size=kernel_size[6], stride=1, padding=(kernel_size[6] - 1) // 2)
        self.bn7 = nn.BatchNorm1d(num_features=channels[6])
        self.pool7 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc_size = self._get_conv_output(input_channels, input_length)

        # Fully Connected Layers + Dropout
        self.fc1 = nn.Linear(self.fc_size, 3000)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(3000, 1200)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1200, 300)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 6)

    def _get_conv_output(self, input_channels, input_length):
        # Dummy pass to get the output size
        input = torch.autograd.Variable(
            torch.rand(1, input_channels, input_length))
        output = self._forward_features(input)
        n_size = output.data.reshape(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        # x = self.pool7(F.relu(self.bn7(self.conv7(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(self.fc3(x))
        x = self.fc4(x)
        return x


def data_load_and_prep(wavelength_path, reflectance_path, n_data=1000, min_wavelength=440, max_wavelength=800):
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
    # Load the data
    wavelengths = np.loadtxt(wavelength_path, delimiter=',')
    reflectances = np.loadtxt(reflectance_path, delimiter=',')
    print("wavelengths shape: ", wavelengths.shape)
    print("reflectances shape: ", reflectances.shape)
    
    # Generate the new wavelengths array
    new_wavelengths = np.linspace(min_wavelength, max_wavelength, n_data)
    print("new_wavelengths shape: ", new_wavelengths.shape)

    # Check if reflectances is a 1D array and reshape it to 2D if necessary
    if reflectances.ndim == 1:
        reflectances = reflectances.reshape(-1, 1)

    # Initialize an array to store the interpolated reflectances
    # new_reflectances = np.zeros((n_data, reflectances.shape[1]))
    new_reflectances = np.zeros((reflectances.shape[0], n_data))
    
    print("wavelegnths shape: ", wavelengths.shape)
    print("reflectances shape: ", reflectances.shape)
    print("new_wavelengths shape: ", new_wavelengths.shape)
    print("new_reflectances shape: ", new_reflectances.shape)

    # Perform the linear interpolation for each column of reflectances
    # for i in range(reflectances.shape[1]):
    #     new_reflectances[:, i] = np.interp(new_wavelengths, wavelengths, reflectances[:, i])
    for i in range(reflectances.shape[0]):
        new_reflectances[i, :] = np.interp(new_wavelengths, wavelengths, reflectances[i, :])
    print("new_reflectances shape: ", new_reflectances.shape)

    # If the original reflectances were a 1D array, reshape the result back to 1D
    if new_reflectances.shape[1] == 1:
        new_reflectances = new_reflectances.ravel()
        print("new_reflectances shape: ", new_reflectances.shape)

    return new_wavelengths, new_reflectances


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


def process_data(path, left=400, right=800, uncertainty_threshold=0.02):
    """
    Read and crop the data within [left, right], and return cropped data, left, right.

    **Parameters:**
    path : str
        path of the data to be processed.
    uncertainty_threshold : float
        uncertainty threshold below which the data is selected.
    left: float
        Manually set the lower bound of the cropped data.
    right: float
        Manually set the upper bound of the cropped data.
    **Returns:**
    data : panda DataFrame
        It returns the cropped data.
    left : float
        The lower bound of the cropped data based on uncertainty threshold.
    right : float
        The upper bound of the cropped data based on uncertainty threshold.
    """
    # Load the data
    data = pd.read_csv(path, names=[
        '# lambda', 'reflectance', 'uncertainty', 'raw', 'dark', 'reference', 'fit'], skiprows=13)
    data = data.rename(columns={'# lambda': 'wavelength'})

    # Filter data based on uncertainty
    if uncertainty_threshold != None and uncertainty_threshold > 0:
        uncertainty_filtered_data = data[data['uncertainty']
                                         <= uncertainty_threshold]

        # Determine the smallest and largest wavelengths
        left = uncertainty_filtered_data['wavelength'].min()
        right = uncertainty_filtered_data['wavelength'].max()

        print("left = ", left)
        print("right = ", right)

        # Filter data based on determined wavelength boundaries
        data = data[(data['wavelength'] >= left) &
                    (data['wavelength'] <= right)]
    else:
        data = data[(data['wavelength'] >= left) &
                    (data['wavelength'] <= right)]

    return data, left, right


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


def mse(predicted, true):
    return mean_squared_error(true, predicted)
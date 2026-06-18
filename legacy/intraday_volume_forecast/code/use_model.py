import numpy as np
import pandas as pd
from fit import *
from kalman import *
from plot import *
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def decompose_volume(purpose, model, data, burn_in_days=0):
    """
    Decomposes the intraday volume into components: daily, seasonal, and dynamic.
    Args:
        purpose (str): "analysis" or "forecast", indicating the purpose of the decomposition.
        model (object): The fitted model object (e.g., from a `fit_volume()` function).
        data (np.array or pd.DataFrame): Intraday volume data.
        burn_in_days (int): Number of initial days to discard for forecasting.

    Returns:
        dict: A dictionary with the original signal, smooth/forecast signal, components, and errors.
    """
    if purpose.lower() == "analysis":
        return smooth_volume_model(data, model)
    elif purpose.lower() == "forecast":
        return forecast_volume_model(data, model, burn_in_days)
    else:
        raise ValueError("Invalid purpose. Choose either 'analysis' or 'forecast'.")

def smooth_volume_model(data, volume_model):
    """
    Perform Kalman smoothing to get the daily, seasonal, and dynamic components.
    Args:
        data (np.array or pd.DataFrame): Intraday volume data.
        volume_model (object): The fitted volume model.
    
    Returns:
        dict: Contains original signal, smoothed signal, components, and error metrics.
    """
    # Validate data type
    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise ValueError("Data must be a numpy array or pandas DataFrame.")
    
    data = clean_data(data)  # Implement this to clean/preprocess data if necessary

    # Assuming model validation and Kalman filter processing occurs here
    if not volume_model.converged:
        raise ValueError("Model parameters are not optimally fitted.")

    # Apply Kalman filter (UNISS)
    smoothed_data = kalman_smooth(data, volume_model)

    # Decompose the signal
    smooth_components = {
        'daily': smoothed_data['daily'],
        'dynamic': smoothed_data['dynamic'],
        'seasonal': smoothed_data['seasonal']
    }

    smooth_signal = smooth_components['daily'] * smooth_components['dynamic'] * smooth_components['seasonal']
    residual = data / smooth_signal

    # Compute error metrics
    errors = {
        'mae': mean_absolute_error(data, smooth_signal),
        'mape': mean_absolute_percentage_error(data, smooth_signal),
        'rmse': np.sqrt(mean_squared_error(data, smooth_signal))
    }

    return {
        'original_signal': data,
        'smooth_signal': smooth_signal,
        'smooth_components': smooth_components,
        'error': errors
    }

def forecast_volume_model(data, volume_model, burn_in_days=0):
    """
    Forecast one-bin-ahead intraday volume.
    Args:
        data (np.array or pd.DataFrame): Intraday volume data.
        volume_model (object): The fitted volume model.
        burn_in_days (int): Number of initial days to discard for forecasting.

    Returns:
        dict: Contains forecast signal, components, and error metrics.
    """
    # Validate data type
    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise ValueError("Data must be a numpy array or pandas DataFrame.")
    
    data = clean_data(data)  # Implement this to clean/preprocess data if necessary
    
    if burn_in_days > data.shape[1]:
        raise ValueError("Burn-in days must be smaller than the number of columns in data.")

    # Forecast using Kalman filter
    forecast_data = kalman_forecast(data, volume_model, burn_in_days)

    forecast_components = {
        'daily': forecast_data['daily'],
        'dynamic': forecast_data['dynamic'],
        'seasonal': forecast_data['seasonal']
    }

    forecast_signal = forecast_components['daily'] * forecast_components['dynamic'] * forecast_components['seasonal']
    residual = data / forecast_signal

    # Compute error metrics
    errors = {
        'mae': mean_absolute_error(data, forecast_signal),
        'mape': mean_absolute_percentage_error(data, forecast_signal),
        'rmse': np.sqrt(mean_squared_error(data, forecast_signal))
    }

    return {
        'original_signal': data,
        'forecast_signal': forecast_signal,
        'forecast_components': forecast_components,
        'error': errors
    }

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import numpy as np

def scale_data(data, method='standard', scaler_columns=None):
    """
    Scale the specified numeric columns in the input DataFrame using the chosen scaling method.

    Parameters:
    data (DataFrame): Input DataFrame.
    method (str): Scaling method to use. Options: 'standard', 'minmax', 'robust'.
    scaler_columns (list): List of column names to be scaled. Default is None (all numeric columns),
    additionaly if None is specified, and binary columns are present, they will be excluded from scaling.

    Returns:
    DataFrame: Scaled DataFrame.
    """
    # Check if specified columns to scale are numeric and non-binary
    if scaler_columns:
        non_binary_numeric_columns = [col for col in scaler_columns if col in data.select_dtypes(include=[np.number]).columns and len(data[col].unique()) > 2]
        if not non_binary_numeric_columns:
            raise ValueError("No valid non-binary numeric columns specified for scaling.")

    # Select the appropriate scaler based on the chosen method
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaling method. Valid methods are: 'standard', 'minmax', 'robust'")

    # Select only the specified columns for scaling
    if scaler_columns:
        data_to_scale = data[scaler_columns]
    else:
        # Default to all numeric non-binary columns if none are specified
        data_to_scale = data.select_dtypes(include=[np.number]).drop(columns=[col for col in data.columns if len(data[col].unique()) == 2])

    # Scale the selected columns
    scaled_data = data.copy()
    scaled_data[data_to_scale.columns] = scaler.fit_transform(data_to_scale)
    
    return scaled_data

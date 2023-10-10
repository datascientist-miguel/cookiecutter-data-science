from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

def imputation(data, method='mean', imp_columns=None, n_neighbors=None, interp_method=None):
    """
    Perform missing value imputation in a dataset using various methods.

    Available imputation methods:
        - 'mode': Impute with mode for categorical variables, not applicable for numeric.
        - 'mean': Impute with mean for numeric variables.
        - 'median': Impute with median for numeric variables.
        - 'knn': Impute using k-nearest neighbors algorithm.
        - 'interpolation': Impute using interpolation methods.

    For 'interpolation' method, supported interpolation methods are:
        'linear', 'time', 'index', 'values', 'nearest', 'zero'.

    Parameters:
    data (DataFrame): Input DataFrame.
    method (str): Imputation method to use. Default is 'mean'.
    imp_columns (list): List of column names to be imputed. Default is None (all columns).
    n_neighbors (int): Number of neighbors for KNN imputation. Required if method is 'knn'.
    interp_method (str): Interpolation method if method is 'interpolation'. Default is None.

    Returns:
    DataFrame: DataFrame with imputed values.
    """
    # Check if valid imputation method is chosen
    valid_methods = ['mode', 'mean', 'median', 'knn', 'interpolation']
    if method not in valid_methods:
        raise ValueError("Invalid imputation method. Valid methods are: 'mode', 'mean', 'median', 'knn', 'interpolation'.")

    # Create a copy of the original dataset
    imputed_data = data.copy()

    # Select columns to be imputed
    if imp_columns:
        non_numeric_columns = [col for col in imp_columns if col not in data.select_dtypes(include=[np.number]).columns]
        if non_numeric_columns and method != 'mode':
            raise ValueError("Selected columns for imputation must be numeric.")

    if imp_columns is None:
        # Default to all numeric columns if none are specified
        imp_columns = data.select_dtypes(include=[np.number]).columns

    # Perform imputation based on the selected method
    if method == 'mode':
        # Impute using mode for categorical variables
        for col in imp_columns:
            if col in data.select_dtypes(include=[np.object]).columns:
                imputed_data[col].fillna(imputed_data[col].mode()[0], inplace=True)
    elif method == 'mean':
        # Impute using mean for numeric variables
        imputed_data[imp_columns] = imputed_data[imp_columns].fillna(imputed_data[imp_columns].mean())
    elif method == 'median':
        # Impute using median for numeric variables
        imputed_data[imp_columns] = imputed_data[imp_columns].fillna(imputed_data[imp_columns].median())
    elif method == 'knn':
        # Impute using k-nearest neighbors
        if n_neighbors is None:
            raise ValueError("Number of neighbors (n_neighbors) must be specified for KNN imputation.")
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data[imp_columns] = imputer.fit_transform(imputed_data[imp_columns])
    elif method == 'interpolation':
        # Impute using interpolation
        if interp_method is None:
            raise ValueError("Interpolation method must be specified for interpolation imputation.")
        imputed_data[imp_columns] = imputed_data[imp_columns].interpolate(method=interp_method)

    return imputed_data
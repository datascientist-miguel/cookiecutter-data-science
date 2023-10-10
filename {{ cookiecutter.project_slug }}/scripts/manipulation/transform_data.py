import numpy as np
from scipy.stats import boxcox, yeojohnson

def transform_data(data, method='log', excluded_columns=[], lambda_value=None):
    """
    Transform the data according to the selected method.

    Parameters
    ----------
    data : DataFrame
        The data to transform.
    method : str, optional (default "log")
        The transformation method to apply. Options are:
        - "log": natural logarithm transformation
        - "sqrt": square root transformation
        - "reciprocal": reciprocal transformation
        - "boxcox": Box-Cox transformation
        - "yeojohnson": Yeo-Johnson transformation
    excluded_columns : list, optional (default [])
        List of columns to exclude from transformation.
    lambda_value : float, optional (default None)
        Lambda value to use in Box-Cox transformation instead of automatically obtained value.
        For Box-Cox, typical lambda values and corresponding transformations are:
        - lambda = 0: log transformation
        - lambda = 0.5: square root transformation
        - lambda = 1: no transformation (identity)
        - lambda = -1: reciprocal transformation
        - lambda = >1: Linear transformation

    Returns
    -------
    DataFrame
        The transformed data.
    """
    transformed_data = data.copy()

    valid_methods = ['log', 'sqrt', 'reciprocal', 'boxcox', 'yeojohnson']

    if method not in valid_methods:
        raise ValueError("Invalid transformation method. Choose from: 'log', 'sqrt', 'reciprocal', 'boxcox', 'yeojohnson'.")

    # Iterate through columns and transform only non-binary numeric ones
    for col in transformed_data.select_dtypes(include=[np.number]).columns:
        if col in excluded_columns:
            continue

        # Skip binary numeric columns
        if len(transformed_data[col].unique()) == 2:
            print(f"Skipping transformation for binary numeric column: {col}")
            continue

        # Explicitly cast column to float before transformation
        transformed_data[col] = transformed_data[col].astype(float)

        if method == 'log':
            transformed_data[col] = np.log(transformed_data[col])
        elif method == 'sqrt':
            transformed_data[col] = np.sqrt(transformed_data[col])
        elif method == 'reciprocal':
            transformed_data[col] = np.reciprocal(transformed_data[col])
        elif method == 'boxcox':
            if np.all(transformed_data[col] > 0):
                if lambda_value is not None:
                    transformed_data[col] = boxcox(transformed_data[col], lmbda=lambda_value)
                else:
                    transformed_data[col], _ = boxcox(transformed_data[col])
            else:
                print(f"Warning: Box-Cox transformation cannot be performed on column '{col}' as it contains non-positive values.")
        elif method == 'yeojohnson':
            positive_values = transformed_data[col][transformed_data[col] > 0]
            transformed_values = yeojohnson(positive_values)[0]
            transformed_data.loc[transformed_data[col] > 0, col] = transformed_values

    return transformed_data
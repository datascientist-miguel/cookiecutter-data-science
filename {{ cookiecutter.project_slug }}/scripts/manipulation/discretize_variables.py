import pandas as pd
import numpy as np

def discretize_variable(data, column, method='equal_width', bins=5):
    """
    Discretize a numerical variable into specified bins or quantiles.

    Parameters:
    data (DataFrame): Input DataFrame.
    column (str): Name of the numerical column to be discretized.
    method (str): Discretization method. Options: 'equal_width', 'equal_frequency'.
                  Default is 'equal_width'.
    bins (int or list): Number of bins (for 'equal_width') or list of bin edges (for custom bins).
                        Default is 5.

    Returns:
    DataFrame: DataFrame with the discretized variable.

    Raises:
    ValueError: If the specified column is not numerical.
    ValueError: If the specified method is invalid.
    ValueError: If the number of bins is not a positive integer.
    """

    # Check if the specified column is numerical
    if data[column].dtype not in [np.number]:
        raise ValueError(f"The column '{column}' is not numerical.")

    # Handle discretization based on the chosen method
    if method == 'equal_width':
        # Discretize into equal-width bins
        try:
            # Convert bins to an integer if possible
            bins = int(bins)
        except ValueError:
            raise ValueError("Number of bins must be a positive integer.")

        if bins <= 0:
            raise ValueError("Number of bins must be a positive integer.")

        data[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=False, duplicates='drop')
    
    elif method == 'equal_frequency':
        # Discretize into equal-frequency bins (quantiles)
        data[f'{column}_binned'] = pd.qcut(data[column], q=bins, labels=False, duplicates='drop')

    else:
        raise ValueError("Invalid discretization method. Valid methods are: 'equal_width', 'equal_frequency'.")

    return data


import pandas as pd
import numpy as np

def replace_with_nan(data, columns, value):
    """
    Replace the specified value in the specified columns of the DataFrame with NaN and return a new DataFrame.

    Args:
    data (pandas DataFrame): Input DataFrame.
    columns (list of str): List of column names to modify.
    value (int or float): Value to replace with NaN.

    Returns:
    pandas DataFrame: New DataFrame with the modified values.
    """
    try:
        # Create a copy of the original DataFrame
        new_data = data.copy()

        # Replace specified values with NaN in each column
        for col in columns:
            new_data.loc[new_data[col] == value, col] = np.nan

        return new_data

    except KeyError as e:
        raise KeyError("One or more specified columns do not exist in the DataFrame.") from e

    except Exception as e:
        raise Exception("An error occurred: {}".format(str(e))) from e
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def detect_multicollinearity(data, exclude_vars=[]):
    """
    Detect multicollinearity in a dataset using Variance Inflation Factor (VIF).

    Parameters:
    data (DataFrame): Input DataFrame.
    exclude_vars (list): List of variables to exclude from the VIF calculation (optional).

    Returns:
    DataFrame: DataFrame containing VIF values and multicollinearity assessment.
    """
    try:
        # Filter out variables to be excluded from the VIF calculation
        features = [col for col in data.columns if col not in exclude_vars]

        # Filter out variables that are not numeric
        numeric_features = [col for col in features if np.issubdtype(data[col].dtype, np.number)]

        # Calculate VIF for each numeric feature
        vif_values = [variance_inflation_factor(data[numeric_features].values, i) for i in range(len(numeric_features))]

        # Create a DataFrame to store the results
        vif_df = pd.DataFrame({'Feature': numeric_features, 'VIF': vif_values})

        # Assess multicollinearity based on a threshold (e.g., VIF > 10)
        vif_df['Multicollinearity'] = ['Yes' if vif > 10 else 'No' for vif in vif_df['VIF']]

        return vif_df

    except Exception as e:
        # Handle any exceptions and print an error message
        print("An error occurred:", str(e))
        return None

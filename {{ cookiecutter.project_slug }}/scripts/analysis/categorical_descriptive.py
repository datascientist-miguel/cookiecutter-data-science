import pandas as pd
import numpy as np

def categorical_descriptives(df):
    """
    Calculate descriptive statistics for categorical columns in a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: DataFrame containing descriptive statistics for categorical columns.
    """
    # Initialize empty lists to store statistics
    var_names = []
    categories = []
    frequencies = []
    proportions = []
    cardinalities = []
    modes = []
    entropies = []
    gini_indices = []
    missing_percentages = []

    # Loop through each column in the DataFrame
    for col in df.columns:
        # Check if the column is categorical
        if df[col].dtype == 'object':
            # Calculate statistics
            value_counts = df[col].value_counts()
            total_counts = len(df[col])

            # Cardinality
            cardinality = len(value_counts)

            # Mode (most frequent category)
            mode = value_counts.idxmax()

            # Entropy
            entropy = -sum((value_counts / total_counts) * np.log2(value_counts / total_counts))

            # Gini Index
            gini_index = 1 - sum((value_counts / total_counts) ** 2)

            # Percentage of missing values
            missing_percentage = (df[col].isnull().sum() / total_counts) * 100

            # Append the statistics to respective lists
            var_names.extend([col] * cardinality)
            categories.extend(value_counts.index.tolist())
            frequencies.extend(value_counts.tolist())
            proportions.extend((value_counts / total_counts).tolist())
            cardinalities.extend([cardinality] * cardinality)
            modes.extend([mode] * cardinality)
            entropies.extend([entropy] * cardinality)
            gini_indices.extend([gini_index] * cardinality)
            missing_percentages.extend([missing_percentage] * cardinality)

    # Create a DataFrame with the calculated statistics
    df_descriptives = pd.DataFrame({
        "Variable": var_names,
        "Category": categories,
        "Frequency": frequencies,
        "Proportion": proportions,
        "Cardinality": cardinalities,
        "Mode": modes,
        "Entropy": entropies,
        "Gini Index": gini_indices,
        "Missing Percentage": missing_percentages
    })

    return df_descriptives


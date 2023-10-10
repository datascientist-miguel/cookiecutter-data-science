import numpy as np
import pandas as pd

def numerical_descriptive(df):
    """
    Calculate descriptive statistics for numerical columns in a DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame.

    Returns:
    DataFrame: DataFrame containing descriptive statistics for numerical columns.
    """
    # Initialize empty lists to store statistics
    var_names = []
    min_values = []
    max_values = []
    range_values = []
    mean_values = []
    median_values = []
    variance_values = []
    std_dev_values = []
    q1_values = []
    q3_values = []
    iqr_values = []
    cv_values = []  # Updated: to store coefficient of variation as a percentage
    skewness_values = []
    kurtosis_values = []
    missing_values = []

    # Loop through each column in the DataFrame
    for col in df.columns:
        # Check if the column is numerical
        if np.issubdtype(df[col].dtype, np.number):
            var_names.append(col)

            # Calculate statistics
            min_val = np.min(df[col])
            max_val = np.max(df[col])
            range_val = max_val - min_val
            variance_val = np.var(df[col], ddof=1)
            std_dev_val = np.std(df[col], ddof=1)
            q1_val = np.percentile(df[col], 25)
            q3_val = np.percentile(df[col], 75)
            iqr_val = q3_val - q1_val
            mean_val = np.mean(df[col])
            median_val = np.median(df[col])
            cv_val = round((std_dev_val / mean_val) * 100, 1) if mean_val != 0 else np.nan  # Round CV to 1 decimal place
            skewness_val = df[col].skew()
            kurtosis_val = df[col].kurtosis()

            # Count missing values
            missing_val_count = df[col].isnull().sum()

            # Append values to respective lists
            min_values.append(min_val)
            max_values.append(max_val)
            range_values.append(range_val)
            variance_values.append(variance_val)
            std_dev_values.append(std_dev_val)
            q1_values.append(q1_val)
            q3_values.append(q3_val)
            iqr_values.append(iqr_val)
            mean_values.append(mean_val)
            median_values.append(median_val)
            cv_values.append(cv_val)
            skewness_values.append(skewness_val)
            kurtosis_values.append(kurtosis_val)
            missing_values.append(missing_val_count)

        else:
            print(f"Skipping non-numeric column: {col}")

    # Create a DataFrame with the calculated statistics
    df_descriptives = pd.DataFrame({
        "Variable": var_names,
        "Min": min_values,
        "Max": max_values,
        "Range": range_values,
        "Mean": mean_values,
        "Median": median_values,
        "Variance": variance_values,
        "Standard Deviation": std_dev_values,
        "Q1": q1_values,
        "Q3": q3_values,
        "Interquartile Range": iqr_values,
        "Coefficient of Variation (%)": cv_values,  # Updated: Show CV as percentage
        "Skewness": skewness_values,
        "Kurtosis": kurtosis_values,
        "Missing Values": missing_values
    })

    return df_descriptives
import pandas as pd
from imblearn.over_sampling import SMOTE

def oversample(X, y, sampling_strategy='auto', k_neighbors=5, random_state=42):
    """
    Apply the SMOTE (Synthetic Minority Over-sampling Technique) oversampling technique to address class imbalance.

    Parameters:
    X (DataFrame): Feature dataset.
    y (Series): Target variable.
    sampling_strategy (float, str, dict, or callable, optional): Desired ratio of the number of samples in the minority
        class over the majority class after resampling.
    k_neighbors (int, optional): Number of nearest neighbors to use to construct synthetic samples.
    random_state (int, optional): Controls the random seed for reproducibility.

    Returns:
    DataFrame: Oversampled dataset using SMOTE.

    Raises:
    ValueError: If the input data is invalid.
    """

    # Verify that the input data is a DataFrame and Series
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise ValueError("X must be a DataFrame, and y must be a Series.")

    # Verify that X and y have the same number of rows
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    # Apply SMOTE to the dataset
    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    except ValueError as e:
        raise ValueError("SMOTE could not be applied. Ensure that each class has at least 2 samples.") from e

    # Convert X_resampled to a DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    # Add the target variable to the DataFrame
    X_resampled_df[y.name] = y_resampled

    return X_resampled_df

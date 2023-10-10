import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def sampling_data(data, sampling_type='random', sampling_fraction=0.1, random_state=42, n_splits=1, stratify_by=None):
    """
    This function is used to sample data from the dataset
    :param data: input dataset
    :param sampling_type: type of sampling, random or stratified
    :param sampling_fraction: fraction of data to be sampled
    :param random_state: random state for reproducibility
    :param n_splits: number of splits for stratified sampling
    :param stratify_by: column name to be used for stratified sampling
    :return: sampled data
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame.")

        if sampling_type not in ['random', 'stratified']:
            raise ValueError("Invalid sampling type. Use 'random' or 'stratified'.")

        if not 0 < sampling_fraction <= 1.0:
            raise ValueError("Sampling fraction should be between 0 and 1.")

        if sampling_type == 'stratified':
            if not isinstance(n_splits, int) or n_splits < 1:
                raise ValueError("Number of splits should be a positive integer.")

            if not isinstance(stratify_by, str) or stratify_by not in data.columns:
                raise ValueError("Invalid column name for stratified sampling.")

            sampling_stratified = 1.0 - sampling_fraction
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=sampling_stratified, random_state=random_state)
            for train_index, _ in sss.split(data, data[stratify_by]):
                data_sampling = data.iloc[train_index]
        else:
            data_sampling = data.sample(frac=sampling_fraction, random_state=random_state)

        return data_sampling

    except ValueError as ve:
        print("Error: ", str(ve))
        return None

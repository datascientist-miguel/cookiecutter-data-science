from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import pandas as pd

def categorical_encoding(data, columns, method='onehot'):
    """
    Transform categorical variables into numerical representations using specified methods.

    Available methods:
    - 'onehot': One-hot encoding
    - 'ordinal': Ordinal encoding
    - 'binary': Binary encoding
    - 'count': Count encoding
    - 'label': Label encoding

    Parameters:
    data (DataFrame): Input DataFrame.
    columns (list): List of column names to be encoded.
    method (str): Encoding method. Default is 'onehot'.

    Returns:
    DataFrame: Transformed DataFrame.
    """

    # Create a copy of the original dataset
    transformed_data = data.copy()

    # Select the transformation method
    if method == 'onehot':
        # One-hot encoding
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        for column in columns:
            encoded_columns = pd.DataFrame(
                onehot_encoder.fit_transform(data[column].values.reshape(-1, 1)),
                columns=[f"{column}_{val}" for val in onehot_encoder.categories_[0]]
            )

            transformed_data = pd.concat(
                [transformed_data.drop(column, axis=1), encoded_columns],
                axis=1
            )

    elif method == 'ordinal':
        # Ordinal encoding
        ordinal_encoder = OrdinalEncoder()

        for column in columns:
            transformed_column = ordinal_encoder.fit_transform(data[column].values.reshape(-1, 1))
            transformed_data[column] = transformed_column

    elif method == 'binary':
        # Binary encoding
        for column in columns:
            binary_encoded_column = pd.get_dummies(data[column], prefix=column, drop_first=False)
            binary_encoded_column = binary_encoded_column.astype('int64')
            transformed_data = pd.concat([transformed_data, binary_encoded_column], axis=1)
            transformed_data = transformed_data.drop(column, axis=1)

    elif method == 'count':
        # Count encoding
        for column in columns:
            frequencies = data[column].value_counts().to_dict()
            transformed_data[column + '_count'] = data[column].map(frequencies)
            transformed_data = transformed_data.drop(column, axis=1)

    elif method == 'label':
        # Label encoding
        label_encoder = LabelEncoder()

        for column in columns:
            transformed_data[column] = label_encoder.fit_transform(data[column])

    else:
        raise ValueError("Invalid encoding method. Valid methods are: 'onehot', 'ordinal', 'binary', 'count', 'label'.")

    return transformed_data

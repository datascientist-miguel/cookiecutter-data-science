import pandas as pd
import numpy as np

def drop_outliers(data, threshold=1.5):
    
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    numerical_columns = [col for col in numerical_columns if len(data[col].unique()) > 2]

    for col in numerical_columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
    return data
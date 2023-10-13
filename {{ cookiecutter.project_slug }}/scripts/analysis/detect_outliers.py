import numpy as np
import pandas as  pd

def detect_outliers(data, threshold=1.5):
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    numerical_columns = [col for col in numerical_columns if len(data[col].unique()) > 2]
    outliers_count = pd.Series(0, index=numerical_columns)
    
    for col in numerical_columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers_count[col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
    print(f'total outliers: {outliers_count.sum()} with Tukey method')
    return outliers_count
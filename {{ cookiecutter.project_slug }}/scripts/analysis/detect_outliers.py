import numpy as np
# Detecta valores atípicos en un conjunto de datos utilizando el método de los umbrales de Tukey.

def detect_outliers(data, exclude_columns=[], threshold=1.5):

    outliers = {}
    for col in data.columns:
        if col in exclude_columns:
            continue
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_count = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].count()
        outliers[col] = outlier_count
    return outliers
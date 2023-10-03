import pandas as pd
"""
    Elimina las filas con valores faltantes en el conjunto de datos.
    """
def remove_valores_nulos(data):

    # Cuenta el número de filas con valores faltantes
    num_missing_rows = data.isnull().sum(axis=1)
    
    # Obtiene un DataFrame que contiene solo las filas sin valores faltantes
    data_no_missing = data[num_missing_rows == 0]
    
    # Devuelve el DataFrame sin valores faltantes
    return pd.dropna(data_no_missing)
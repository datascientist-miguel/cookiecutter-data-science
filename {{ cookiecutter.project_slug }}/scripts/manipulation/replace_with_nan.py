import pandas as pd
import numpy as np

def reemplazar_con_nan(data, cols, valor):
    """
    Reemplaza el valor en el índice especificado en las columnas especificadas
    del DataFrame con NaN y devuelve un nuevo DataFrame.
    
    Args:
    data: pandas DataFrame
    cols: lista de cadenas - columnas a modificar
    valor: int o float - valor a reemplazar con NaN
    
    Returns:
    pandas DataFrame - nuevo DataFrame con los valores modificados
    """
    nuevo_data = data.copy() # crear una copia del DataFrame original
    for col in cols:
        nuevo_data.loc[nuevo_data[col] == valor, col] = np.nan # reemplazar valores con NaN en cada columna
    
    return nuevo_data
import pandas as pd
import numpy as np

def remove_outliers(data, exclude=[], threshold=1.5):
    """
    Elimina los valores atípicos de un DataFrame de pandas utilizando el método de Tukey.
    
    Parámetros:
    -----------
    data: pandas DataFrame
        Datos que se van a limpiar.
    exclude: lista de str
        Lista de nombres de columnas que se excluyen de la detección de valores atípicos.
    threshold: float
        Umbral para la detección de valores atípicos.
        
    Retorna:
    --------
    pandas DataFrame
        Datos sin valores atípicos.
    """
    
    # Crear una copia de los datos de entrada
    data_cleaned = data.copy()
    
    # Recorrer las columnas de los datos
    for col in data.columns:
        
        # Comprobar si la columna está en la lista de exclusiones
        if col in exclude:
            continue
        
        # Calcular los cuartiles y el rango intercuartílico (IQR)
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        IQR = Q3 - Q1
        
        # Calcular los límites superior e inferior para los valores atípicos
        limite_inferior = Q1 - threshold * IQR
        limite_superior = Q3 + threshold * IQR
        
        # Eliminar los valores atípicos de la columna
        data_cleaned = data_cleaned[(data_cleaned[col] >= limite_inferior) & (data_cleaned[col] <= limite_superior)]
    
    return data_cleaned
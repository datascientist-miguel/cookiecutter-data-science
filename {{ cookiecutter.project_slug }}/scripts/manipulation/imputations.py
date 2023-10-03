from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

def imputation(data, metodo='moda', excluidas=[], n_neighbors=None, metodo_interp=None):
    """
    Función que realiza imputación de valores faltantes en un conjunto de datos.
    Los métodos disponibles son 'moda', 'media', 'mediana', 'knn' e 'interpolación'.
    Tener en cuenta que la imputacion de variables categóricas se realiza con 'moda'
    
    Los métodos de interpolación disponibles en pandas son:
        linear (Interpolación lineal)
        time (Interpolación basada en el tiempo)
        index (Interpolación basada en el índice)
        values (Interpolación basada en los valores)
        nearest (Interpolación basada en el valor más cercano)
        zero (Rellena los valores faltantes con 0)
    """
    # Crear una copia del conjunto de datos original
    data_imputada = data.copy()

    # Excluir las columnas proporcionadas por el usuario
    data_imputada = data_imputada.drop(excluidas, axis=1)

    # Seleccionar el método de imputación
    if metodo == 'moda':
        # Imputación por moda
        data_imputada = data_imputada.fillna(data_imputada.mode().iloc[0])
    elif metodo == 'media':
        # Imputación por media
        data_imputada = data_imputada.fillna(data_imputada.mean())
    elif metodo == 'mediana':
        # Imputación por mediana
        data_imputada = data_imputada.fillna(data_imputada.median())
    elif metodo == 'knn':
        # Imputación por knn
        # Implementar el algoritmo knn con el número de vecinos indicado
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data_imputada = pd.DataFrame(imputer.fit_transform(data_imputada), columns=data_imputada.columns)
    elif metodo == 'interpolación':
        # Imputación por interpolación
        data_imputada = data_imputada.interpolate(method=metodo_interp)

    # Agregar las columnas excluidas de vuelta al conjunto de datos
    data_imputada = pd.concat([data_imputada, data[excluidas]], axis=1)

    # Retornar el conjunto de datos imputado
    return data_imputada
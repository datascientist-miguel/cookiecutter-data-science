import pandas as pd
from scipy.stats import spearmanr

def correlacion_spearman(dataset):
    # Calcular la matriz de correlación con el método de Pearson
    correlaciones = dataset.corr(method='spearman')

    # Crear una tabla que muestre las correlaciones en las columnas
    tabla_correlaciones = pd.DataFrame(columns=correlaciones.columns, index=correlaciones.columns)

    # Llenar la tabla con los valores de correlación
    for columna in correlaciones.columns:
        for fila in correlaciones.index:
            coeficiente, p_valor = spearmanr(dataset[columna], dataset[fila])
            tabla_correlaciones.loc[fila, columna] = float(coeficiente)
     
    return tabla_correlaciones
import pandas as pd
from scipy.stats import pearsonr

def correlacion_pearson(dataset):
    # Calcular la matriz de correlación con el método de Pearson
    correlaciones = dataset.corr(method='pearson')

    # Crear una tabla que muestre las correlaciones en las columnas
    tabla_correlaciones = pd.DataFrame(columns=correlaciones.columns, index=correlaciones.columns)

    # Llenar la tabla con los valores de correlación
    for columna in correlaciones.columns:
        for fila in correlaciones.index:
            coeficiente, p_valor = pearsonr(dataset[columna], dataset[fila])
            tabla_correlaciones.loc[fila, columna] = float(coeficiente)
     
    return tabla_correlaciones
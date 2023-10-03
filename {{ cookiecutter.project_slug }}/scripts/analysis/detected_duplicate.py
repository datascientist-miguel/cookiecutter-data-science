import pandas as pd

def detect_duplicados(dataframe):

    # Encontrar filas duplicadas
    duplicates = dataframe[dataframe.duplicated(keep=False)]

    # Imprimir un mensaje con las filas duplicadas
    if len(duplicates) > 0:
        print('Se encontraron', len(duplicates), 'filas duplicadas:')
        print(duplicates)
    else:
        print('No se encontraron filas duplicadas.')
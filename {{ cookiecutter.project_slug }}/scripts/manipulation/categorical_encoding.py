from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import pandas as pd
"""
    Función que transforma variables categóricas en variables numéricas.
    Los métodos disponibles son 'onehot', 'ordinal', 'binary' y 'count'.
    """
    
def categorical_encoding(data, columnas, metodo='onehot'):

    # Crear una copia del conjunto de datos original
    data_transformada = data.copy()

    # Seleccionar el método de transformación
    if metodo == 'onehot':
        # Transformación por one-hot encoding
        # Crear un objeto OneHotEncoder
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Iterar sobre las columnas seleccionadas
        for columna in columnas:
            # Crear un DataFrame con las variables codificadas
            columnas_codificadas = pd.DataFrame(
                onehot_encoder.fit_transform(data_transformada[columna].values.reshape(-1,1)),
                columns=[columna + '_' + str(int(i)) for i in range(onehot_encoder.categories_[0].size)]
            )

            # Eliminar la columna original y concatenar la codificación one-hot
            data_transformada = pd.concat(
                [data_transformada.drop(columna, axis=1), columnas_codificadas], axis=1
            )

    elif metodo == 'ordinal':
        # Transformación por label encoding ordinal
        # Crear un objeto OrdinalEncoder
        ordinal_encoder = OrdinalEncoder()

        # Iterar sobre las columnas seleccionadas
        for columna in columnas:
            # Crear una copia de la columna y transformarla
            columna_transformada = ordinal_encoder.fit_transform(data_transformada[columna].values.reshape(-1,1))

            # Reemplazar la columna original con la columna transformada
            data_transformada[columna] = columna_transformada

    elif metodo == 'binary':
        # Transformación por binary encoding
        # Iterar sobre las columnas seleccionadas
        for columna in columnas:
            # Crear una copia de la columna y transformarla
            columna_transformada = pd.get_dummies(data_transformada[columna], prefix=columna, drop_first=True)

            # Concatenar la columna transformada con el conjunto de datos original
            data_transformada = pd.concat([data_transformada, columna_transformada], axis=1)

            # Eliminar la columna original
            data_transformada = data_transformada.drop(columna, axis=1)

    elif metodo == 'count':
        # Transformación por count encoding
        # Iterar sobre las columnas seleccionadas
        for columna in columnas:
            # Crear un diccionario con la frecuencia de cada valor en la columna
            frecuencias = data_transformada[columna].value_counts().to_dict()

            # Crear una nueva columna con la frecuencia de cada valor
            data_transformada[columna+'_count'] = data_transformada[columna].map(frecuencias)

            # Eliminar la columna original
            data_transformada = data_transformada.drop(columna, axis=1)

    # Retornar el conjunto de datos transformado
    return data_transformada
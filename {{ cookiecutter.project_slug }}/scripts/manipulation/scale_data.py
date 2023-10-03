from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd

def scale_data(data, method='standard', columns=None):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Método de escalado no válido. Los métodos válidos son: 'standard', 'minmax', 'robust'")

    if columns is None:
        columns = data.columns

    # Crear una lista con las columnas que no se van a escalar
    non_scaled_columns = [col for col in data.columns if col not in columns]
    
    # Crear un nuevo dataframe con las columnas que no se van a escalar
    non_scaled_data = data[non_scaled_columns]

    # Escalar solo las columnas seleccionadas
    scaled_data = scaler.fit_transform(data[columns])
    scaled_data = pd.DataFrame(scaled_data, columns=columns, index=data.index)

    # Crear un nuevo dataframe con las columnas escaladas
    scaled_data = pd.concat([scaled_data, non_scaled_data], axis=1)

    return scaled_data
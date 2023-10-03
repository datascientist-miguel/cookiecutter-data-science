import numpy as np
from scipy.stats import boxcox, yeojohnson


def transformar_datos(data, metodo='log', columnas_excluidas=[], valor_lambda=None):
    """
    Transforma los datos según el método seleccionado.

    Parámetros
    ----------
    data : DataFrame
        Los datos a transformar.
    metodo : str, opcional (por defecto "log")
        El método de transformación a aplicar. Las opciones son:
        - "log": transformación logarítmica natural
        - "sqrt": raíz cuadrada
        - "reciprocal": recíproco
        - "boxcox": transformación Box-Cox
        - "yeojohnson": transformación Yeo-Johnson
    columnas_excluidas : list, opcional (por defecto [])
        Lista de columnas a excluir de la transformación
    valor_lambda : float, opcional (por defecto None)
        Valor lambda a utilizar en la transformación Box-Cox en lugar del valor obtenido automáticamente.

    Retorna
    -------
    DataFrame
        Los datos transformados.
    """
    
    # Crear una copia de los datos originales
    datos_transformados = data.copy()

    # Transformar los datos de cada columna
    for col in datos_transformados.columns:
        if col in columnas_excluidas:
            continue
        if metodo == "log":
            datos_transformados[col] = np.log(datos_transformados[col])
        elif metodo == "sqrt":
            datos_transformados[col] = np.sqrt(datos_transformados[col])
        elif metodo == "reciprocal":
            datos_transformados[col] = np.reciprocal(datos_transformados[col])
        elif metodo == "boxcox":
            if np.all(datos_transformados[col] > 0):
                if valor_lambda is not None:
                    datos_transformados[col] = boxcox(datos_transformados[col], lmbda=valor_lambda)
                else:
                    datos_transformados[col], _ = boxcox(datos_transformados[col])
            else:
                print(f"Advertencia: No se puede realizar la transformación Box-Cox en la columna '{col}' porque contiene valores Negativos.")
        elif metodo == "yeojohnson":
            valores_positivos = datos_transformados[col][datos_transformados[col] > 0]
            valores_transformados = yeojohnson(valores_positivos)[0]
            datos_transformados.loc[datos_transformados[col] > 0, col] = valores_transformados
        else:
            print('Error: Método de transformación no válido.')
            return None
    
    return datos_transformados
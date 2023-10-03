import pandas as pd
import numpy as np

def descriptives(df):
    var_names = list(df.columns)
    rango = [np.max(df[var])-np.min(df[var]) for var in var_names]
    var_muestral = [np.var(df[var], ddof=1) for var in var_names]
    desv_muestral = [np.std(df[var], ddof=1) for var in var_names]
    q1 = [np.percentile(df[var], 25) for var in var_names]
    q3 = [np.percentile(df[var], 75) for var in var_names]
    rango_intercuartilico = [q3[i] - q1[i] for i in range(len(var_names))]
    coeficiente_variacion = [desv_muestral[i] / np.mean(df[var_names[i]]) for i in range(len(var_names))]
    asimetria = [df[var_names[i]].skew() for i in range(len(var_names))]
    curtosis = [df[var_names[i]].kurtosis() for i in range(len(var_names))]

    df_descriptives = pd.DataFrame({"Variable": var_names, 
                                    "Rango": rango,
                                    "Varianza Muestral": var_muestral,
                                    "Desviación Estándar Muestral": desv_muestral,
                                    "Rango Intercuartílico": rango_intercuartilico,
                                    "Coeficiente de Variación": coeficiente_variacion,
                                    "Asimetría": asimetria,
                                    "Curtosis": curtosis})
    
    df_descriptives = df_descriptives.round(2)
    return df_descriptives
import matplotlib.pyplot as plt
import numpy as np

def boxplot_dataframe(data):
    #seleccionar solo las columnas numéricas del dataframe
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    #crear un gráfico de caja para cada columna numérica
    fig, axs = plt.subplots(ncols=len(numeric_cols), figsize=(16,5))
    for i, col in enumerate(numeric_cols):
        axs[i].boxplot(data[col].dropna())
        axs[i].set_title(col)
    
    #ajustar el espaciado entre los subplots y mostrar el gráfico
    plt.tight_layout()
    plt.show()
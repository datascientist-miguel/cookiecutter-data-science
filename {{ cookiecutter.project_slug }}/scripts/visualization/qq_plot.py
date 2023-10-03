import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def qq_plot(data, save_fig=False, fig_name="qq-plot.png"):

    fig, axs = plt.subplots(nrows=1, ncols=data.shape[1], figsize=(15, 5))
    fig.tight_layout(pad=3.0)
    ref_line = np.arange(-4, 4, 0.1)

    for i, col in enumerate(data.columns):
        # Calcular los cuantiles teóricos y muestrales para el gráfico QQ
        norm = stats.norm.fit(data[col])
        theor_quantiles = stats.norm.ppf(np.arange(0.01, 1, 0.01), *norm)
        samp_quantiles = np.percentile(data[col], np.arange(0.01, 1, 0.01) * 100)

        # Graficar el gráfico QQ con línea de referencia
        axs[i].scatter(theor_quantiles, samp_quantiles)
        axs[i].plot(ref_line, ref_line, '--', color='gray')
        axs[i].set_title(col)
        axs[i].set_xlabel('Cuantiles teóricos')
        axs[i].set_ylabel('Cuantiles muestrales')

    plt.show()

    # Guardar la figura en un archivo si save_fig es True
    if save_fig:
        fig.savefig(fig_name)
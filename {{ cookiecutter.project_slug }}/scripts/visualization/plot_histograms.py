import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np



def plot_histograms(df):
    num_cols = len(df.columns)
    num_rows = num_cols // 5 + (num_cols % 5 > 0)
    fig, axs = plt.subplots(nrows=num_rows, ncols=5, figsize=(20, num_rows*4))
    axs = axs.flatten()

    for i, col in enumerate(df.columns):
        data = df[col]
        mu, std = np.mean(data), np.std(data)

        axs[i].hist(data, bins=10, density=True, alpha=0.7, color='skyblue')
        xmin, xmax = axs[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        axs[i].plot(x, p, 'k', linewidth=2)
        axs[i].axvline(mu-3*std, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu-2*std, color='y', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu-std, color='g', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu, color='purple', linestyle='-', linewidth=2, alpha=0.5)
        axs[i].axvline(mu+std, color='g', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu+2*std, color='y', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].axvline(mu+3*std, color='r', linestyle='--', linewidth=2, alpha=0.5)
        axs[i].set_xlabel(col)
        axs[i].set_ylabel('Frecuencia relativa')

    for i in range(num_cols, num_rows*5):
        fig.delaxes(axs[i])

    fig.suptitle('Histogramas con densidad de probabilidad y regla empírica', fontsize=18)
    fig.tight_layout()
    plt.show()
        
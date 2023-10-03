import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_matrix(data):
    corr_matrix = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Matrix de Correlación')
    plt.show()
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(data, method='pearson'):
    """
    Plots a correlation heatmap for the given data using the specified correlation method.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing numerical columns.
        method (str): The correlation method to use: 'pearson', 'spearman', or 'kendall'.

    Returns:
        None

    Plots a heatmap representing the correlation matrix.

    Usage:
        plot_correlation_heatmap(dataframe, method='pearson')
    """
    # Check if all variables in data are numeric
    non_numeric_vars = [col for col in data.columns if col not in data.select_dtypes(include=[np.number]).columns]
    if non_numeric_vars:
        print("Non-numeric variables found. Skipping correlation calculation for those variables.")
        data = data.drop(non_numeric_vars, axis=1)

    # Calculate the correlation matrix
    corr_matrix = data.corr(method=method)

    # Create a mask to only plot the lower triangle of the matrix (excluding the diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Get the number of columns and rows for adjusting the figure size
    num_cols = corr_matrix.shape[1]
    num_rows = corr_matrix.shape[0]

    # Calculate the appropriate figure size based on the number of columns and rows
    fig_size = (num_cols * 0.5, num_rows * 0.5)

    # Set up the matplotlib figure with the calculated size
    plt.figure(figsize=fig_size)

    # Set a nice color palette
    colors = sns.color_palette("viridis", as_cmap=True)

    # Plot the heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap=colors, annot=True, fmt='.1f', linewidths=0.5)

    plt.title(f'Correlation Matrix - {method.capitalize()} Correlation', fontsize=15)
    plt.show()
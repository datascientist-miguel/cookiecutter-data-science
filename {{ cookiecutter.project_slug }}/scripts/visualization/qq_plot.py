import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def qq_plots(df, qq_type='norm'):
    """
    Plots QQ plots for each numerical column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing numerical columns.
        qq_type (str): The type of QQ plot to generate. Options: 'norm', 'log', 'exp', 'uniform'.

    Returns:
        None

    Plots QQ plots to assess the normality of the data in each numerical column.
    QQ plots compare the quantiles of the data with the theoretical quantiles of a specified distribution.

    Usage:
        plot_qq_plots(dataframe, qq_type='norm')
    """
    # Filter out only numerical columns (excluding binary numerical columns)
    numerical_cols = [col for col in df.select_dtypes(include=[np.number]).columns if len(df[col].unique()) > 2]

    num_cols = len(numerical_cols)
    num_rows = num_cols // 5 + (num_cols % 5 > 0)
    fig, axs = plt.subplots(nrows=num_rows, ncols=5, figsize=(20, num_rows*4))
    axs = axs.flatten()

    for i, col in enumerate(numerical_cols):
        data = df[col]
        # Calculate the quantiles for the data
        quantiles_data = np.percentile(data, np.linspace(0, 100, 100))

        if qq_type == 'norm':
            # Generate theoretical quantiles from a normal distribution
            quantiles_theoretical = np.percentile(np.random.normal(np.mean(data), np.std(data), len(data)), np.linspace(0, 100, 100))
        elif qq_type == 'log':
            # Generate theoretical quantiles from a log-normal distribution
            quantiles_theoretical = np.percentile(np.random.lognormal(np.mean(data), np.std(data), len(data)), np.linspace(0, 100, 100))
        elif qq_type == 'exp':
            # Generate theoretical quantiles from an exponential distribution
            quantiles_theoretical = np.percentile(np.random.exponential(np.mean(data), len(data)), np.linspace(0, 100, 100))
        elif qq_type == 'uniform':
            # Generate theoretical quantiles from a uniform distribution
            quantiles_theoretical = np.percentile(np.random.uniform(np.min(data), np.max(data), len(data)), np.linspace(0, 100, 100))
        else:
            raise ValueError("Invalid qq_type. Supported types: 'norm', 'log', 'exp', 'uniform'.")

        axs[i].plot(quantiles_theoretical, quantiles_data, 'o', color='skyblue')
        axs[i].plot(quantiles_theoretical, quantiles_theoretical, color='k', linestyle='--')
        axs[i].set_xlabel('Theoretical Quantiles')
        axs[i].set_ylabel('Sample Quantiles')
        axs[i].set_title(col)

    for i in range(num_cols, num_rows*5):
        fig.delaxes(axs[i])

    fig.suptitle(f'QQ Plots for Assessing {qq_type.capitalize()} Distribution', fontsize=15)
    fig.tight_layout()
    plt.show()
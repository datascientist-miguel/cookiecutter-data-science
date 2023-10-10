import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def kdeplot_stats(data):
    """
    This function is used to plot kde plots with statistical measures for numerical variables
    :param data: input dataset
    :return: plot
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame.")

        # Leave only the numerical columns that are not binary
        numerical_columns = [col for col in data.columns if np.issubdtype(data[col].dtype, np.number) and len(data[col].unique()) > 2]
        
        # Calculate the number of rows needed for subplots
        num_columns = len(numerical_columns)
        num_rows = (num_columns + 2) // 3  # Ceiling division to determine the number of rows

        # Set up the subplots
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        # Plot each numerical column
        for i, column in enumerate(numerical_columns):
            ax = axes[i]
            sns.kdeplot(data=data, x=column, ax=ax, color='lightblue')
            ax.set_title(f"KDE Plot for {column}")
            
            # Calculate measures
            mean_value = data[column].mean()
            median_value = data[column].median()
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)

            # Add lines for statistics
            ax.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_value:.2f} mean)')
            ax.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median ({median_value:.2f} median)')
            ax.axvline(q1, color='orange', linestyle='dashed', linewidth=2, label=f'Q1 ({q1:.1f} Q1)')
            ax.axvline(q3, color='purple', linestyle='dashed', linewidth=2, label=f'Q3 ({q3:.1f} Q3)')

            ax.legend()

            ax.set_title(f"KDE Plot for {column}")
        
        # Hide any empty subplots
        for j in range(num_columns, num_rows * 3):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    except ValueError as ve:
        print("Error: ", str(ve))
        return None


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def boxplot(data, hue):
    """
    This function is used to plot boxplots for numerical variables
    :param data: input dataset
    :param hue: categorical variable
    :return: plot
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame.")

        # Remove the 'hue' column from the list of columns to plot
        numerical_columns = [col for col in data.columns if col != hue]
        
        # Leave only the numerical columns that are not binary
        numerical_columns = [col for col in numerical_columns if np.issubdtype(data[col].dtype, np.number) and len(data[col].unique()) > 2]
        
        # Calculate the number of rows needed for subplots
        num_columns = len(numerical_columns)
        num_rows = (num_columns + 2) // 3  # Ceiling division to determine the number of rows

        # Set up the subplots
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        # Plot each numerical column
        for i, column in enumerate(numerical_columns):
            sns.boxplot(data=data, x=hue, y=column, ax=axes[i], color='lightblue')
            axes[i].set_title(f"Boxplot for {column}")
        
        # Hide any empty subplots
        for j in range(num_columns, num_rows * 3):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    except ValueError as ve:
        print("Error: ", str(ve))
        return None
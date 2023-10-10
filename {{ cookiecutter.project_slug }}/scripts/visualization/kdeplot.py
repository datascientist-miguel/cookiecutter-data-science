import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def kdplot(data, hue):
    """
    This function is used to plot the distribution of a numerical variable
    :param data: input dataset
    :param hue: categorical variable
    :return: plot
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data should be a pandas DataFrame.")
        if not isinstance(hue, str) or hue not in data.columns:
            raise ValueError("Invalid column name for hue.")

        # Remove the 'hue' column from the list of columns to plot
        numerical_columns = [col for col in data.columns if col != hue]
        # Leave only the numerical columns
        numerical_columns = [col for col in numerical_columns if np.issubdtype(data[col].dtype, np.number)]
        
        # Calculate the number of rows needed for subplots
        num_columns = len(numerical_columns)
        num_rows = (num_columns + 2) // 3  
        # Set up the subplots
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        # Plot each numerical column
        for i, column in enumerate(numerical_columns):
            sns.kdeplot(data=data, x=column, hue=hue, multiple="stack", palette="crest", ax=axes[i])
            axes[i].set_title(f"KDE Plot for {column}")
        
        # Hide any empty subplots
        for j in range(num_columns, num_rows * 3):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    except ValueError as ve:
        print("Error: ", str(ve))
        return None

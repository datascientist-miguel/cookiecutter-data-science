from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pca_lda_analysis(data, n_components=None, excluded_cols=None, method='pca', target_column=None):
    """
    Perform PCA or LDA analysis on the input dataset and visualize the results.

    Parameters:
    data (DataFrame): Input DataFrame.
    n_components (int): Number of components for PCA. Default is None.
    excluded_cols (list): List of columns to exclude. Default is None.
    method (str): Analysis method. Options: 'pca', 'lda'. Default is 'pca'.
    target_column (str): Name of the target column for LDA. Default is None.

    Returns:
    DataFrame or tuple: Transformed DataFrame (for PCA) or tuple of components and weights (for LDA).

    Raises:
    ValueError: If the specified method is invalid.
    """

    # Check if excluded columns are provided
    if excluded_cols is None:
        excluded_cols = []

    # Handle PCA or LDA based on the chosen method
    if method == 'pca':
        # Filter out non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data_numeric = data[numeric_cols]

        pca = PCA(n_components=n_components)
        pca.fit(data_numeric.drop(excluded_cols, axis=1))

        # Get the principal components
        components = pca.transform(data_numeric.drop(excluded_cols, axis=1))
        
        # Get the explained variance ratio
        explained_var = pca.explained_variance_ratio_

        # Create a DataFrame from the principal components
        components_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(len(explained_var))])

        # Visualize the proportion of variance explained by each principal component
        plt.figure(figsize=(8, 6))
        sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_var))], y=explained_var)
        plt.title("Proportion of Variance Explained by Each Principal Component")
        plt.xlabel("Principal Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.show()

        # Get the weights of the variables for the first principal component
        weights = pca.components_[0]

        # Get the names of the original variables from the DataFrame
        variable_names_pca = data_numeric.drop(excluded_cols, axis=1).columns.tolist()

        # Visualize the weights of the variables
        plt.figure(figsize=(8, 6))
        sns.barplot(x=variable_names_pca, y=weights)
        plt.title("Variable Weights in the First Principal Component (PCA)")
        plt.xlabel("Variable")
        plt.ylabel("Weight")
        plt.xticks(rotation=90)
        plt.show()

        return components_df

    elif method == 'lda':
        if target_column is None:
            raise ValueError("For LDA, a target column must be specified using the 'target_column' parameter.")

        # Filter out non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data_numeric = data[numeric_cols]

        lda = LinearDiscriminantAnalysis()
        lda.fit(data_numeric.drop(excluded_cols, axis=1), data[target_column])

        # Get the first discriminant component
        lda_components = lda.transform(data_numeric.drop(excluded_cols, axis=1))

        # Get the weights of the variables for the first discriminant component
        weights_lda = lda.coef_[0]
        
        # Get the names of the original variables from the DataFrame
        variable_names_lda = data_numeric.drop(excluded_cols, axis=1).columns.tolist()
        
        # Get explained variance ratio, means, and priors
        explained_variance = lda.explained_variance_ratio_
        class_means = lda.means_
        priors = lda.priors_
        
        # Visualize the proportion of variance explained by each component
        plt.figure(figsize=(8, 6))
        sns.barplot(x=[f"Component {i+1}" for i in range(len(explained_variance))], y=explained_variance)
        plt.title("Proportion of Variance Explained by Each Component (LDA)")
        plt.xlabel("Component")
        plt.ylabel("Proportion of Variance Explained")
        plt.show()

        # Visualize the weights of the variables
        plt.figure(figsize=(8, 6))
        sns.barplot(x=variable_names_lda, y=weights_lda)
        plt.title("Variable Weights in the First Discriminant Component (LDA)")
        plt.xlabel("Variable")
        plt.ylabel("Weight")
        plt.xticks(rotation=90)
        plt.show()

        return lda_components, weights_lda, explained_variance, class_means, priors

    else:
        raise ValueError("Invalid analysis method. Valid methods are: 'pca' or 'lda'.")
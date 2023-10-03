from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def pca_lda_analysis(data, n_components=None, excluded_cols=None, method='pca'):
    if excluded_cols is None:
        excluded_cols = []
    # Realizar PCA o LDA
    if method == 'pca':
        pca = PCA(n_components=n_components)
        pca.fit(data.drop(excluded_cols, axis=1))

        # Obtener las componentes principales
        components = pca.transform(data.drop(excluded_cols, axis=1))
        
        # Obtener el porcentaje de varianza explicada
        explained_var = pca.explained_variance_ratio_
        
        # Crear un DataFrame a partir de los componentes principales
        components_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(len(explained_var))])

        # Graficar la proporción de varianza explicada por cada componente principal
        plt.figure(figsize=(8, 6))
        sns.barplot(x=[f"PC{i+1}" for i in range(len(explained_var))], y=explained_var)
        plt.title("Proporción de varianza explicada por cada componente principal")
        plt.xlabel("Componente principal")
        plt.ylabel("Proporción de varianza explicada")

        # Obtener los pesos de las variables para la primera componente principal
        weights = pca.components_[0]

        # Obtener los nombres de las variables originales del DataFrame
        variable_names_pca = data.drop(excluded_cols, axis=1).columns.tolist()

        # Graficar los pesos de las variables
        plt.figure(figsize=(8, 6))
        sns.barplot(x=variable_names_pca, y=weights)
        plt.title("Pesos de las variables en el primera componente principal (PCA)")
        plt.xlabel("Variable")
        plt.ylabel("Peso")
        # Rotar los nombres de las variables en el eje X en 90 grados
        plt.xticks(rotation=90)

        # Retornar las componentes principales, el porcentaje de varianza explicada y los pesos de las variables
        return components_df
    
    
    elif method == 'lda':
        lda = LinearDiscriminantAnalysis()
        lda.fit(data.drop(excluded_cols, axis=1), data.iloc[:, -1])

        # Obtener la primera componente discriminante
        lda_components = lda.transform(data.drop(excluded_cols, axis=1))

        # Obtener los pesos de las variables para la primera componente discriminante
        weights_lda = lda.coef_[0]

        # Obtener los nombres de las variables originales del DataFrame
        variable_names_lda = data.drop(excluded_cols, axis=1).columns.tolist()

        # Graficar los pesos de las variables
        plt.figure(figsize=(8, 6))
        sns.barplot(x=variable_names_lda, y=weights_lda)
        plt.title("Pesos de las variables en el primera componente discriminante (LDA)")
        plt.xlabel("Variable")
        plt.ylabel("Peso")
        # Rotar los nombres de las variables en el eje X en 90 grados
        plt.xticks(rotation=90)

        # Retornar las componentes discriminantes y los pesos de las variables
        return lda_components, weights_lda
    else:
        raise ValueError("El método especificado no es válido. Los métodos válidos son 'pca' o 'lda'.")
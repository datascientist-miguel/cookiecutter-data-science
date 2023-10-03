from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

def hierarchical_train(data):
    dict_params = {
        "n_clusters": [2, 3, 4, 5, 6],
        "linkage": ['ward'],
        "metric": ['euclidean']
    }
    results = pd.DataFrame(columns=["n_clusters", "linkage", "metric", "silhouette", "calinski"])
    
    for params in product(*dict_params.values()):
        param_names = dict_params.keys()
        param_dict = dict(zip(param_names, params))
        
        hierarchical = AgglomerativeClustering(n_clusters=param_dict["n_clusters"], linkage=param_dict["linkage"],
                                               metric=param_dict["metric"])
        hierarchical.fit(data)
        
        silhouette = silhouette_score(data, hierarchical.labels_)
        calinski = calinski_harabasz_score(data, hierarchical.labels_)
        
        results = pd.concat([results, pd.DataFrame({"n_clusters": [param_dict["n_clusters"]],
                                                    "linkage": [param_dict["linkage"]],
                                                    "metric": [param_dict["metric"]],
                                                    "silhouette": [silhouette],
                                                    "calinski": [calinski]})], ignore_index=True)
    
    plt.plot(results["n_clusters"], results["silhouette"], "-o")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Coeficiente de Silueta")
    plt.title("Gráfico de Silueta")
    plt.show()

    plt.plot(results["n_clusters"], results["calinski"], "-o")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Índice de Calinski-Harabasz")
    plt.title("Gráfico de Calinski-Harabasz")
    plt.show()
    
    best_params = results.loc[results["silhouette"].idxmax()]
    hierarchical = AgglomerativeClustering(n_clusters=best_params["n_clusters"], linkage=best_params["linkage"],
                                           metric=best_params["metric"])
    hierarchical.fit(data)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=hierarchical.labels_, cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Clusters")
    plt.show()
    
    # Dendrograma
    plt.figure(figsize=(12, 8))
    Z = linkage(data, method=best_params["linkage"], metric=best_params["metric"])
    dendrogram(Z, truncate_mode='level', p=5)
    plt.xlabel("Índices de las muestras")
    plt.ylabel("Distancia")
    plt.title("Dendrograma")
    plt.show()
    
    return hierarchical, results
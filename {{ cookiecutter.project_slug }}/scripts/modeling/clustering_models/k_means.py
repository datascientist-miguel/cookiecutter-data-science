from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

def kmeans_train(data, n_init=10, random_state=42):
    dict_params = {
        "n_clusters": [2, 3, 4, 5, 6],
        "init": ["k-means++", "random"]
    }
    results = pd.DataFrame(columns=["n_clusters", "init", "inertia", "silhouette", "calinski"])
    for params in product(*dict_params.values()):
        param_names = dict_params.keys()
        param_dict = dict(zip(param_names, params))
        kmeans = KMeans(n_clusters=param_dict["n_clusters"], init=param_dict["init"], n_init=n_init, random_state=random_state)
        kmeans.fit(data)
        inertia = kmeans.inertia_
        silhouette = silhouette_score(data, kmeans.labels_)
        calinski = calinski_harabasz_score(data, kmeans.labels_)
        results = pd.concat([results, pd.DataFrame({"n_clusters": [param_dict["n_clusters"]], "init": [param_dict["init"]],
                                                    "inertia": [inertia], "silhouette": [silhouette],
                                                    "calinski": [calinski]})], ignore_index=True)

    plt.plot(results["n_clusters"], results["inertia"], "-o")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Inercia")
    plt.title("Gráfico de Codo")
    plt.show()

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
    kmeans = KMeans(n_clusters=best_params["n_clusters"], init=best_params["init"], n_init=n_init, random_state=random_state)
    kmeans.fit(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=kmeans.labels_, cmap="viridis")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="x", s=200, linewidths=3, color="black")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Clusters")
    plt.show()

    return kmeans, results
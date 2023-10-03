from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

def dbscan_train(data, eps_range=(0.1, 2.0), min_samples_range=(2, 10)):
    dict_params = {
        "eps": np.linspace(eps_range[0], eps_range[1], num=10),
        "min_samples": np.arange(min_samples_range[0], min_samples_range[1]+1)
    }
    
    best_silhouette = -1
    best_params = {}
    best_dbscan = None
    
    for params in product(*dict_params.values()):
        param_names = dict_params.keys()
        param_dict = dict(zip(param_names, params))
        
        dbscan = DBSCAN(eps=param_dict["eps"], min_samples=param_dict["min_samples"])
        dbscan.fit(data)
        
        labels = dbscan.labels_
        n_labels = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise points
        
        if n_labels > 1:
            silhouette = silhouette_score(data, labels)
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_params = param_dict
                best_dbscan = dbscan
    
    best_eps = best_params["eps"]
    best_min_samples = best_params["min_samples"]
    
    best_dbscan.fit(data)

    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=best_dbscan.labels_, cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Clusters")
    plt.show()

    return best_dbscan, best_silhouette
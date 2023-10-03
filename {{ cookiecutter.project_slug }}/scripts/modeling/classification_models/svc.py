import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def support_vector_classifier(X, y):
    # Definir el diccionario de hiperparámetros para evaluar
    param_grid = {
        'C': [0.1, 1.0, 10.0],  # Parámetro de regularización
        'kernel': ['linear', 'rbf', 'poly'],  # Tipo de kernel
        'gamma': ['scale', 'auto']  # Coeficiente del kernel para 'rbf' y 'poly'
    }

    # Crear un modelo de SVC
    model = SVC()

    # Aplicar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)

    # Obtener los mejores hiperparámetros y el accuracy correspondiente
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    # Reentrenar el modelo con los mejores hiperparámetros
    model.set_params(**best_params)
    model.fit(X, y)

    # Calcular el accuracy total
    y_pred = cross_val_predict(model, X, y, cv=5)
    total_accuracy = accuracy_score(y, y_pred)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y, y_pred)

    # Mostrar la matriz de confusión
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='red')

    plt.title('Matriz de Confusión')
    plt.colorbar()
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
    plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
    plt.show()

    return best_params, best_accuracy, total_accuracy, conf_matrix
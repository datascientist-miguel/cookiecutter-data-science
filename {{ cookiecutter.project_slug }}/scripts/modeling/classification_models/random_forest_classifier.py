import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree

def random_forest_classifier(X, y):
    # Definir el diccionario de hiperparámetros para evaluar
    param_grid = {
        'n_estimators': [100],  # Número de árboles en el bosque
        'criterion': ['gini'],  # Criterio para medir la calidad de la división
        'max_depth': [5],  # Profundidad máxima de los árboles
        'min_samples_split': [2],  # Número mínimo de muestras requeridas para dividir un nodo
        'min_samples_leaf': [2]  # Número mínimo de muestras requeridas en una hoja
    }

    # Crear un modelo de Random Forest
    model = RandomForestClassifier()

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

    # Graficar el árbol promedio de los estimadores en el bosque
    plt.figure(figsize=(10, 8))
    plot_tree(model.estimators_[0], feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True)
    plt.show()

    return best_params, best_accuracy, total_accuracy, conf_matrix

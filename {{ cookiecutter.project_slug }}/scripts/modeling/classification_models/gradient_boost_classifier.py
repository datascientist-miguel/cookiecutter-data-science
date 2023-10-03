import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

def gradient_boost_classifier(X, y):
    # Definir el diccionario de hiperparámetros para evaluar
    param_grid = {
        'n_estimators': [100, 200, 500],  # Número de estimadores
        'learning_rate': [0.01, 0.1, 1.0],  # Tasa de aprendizaje
        'max_depth': [3, 5, 10],  # Profundidad máxima de los árboles base
        'subsample': [0.8, 1.0]  # Proporción de muestras utilizadas para entrenar cada árbol
    }

    # Crear un modelo de Gradient Boosting
    model = GradientBoostingClassifier()

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

    # Calcular la importancia de las características
    feature_importance = permutation_importance(model, X, y, n_repeats=10, random_state=0)

    # Obtener los nombres de las características
    feature_names = X.columns

    # Graficar la importancia de las características
    sorted_idx = feature_importance.importances_mean.argsort()
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance.importances_mean[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.xlabel('Importancia')
    plt.title('Importancia de las Características')
    plt.show()

    return best_params, best_accuracy, total_accuracy, conf_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def mlp_classifier(X, y):
    # Definir el diccionario de hiperparámetros para evaluar
    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (20, 20, 20)],  # Tamaños de las capas ocultas
        'activation': ['relu', 'tanh', 'logistic'],  # Funciones de activación
        'solver': ['sgd', 'adam'],  # Algoritmo de optimización
        'learning_rate': ['constant', 'adaptive']  # Tasa de aprendizaje
    }

    # Crear un modelo de MLPClassifier
    model = MLPClassifier()

    # Aplicar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)

    # Obtener los mejores hiperparámetros y el accuracy correspondiente
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    # Reentrenar el modelo con los mejores hiperparámetros
    model.set_params(**best_params)
    model.fit(X, y)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X)

    # Calcular el accuracy total
    total_accuracy = accuracy_score(y, y_pred)

    # Crear una gráfica de dispersión para visualizar las predicciones
    plt.scatter(range(len(y)), y, color='blue', label='Real')
    plt.scatter(range(len(y)), y_pred, color='red', label='Predicción')
    plt.xlabel('Muestras')
    plt.ylabel('Etiquetas')
    plt.title('Comparación entre datos reales y predicciones')
    plt.legend()
    plt.show()

    return best_params, best_accuracy, total_accuracy
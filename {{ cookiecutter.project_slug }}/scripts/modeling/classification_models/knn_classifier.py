import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

def knn_classifier(X, y):
    # Definir el diccionario de hiperparámetros para evaluar
    param_grid = {
        'n_neighbors': [3, 5, 7],  # Número de vecinos más cercanos
        'weights': ['uniform', 'distance'],  # Peso de los vecinos
        'p': [1, 2]  # Parámetro de la distancia (1: distancia de Manhattan, 2: distancia euclidiana)
    }

    # Crear un modelo de KNeighborsClassifier
    model = KNeighborsClassifier()

    # Aplicar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(model, param_grid, cv=10)
    grid_search.fit(X, y)

    # Obtener los mejores hiperparámetros y el accuracy correspondiente
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_
    
    # Realizar validación cruzada para obtener una estimación más precisa del rendimiento
    cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5)
    cv_accuracy = cv_scores.mean()

    # Reentrenar el modelo con los mejores hiperparámetros
    model.set_params(**best_params)
    model.fit(X, y)

    # Realizar predicciones en los datos de entrenamiento
    y_pred = model.predict(X)

    # Calcular el accuracy total
    total_accuracy = accuracy_score(y, y_pred)

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y, y_pred)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(y, y_pred)

    # Graficar la relación entre los datos reales y predichos
    plt.scatter(range(len(y)), y, label='Real')
    plt.scatter(range(len(y_pred)), y_pred, label='Predicho')
    plt.xlabel('Índice de la muestra')
    plt.ylabel('Valor de la variable objetivo')
    plt.title('Relación entre datos reales y predichos')
    plt.legend()
    plt.show()

    # Crear un dataframe con los hiperparámetros evaluados
    param_values = grid_search.cv_results_['params']
    param_results = pd.DataFrame(param_values)
    param_results['Accuracy'] = grid_search.cv_results_['mean_test_score']

    return param_results, best_params, best_accuracy, total_accuracy, mse, conf_matrix, cv_accuracy
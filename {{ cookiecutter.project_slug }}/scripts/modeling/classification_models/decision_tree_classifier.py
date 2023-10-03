import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

def decision_tree_classifier(X, y):
    # Definir el diccionario de hiperparámetros para evaluar
    param_grid = {
        'criterion': ['gini', 'entropy'],  # Criterio para medir la calidad de la división
        'max_depth': [2, 3, 4, 5],  # Profundidad máxima del árbol
        'min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo
        'min_samples_leaf': [1, 2, 4]  # Número mínimo de muestras requeridas en una hoja
    }

    # Crear un modelo de DecisionTreeClassifier
    model = DecisionTreeClassifier()

    # Aplicar GridSearchCV para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)

    # Obtener los mejores hiperparámetros y el accuracy correspondiente
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

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

    # Graficar el árbol de decisión
    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=['0', '1'], filled=True)
    plt.title('Árbol de Decisión')
    plt.show()

    # Crear un dataframe con los hiperparámetros evaluados
    param_values = grid_search.cv_results_['params']
    param_results = pd.DataFrame(param_values)
    param_results['Accuracy'] = grid_search.cv_results_['mean_test_score']

    return param_results, best_params, best_accuracy, total_accuracy, mse, conf_matrix
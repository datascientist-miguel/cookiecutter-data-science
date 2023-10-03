import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve

def logistic_regression(X, y, test_size=0.2):
    try:
        # Crear un modelo de regresión logística
        model = LogisticRegression()

        # Realizar validación cruzada en todo el conjunto de datos
        scores = cross_val_score(model, X, y, cv=5)

        # Calcular la puntuación media de validación cruzada
        mean_score = scores.mean()

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        param_grid = {
            'solver': ['lbfgs', 'liblinear', 'saga'],  # Algoritmo solucionador
            'max_iter': [5000],  # Número máximo de iteraciones
        }

        # Aplicar GridSearchCV para encontrar los mejores hiperparámetros
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Obtener los mejores hiperparámetros y el accuracy correspondiente
        best_params = grid_search.best_params_

        # No es necesario volver a entrenar el modelo con los mejores hiperparámetros, ya que se hace en GridSearchCV

        # Evaluar el modelo
        y_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        # Calcular la matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Convertir la matriz de confusión en un DataFrame de pandas para facilitar la visualización
        confusion_df = pd.DataFrame(conf_matrix)

        # Crear un mapa de calor de la matriz de confusión con etiquetas y porcentajes
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_df, annot=True, fmt='', cmap='Blues', cbar=False)
        plt.xlabel('Valor Predicho')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión')
        plt.savefig("../images/matriz_confusion_rl.png")
        plt.close()

        # Calcular las métricas de precisión, recall, F1-score
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Calcular la curva ROC y el AUC
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Calibración
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=25, strategy='uniform')

        # Crear el DataFrame result con la información solicitada
        result = pd.DataFrame({
            'Métrica': ['Exactitud', 'Precisión', 'Recall', 'F1-Score', 'Media Validación Cruzada', 'ROC AUC'],
            'Valor': [test_accuracy, precision, recall, f1, mean_score, roc_auc]
        })

        # Graficar la curva ROC
        plt.figure()
        plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Tasa de Falsos Positivos (FPR)')
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.savefig("../images/curva_roc.png")
        plt.close()

        # Graficar la curva de calibración
        plt.figure()
        plt.plot(prob_pred, prob_true, 's-', label='Curva de Calibración')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Probabilidad Predicha')
        plt.ylabel('Frecuencia Real')
        plt.title('Curva de Calibración')
        plt.legend(loc="lower right")
        plt.savefig("../images/curva_calibracion.png")
        plt.close()

        return best_params, result

    except ValueError as ve:
        print("ValueError:", str(ve))
    except Exception as e:
        print("Error:", str(e))

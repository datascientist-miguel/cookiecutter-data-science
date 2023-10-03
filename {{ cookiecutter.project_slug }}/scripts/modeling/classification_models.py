from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def train_classification_models(X, y, k=5):
    # Modelos a entrenar
    models = [
        ("Logistic Regression", LogisticRegression()),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("Gradient Boosting", GradientBoostingClassifier()),
        ("Support Vector Machine", SVC(probability=True))
    ]
    
    # Parametros para la validación cruzada
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Entrenamiento y evaluación de modelos
    results = {}
    roc_curves = {}
    
    for model_name, model in models:
        accuracy_scores = []
        confusion_matrices = []
        roc_auc_scores = []  
        mean_fpr = np.linspace(0, 1, 100) 
        
        plt.figure(figsize=(8, 6))  
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
            
            # Cálculo de la curva ROC y AUC-ROC
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc_scores.append(auc(fpr, tpr))
            
            # Gráfico de la curva ROC para cada iteración
            plt.plot(fpr, tpr, alpha=0.3)
        
        avg_accuracy = np.mean(accuracy_scores)
        avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
        avg_roc_auc = np.mean(roc_auc_scores)
        
        # Gráfico de la curva ROC promedio con leyenda del modelo
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess')
        plt.plot(mean_fpr, np.mean(tpr, axis=0), color='b', label=f'{model_name} (AUC = {avg_roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.show()
        
        results[model_name] = {
            "model": model,
            "Accuracy": avg_accuracy,
            "Confusion Matrix": avg_confusion_matrix,
            "AUC-ROC": avg_roc_auc
        }
    
    return results
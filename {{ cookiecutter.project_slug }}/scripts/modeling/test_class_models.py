from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def classification_models(X, y):
    """
    Entrena y evalúa varios modelos de clasificación sin validación cruzada.

    Parameters:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.

    Returns:
        dict: Diccionario que contiene los resultados para cada modelo.

    """
    # Modelos a entrenar
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("K-Nearest Neighbors", KNeighborsClassifier()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("Gradient Boosting", GradientBoostingClassifier()),
        ("Support Vector Machine", SVC(probability=True))
    ]

    # Entrenamiento y evaluación de modelos
    results = {}
    
    # Split  data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Especificidad
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        # Cálculo de la curva ROC y AUC-ROC
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        confusion_matrix_result = confusion_matrix(y_test, y_pred)

        results[model_name] = {
            "model": model,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Specificity": specificity,
            "AUC-ROC": roc_auc,
            "Confusion Matrix": confusion_matrix_result
        }

    return results

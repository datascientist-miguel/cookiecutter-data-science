from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np

def train_regression_models(X, y, k=5):
    # Modelos a entrenar
    models = {
        "Linear Regression" : LinearRegression(),
        "Ridge Regression" : Ridge(),
        "Lasso Regression" : Lasso(),
        "ElasticNet Regression" : ElasticNet(),
        "Decision Tree" : DecisionTreeRegressor(),
        "Random Forest" : RandomForestRegressor(),
        "Gradient Boosting" : GradientBoostingRegressor(),
        "Support Vector Machine" : SVR(),
        "Neural Network" : MLPRegressor()
    }
    
    # Parametros para la validación cruzada
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Entrenamiento y evaluación de modelos
    results = {}
    mse_scores = []
    r2_scores = []
    for model_name, model in models.items():
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        avg_mse = np.mean(mse_scores)
        avg_r2 = np.mean(r2_scores)
        results[model_name] = {
            "model" : model,
            "MSE" : avg_mse,
            "R2" : avg_r2
        }
    return results
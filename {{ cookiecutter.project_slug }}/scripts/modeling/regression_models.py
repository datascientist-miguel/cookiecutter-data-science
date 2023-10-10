from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split

def regression_models(X, y):
    """
    Train various regression models and evaluate them.

    Parameters:
    X (DataFrame): Features.
    y (Series): Target variable.

    Returns:
    dict: Results including trained models, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2) score for each model.
    """
    
    # Models to train
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet Regression": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Support Vector Machine": SVR(),
        "Neural Network": MLPRegressor()
    }
    
    # Training and evaluating models
    results = {}
    
    # Split  data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    
    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            "model": model,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2
        }
    
    
    return results

from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def lasso_regression_tts(X, y, test_size=0.2, random_state=42):
    param_grid = {
        "alpha": [0.0001, 0.0001, 0.001, 0.01, 0.1],  # Valores de alpha a probar
        "fit_intercept": [True, False],
        "max_iter": [1000, 2000, 5000]  # Valores de max_iter a probar
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = Lasso()
    grid_search = GridSearchCV(model, param_grid=param_grid)
    grid_search.fit(X_train, y_train)
    best_model = Lasso(**grid_search.best_params_)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    coefficients = best_model.coef_
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='b', alpha=0.5, label='Valores predichos')
    plt.scatter(y_test, y_test, color='r', alpha=0.5, label='Valores reales')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.title('Comparación entre valores reales y predichos')
    plt.legend()
    plt.show()

    return best_model, mse, coefficients


def lasso_cv_regression_cv(X, y, cv=5):
    model = LassoCV(cv=cv)
    model.fit(X, y)
    best_model = Lasso(alpha=model.alpha_, max_iter=1000)
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    cv_score = cross_val_score(best_model, X, y, cv=cv)
    coefficients = best_model.coef_

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='b', alpha=0.5, label='Valores predichos')
    plt.scatter(y, y, color='r', alpha=0.5, label='Valores reales')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.title('Comparación entre valores reales y predichos')
    plt.legend()
    plt.show()

    return best_model, mse, cv_score.mean(), coefficients


def lasso_lars_cv_regression_cv(X, y, cv=5):
    model = LassoLarsCV(cv=cv)
    model.fit(X, y)
    best_model = LassoLarsCV(max_n_alphas=model.max_n_alphas, max_iter=1000)
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    cv_score = cross_val_score(best_model, X, y, cv=cv)
    coefficients = best_model.coef_

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='b', alpha=0.5, label='Valores predichos')
    plt.scatter(y, y, color='r', alpha=0.5, label='Valores reales')
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.title('Comparación entre valores reales y predichos')
    plt.legend()
    plt.show()

    return best_model, mse, cv_score.mean(), coefficients
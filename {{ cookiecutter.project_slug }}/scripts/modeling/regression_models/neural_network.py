from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def neural_network_regressor(X, y, cv, random_state, image_path):
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (200,)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "lbfgs"],
        "alpha": [0.0001, 0.001, 0.01]
    }
    nn = MLPRegressor(random_state=random_state, max_iter=1000)
    grid_search = GridSearchCV(nn, param_grid=param_grid, cv=cv, n_jobs=1, verbose=1)
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("Mejores Hiperparámetros: ", best_params)
    print("MSE: ", mse)
    print("R2: ", r2)
    
    plt.scatter(range(len(y)), y, color='blue', label='Actual')
    plt.scatter(range(len(y)), y_pred, color='red', label='Predicted')
    plt.xlabel('Data Points')
    plt.ylabel('Target Values')
    plt.legend()
    plt.savefig(image_path)
    plt.show()
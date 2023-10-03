from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def support_vector_regressor(X, y, cv, random_state, image_path):
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"]
    }
    
    svm = SVR()
    grid_search = GridSearchCV(svm, param_grid=param_grid, cv=cv, n_jobs=1, verbose=1)
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("Mejores Hiperparámetros: ", best_params)
    print("MSE: ", mse)
    print("R2: ", r2)
    
    svm_best = SVR(**best_params)
    svm_best.fit(X, y)
    

    plt.scatter(range(len(y)), y, color='blue', label='Actual')
    plt.scatter(range(len(y)), y_pred, color='red', label='Predicted')
    plt.xlabel('Data Points')
    plt.ylabel('Target Values')
    plt.legend()
    plt.savefig(image_path)
    plt.show()
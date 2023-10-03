from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import display

def gradient_boost_regressor(X, y, cv, random_state, image_path):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.1, 0.01, 0.001],
        "max_depth": [3, 5, 10],
        "max_features": ["sqrt", "log2"]
    }
    
    gb = GradientBoostingRegressor(random_state=random_state)
    grid_search = GridSearchCV(gb, param_grid=param_grid, cv=cv, n_jobs=1, verbose=1)
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print("Mejores Hiperparámetros: ", best_params)
    print("MSE: ", mse)
    print("R2: ", r2)
    
    gb_best = GradientBoostingRegressor(random_state=random_state, **best_params)
    gb_best.fit(X, y)
    
    dot_data = export_graphviz(gb_best.estimators_[0, 0], out_file=None, filled=True, rounded=True, feature_names=X.columns, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename=image_path, format="png", cleanup=True)
    display(graph)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from io import StringIO
from IPython.display import Image, display
from sklearn import tree
import pydotplus

def decision_tree_regression(X, y, cv=5):
    param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'min_impurity_decrease': [0.0, 0.1, 0.2]
} 
    model = DecisionTreeRegressor()
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    dot_data = StringIO()
    tree.export_graphviz(best_model, out_file=dot_data, feature_names=list(X.columns))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('../images/decision_tree.png')

    return best_model, best_params, mse, r2
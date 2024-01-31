# This is a Python project that contains a regression pipeline.
# We will use scikit-learn's regression models.

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# Define regressors
regressors = {
    'Random Forest': RandomForestRegressor(),
    'SVR': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'AdaBoost': AdaBoostRegressor()
}

# Define hyperparameter grids for each regressor
param_grids = {
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]},
    'SVR': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'Linear Regression': {'normalize': [True, False]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Dictionary to store the best models
best_models = {}

# Perform GridSearchCV and cross-validation on each regressor
for regressor_name, regressor in regressors.items():
    grid_search = GridSearchCV(regressor, param_grids[regressor_name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # fit to training dataset
    # grid_search.fit(X_train, y_train)
    grid_search.fit()

    # Get the best model from Grid Search
    best_model = grid_search.best_estimator_

    # Perform cross-validation
    # cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    cv_scores = cross_val_score()

    # Store the best model in the dictionary
    best_models[regressor_name] = {
        'model': best_model,
        'cross_val_scores': cv_scores,
        'mean_mse': np.mean(cv_scores),
        'best_parameters': grid_search.best_params_
    }

# Select the best model based on mean cross-validation MSE
best_regressor = min(best_models, key=lambda k: best_models[k]['mean_mse'])

# Print information about the best regressor
print(f"Best Regressor: {best_regressor}")
print(f"Cross-Validation Mean MSE: {best_models[best_regressor]['mean_mse']:.4f}")
print(f"Best Parameters: {best_models[best_regressor]['best_parameters']}")

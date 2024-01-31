# This is a Python project that contains a classification pipeline.
# We will use scikit-learn's classification models.

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Define hyperparameter grids for each classifier
param_grids = {
    'Random Forest': {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20, 30]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
    'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
    'Decision Tree': {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    'Gradient Boosting': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]},
    'AdaBoost': {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 0.2]}
}

# Dictionary to store the best models
best_models = {}

# Perform GridSearchCV on each classifier
for classifier_name, classifier in classifiers.items():
    grid_search = GridSearchCV(classifier, param_grids[classifier_name], cv=5, scoring='accuracy', n_jobs=-1)

    # fit to training dataset
    # grid_search.fit(X_train, y_train)
    grid_search.fit()

    # Get the best model from Grid Search
    best_model = grid_search.best_estimator_

    # Perform cross-validation on best_model
    # cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_scores = cross_val_score()

    # Store the best model in the dictionary
    best_models[classifier_name] = {
        'model': best_model,
        'cross_val_scores': cv_scores,
        'mean_accuracy': np.mean(cv_scores),
        'best_parameters': grid_search.best_params_
    }

# Select the best model based on mean cross-validation accuracy
best_classifier = max(best_models, key=lambda k: best_models[k]['mean_accuracy'])

# Print information about the best model
print(f"Best Classifier: {best_classifier}")
print(f"Cross-Validation Scores: {best_models[best_classifier]['cross_val_scores']}")
print(f"Mean Accuracy: {best_models[best_classifier]['mean_accuracy']:.4f}")
print(f"Best Parameters: {best_models[best_classifier]['best_parameters']}")

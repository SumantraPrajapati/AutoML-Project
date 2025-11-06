import optuna
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np


class HyperparameterTuner:
    def __init__(self, problem_type='classification'):
        """
        problem_type: 'classification' or 'regression'
        """
        self.problem_type = problem_type

    def grid_search(self, model, param_grid, X, y, cv=5, scoring=None, n_jobs = 1):
        """
        Perform Grid Search CV for the given model.
        """
        print(f"\n[INFO] Running GridSearchCV for {model.__class__.__name__}...")
        if scoring is None:
            scoring = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'

        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X, y)

        print(f"[DONE] Best Params: {grid_search.best_params_}")
        print(f"[DONE] Best Score: {grid_search.best_score_}")

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def optuna_search(self, model_class, X, y, objective_func, n_trials=30):
        """
        Perform Optuna-based hyperparameter optimization.
        model_class: sklearn model class (not object)
        objective_func: user-defined function that takes (trial, model_class, X, y)
        """
        print(f"\n[INFO] Running Optuna optimization for {model_class.__name__}...")

        study = optuna.create_study(direction='maximize' if self.problem_type == 'classification' else 'minimize')
        study.optimize(lambda trial: objective_func(trial, model_class, X, y), n_trials=n_trials)

        print(f"[DONE] Best Params: {study.best_params}")
        print(f"[DONE] Best Score: {study.best_value}")

        return study.best_params, study.best_value

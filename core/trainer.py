import numpy as np
import pandas as pd
from core.tuning import HyperparameterTuner
from configs.config_params import param_grids
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor , VotingClassifier

class Train_Model:
    
    def __init__(self, problem_type='regression', test_size=0.2, random_state=42, verbose=True, tuning_enabled = True):
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state
        self.tuning_enabled = tuning_enabled 
        self.verbose = verbose
        self.models = {}
        self.best_model = None
        self.best_metrics = {}
        self.tuner = HyperparameterTuner(problem_type=self.problem_type)


    def train_auto(self, df: pd.DataFrame, target: str):
        
        """
        Automatically train regression models and select the best one.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")
        
        if not pd.api.types.is_numeric_dtype(df[target]):
            raise TypeError(f"❌ Target column '{target}' must be numeric, but got {df[target].dtype}.")

        # Split data
        X = df.drop(columns=[target])
        y = df[target]

        if X.empty:
            raise ValueError("No numeric features found for training.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # --- Available regression models ---
        candidate_models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            "GradientBoosting": GradientBoostingRegressor(random_state=self.random_state)
            # "CustomGradientDescent": None,  # We'll plug our own optimizer here
        }

        results = {}

        # --- Training loop ---
        for name, model in candidate_models.items():
            if self.tuning_enabled and name in param_grids:
                if self.verbose:
                    print(f"Tunning Hyperparameters for {name}...")
                model , best_params , best_score = self.tuner.grid_search(
                    model , param_grids[name] , X_train , y_train)
                if self.verbose:
                    print(f"Best Parameters  for {name} : {best_params}")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
            results[name] = metrics

            if self.verbose:
                print(f"{name} trained → R2: {r2:.4f}, RMSE: {rmse:.4f}")

        sorted_models = sorted(results.items(), key=lambda x: x[1]["R2"], reverse=True)
        top_models = sorted_models[:3]

        if self.verbose:
            print("\nTop 3 models for ensemble:")
            for name, metrics in top_models:
                print(f"  {name} → R2: {metrics['R2']:.4f}")

        # --- Build Voting Regressor ---
        voting_estimators = [(name, candidate_models[name]) for name, _ in top_models]
        voting_regressor = VotingRegressor(estimators=voting_estimators)

        voting_regressor.fit(X_train, y_train)
        y_pred_vote = voting_regressor.predict(X_test)

        mse_vote = mean_squared_error(y_test, y_pred_vote)
        rmse_vote = np.sqrt(mse_vote)
        mae_vote = mean_absolute_error(y_test, y_pred_vote)
        r2_vote = r2_score(y_test, y_pred_vote)

        voting_metrics = {"RMSE": rmse_vote, "MAE": mae_vote, "R2": r2_vote}

        if self.verbose:
            print(f"\nVoting Ensemble → R2: {r2_vote:.4f}, RMSE: {rmse_vote:.4f}")

        # --- Compare voting with best individual model ---
        best_model_name = max(results, key=lambda m: results[m]["R2"])
        if r2_vote > results[best_model_name]["R2"]:
            self.best_model = voting_regressor
            self.best_metrics = voting_metrics
            best_model_name = "VotingEnsemble"
        else:
            self.best_model = candidate_models[best_model_name]
            self.best_metrics = results[best_model_name]

        if self.verbose:
            print(f"\nBest Overall Model → {best_model_name}")
            print("Metrics:", self.best_metrics)

        return self.best_model, self.best_metrics







# core/trainer.py
import logging
import warnings
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Local project imports (assumed to exist)
from core.tuning import HyperparameterTuner
from configs.config_params import param_grids

# ----------------------------
# Minimal Color Helpers (Option C)
# ----------------------------
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"

def color_value(val: float, kind: str) -> str:
    """
    Minimal coloring: only color important values.
    kind: "rmse" | "r2" | "overfit"
    """
    if kind == "rmse":
        # RMSE: highlight if small -> green, else yellow
        if val <= 1.0:
            c = ANSI_GREEN
        elif val <= 10.0:
            c = ANSI_YELLOW
        else:
            c = ANSI_RED
    elif kind == "r2":
        # R2: good => green, moderate => yellow, poor => red
        if val >= 0.9:
            c = ANSI_GREEN
        elif val >= 0.7:
            c = ANSI_YELLOW
        else:
            c = ANSI_RED
    elif kind == "overfit":
        # Overfit factor: close to 1 is good
        if val <= 1.2:
            c = ANSI_GREEN
        elif val <= 1.5:
            c = ANSI_YELLOW
        else:
            c = ANSI_RED
    else:
        c = ""
    return f"{c}{val:.4f}{ANSI_RESET}"

# ----------------------------
# Logging Setup (simple)
# ----------------------------
def get_logger(name: str = "AutoML") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        fmt = "%(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

logger = get_logger("trainer")

# Silence sklearn convergence warnings if desired (we'll still log them)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------
# Pretty Summary (Minimal Professional)
# ----------------------------
def pretty_print_summary(
    model_name: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    overfit_factor: float,
    tuning_enabled: bool,
) -> None:
    sep = "=" * 62
    logger.info("\n" + sep)
    logger.info("                    AUTO ML REGRESSION SUMMARY               ")
    logger.info(sep + "\n")

    logger.info(f"Best Model            : {model_name}")
    logger.info("-" * 62)

    # Only highlight important numeric values (RMSE, R2, Overfit)
    tr_rmse = train_metrics.get("RMSE", float("nan"))
    te_rmse = test_metrics.get("RMSE", float("nan"))
    tr_mae = train_metrics.get("MAE", float("nan"))
    te_mae = test_metrics.get("MAE", float("nan"))
    tr_r2 = train_metrics.get("R2", train_metrics.get("Training_R2", float("nan")))
    te_r2 = test_metrics.get("R2", float("nan"))

    logger.info(f"Training RMSE         : {ANSI_YELLOW}{tr_rmse:.4f}{ANSI_RESET}")
    logger.info(f"Testing  RMSE         : {ANSI_YELLOW}{te_rmse:.4f}{ANSI_RESET}")
    logger.info(f"Training MAE          : {tr_mae:.4f}")
    logger.info(f"Testing  MAE          : {te_mae:.4f}")

    logger.info(f"Training R2 Score     : {color_value(tr_r2, 'r2')}")
    logger.info(f"Testing  R2 Score     : {color_value(te_r2, 'r2')}")

    logger.info("-" * 62)
    logger.info(f"Overfitting Factor    : {color_value(overfit_factor, 'overfit')}")

    if overfit_factor <= 1.2:
        stability = "GOOD"
    elif overfit_factor <= 1.5:
        stability = "MODERATE"
    else:
        stability = "POOR"
    logger.info(f"Model Stability       : {stability}")

    logger.info("-" * 62)
    logger.info("Additional Notes:")
    if tuning_enabled:
        logger.info("✓ Model trained with Hyperparameter Tuning enabled")
    else:
        logger.info("✗ Hyperparameter Tuning disabled")
    logger.info("✓ Ensemble methods (Voting, Stacking, Bagging) evaluated")
    logger.info("✓ Best-performing model automatically selected")
    logger.info(sep + "\n")


# ----------------------------
# Train_Model implementation
# ----------------------------
class Train_Model:
    def __init__(
        self,
        problem_type: str = "regression",
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
        tuning_enabled: bool = True,
        bagging_threshold: float = 2.0,
    ):
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state
        self.tuning_enabled = tuning_enabled
        self.verbose = verbose
        self.bagging_threshold = bagging_threshold

        self.models = {}
        self.best_model = None
        self.best_metrics: Dict[str, float] = {}
        # Tuner expects problem_type arg (classification/regression) — assumed implemented.
        self.tuner = HyperparameterTuner(problem_type=self.problem_type)

    def _safe_tune(self, name: str, model, X, y):
        """Run grid_search if param grid exists; return (model, best_params, best_score)"""
        try:
            if name in param_grids and self.tuning_enabled:
                if self.verbose:
                    logger.info(f"Tuning hyperparameters for {name} ...")
                tuned_model, best_params, best_score = self.tuner.grid_search(
                    model, param_grids[name], X, y
                )
                if self.verbose:
                    logger.info(f"  Best params for {name}: {best_params}")
                return tuned_model, best_params, best_score
        except Exception as e:
            logger.warning(f"Tuner failed for {name}: {e} — using default model.")
        return model, None, None

    def train_auto(self, df: pd.DataFrame, target: str) -> Tuple[Any, Dict[str, float]]:
        """
        Main entry point to automatically train models (regression).
        Returns (best_model_estimator, best_metrics_dict)
        """
        # Basic validations
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        if not pd.api.types.is_numeric_dtype(df[target]):
            raise TypeError(
                f"Target column '{target}' must be numeric, but got {df[target].dtype}."
            )

        X = df.drop(columns=[target])
        y = df[target]

        if X.shape[1] == 0:
            raise ValueError("No features found after dropping target column.")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Candidate models (instances). We'll keep the mapping so we can replace models after tuning.
        candidate_models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(alpha=0.1),
            "RandomForest": RandomForestRegressor(random_state=self.random_state),
            "GradientBoosting": GradientBoostingRegressor(random_state=self.random_state),
            "AdaBoost": AdaBoostRegressor(random_state=self.random_state) ,
            "HistgradientBoosting": HistGradientBoostingRegressor(random_state=self.random_state)
        }

        results: Dict[str, Dict[str, float]] = {}
        overfitting_array: Dict[str, Dict[str, float]] = {}

        # training loop
        for name, model in list(candidate_models.items()):
            # tune if configured
            model, best_params, best_score = self._safe_tune(name, model, X_train, y_train)
            # store back possibly tuned estimator into mapping (important for ensembles later)
            candidate_models[name] = model

            # fit
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Model {name} failed to fit: {e}. Skipping.")
                continue

            # predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # metrics (NOTE: correct r2 usage)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = float(np.sqrt(train_mse))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = float(r2_score(y_train, y_train_pred))

            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = float(np.sqrt(test_mse))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = float(r2_score(y_test, y_test_pred))

            results[name] = {
                "RMSE": test_rmse,
                "MAE": test_mae,
                "R2": test_r2,
                "Training_R2": train_r2,
            }

            # Avoid division by zero when computing overfit factor
            denom = min(train_mse, test_mse)
            ofactor = float(max(train_mse, test_mse) / denom) if denom > 0 else float("inf")
            overfitting_array[name] = {
                "Training": train_mse,
                "Testing": test_mse,
                "OverfittingFactor": ofactor,
            }

            if self.verbose:
                logger.info(
                    f"{name} trained → train R2: {color_value(train_r2, 'r2')}  test R2: {color_value(test_r2, 'r2')}  test RMSE: {test_rmse:.4f}"
                )

        if not results:
            raise RuntimeError("No models were successfully trained.")

        # choose top 3 by test R2 (robust selection)
        sorted_models = sorted(results.items(), key=lambda x: x[1]["R2"], reverse=True)
        top_models = sorted_models[:3]

        if self.verbose:
            logger.info("\nTop 3 models:")
            for name, metrics in top_models:
                logger.info(f"  {name} → test R2: {metrics['R2']:.4f}")

        # ------------------
        # Voting Regressor
        # ------------------
        voting_estimators = [(name, candidate_models[name]) for name, _ in top_models]
        voting_regressor = VotingRegressor(estimators=voting_estimators)
        try:
            voting_regressor.fit(X_train, y_train)
            y_pred_vote = voting_regressor.predict(X_test)
            voting_metrics = {
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_vote))),
                "MAE": float(mean_absolute_error(y_test, y_pred_vote)),
                "R2": float(r2_score(y_test, y_pred_vote)),
            }
            if self.verbose:
                logger.info(f"Voting ensemble → R2: {color_value(voting_metrics['R2'], 'r2')}")
        except Exception as e:
            voting_metrics = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("-inf")}
            logger.warning(f"Voting regressor failed: {e}")

        # ------------------
        # Bagging if overfitting detected
        # ------------------
        most_overfit_model = max(
            overfitting_array.items(), key=lambda kv: kv[1]["OverfittingFactor"]
        )[0]
        overfit_value = overfitting_array[most_overfit_model]["OverfittingFactor"]
        bagging_used = False
        bagging_metrics = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("-inf")}
        bagging_regressor = None

        if self.verbose:
            logger.info(
                f"Most overfitted: {most_overfit_model} -> factor: {color_value(overfit_value, 'overfit')}"
            )

        if overfit_value > self.bagging_threshold:
            if self.verbose:
                logger.info("Overfitting detected — applying Bagging to the overfitted base model.")
            base_model = candidate_models.get(most_overfit_model)
            if base_model is not None:
                bagging_regressor = BaggingRegressor(
                    estimator=base_model, n_estimators=10, random_state=self.random_state
                )
                try:
                    bagging_regressor.fit(X_train, y_train)
                    y_pred_bag = bagging_regressor.predict(X_test)
                    bagging_metrics = {
                        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_bag))),
                        "MAE": float(mean_absolute_error(y_test, y_pred_bag)),
                        "R2": float(r2_score(y_test, y_pred_bag)),
                    }
                    bagging_used = True
                    if self.verbose:
                        logger.info(f"Bagging → R2: {color_value(bagging_metrics['R2'], 'r2')}")
                except Exception as e:
                    logger.warning(f"Bagging failed: {e}")

        # ------------------
        # Stacking
        # ------------------
        stacking_estimators = [(name, candidate_models[name]) for name, _ in top_models]
        stacking_regressor = StackingRegressor(
            estimators=stacking_estimators, final_estimator=Ridge(), passthrough=True, cv=5
        )
        stacking_metrics = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("-inf")}
        try:
            stacking_regressor.fit(X_train, y_train)
            y_pred_stack = stacking_regressor.predict(X_test)
            stacking_metrics = {
                "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_stack))),
                "MAE": float(mean_absolute_error(y_test, y_pred_stack)),
                "R2": float(r2_score(y_test, y_pred_stack)),
            }
            if self.verbose:
                logger.info(f"Stacking → R2: {color_value(stacking_metrics['R2'], 'r2')}")
        except Exception as e:
            logger.warning(f"Stacking failed: {e}")

        # ------------------
        # Final comparison: best single vs voting vs bagging vs stacking
        # ------------------
        best_single_name = max(results, key=lambda m: results[m]["R2"])
        chosen_model = candidate_models[best_single_name]
        chosen_metrics = results[best_single_name]
        best_model_name = best_single_name

        # compare ensembles
        candidates_for_compare = [
            ("VotingEnsemble", voting_metrics, voting_regressor),
            ("BaggingRegressor", bagging_metrics, bagging_regressor) if bagging_used else None,
            ("StackingEnsemble", stacking_metrics, stacking_regressor),
        ]
        for item in filter(None, candidates_for_compare):
            nm, mt, est = item
            if mt["R2"] > chosen_metrics["R2"]:
                chosen_metrics = mt
                chosen_model = est
                best_model_name = nm

        self.best_model = chosen_model
        self.best_metrics = chosen_metrics

        if self.verbose:
            logger.info(f"\nFINAL SELECTED MODEL → {best_model_name}")
            logger.info(f"Final metrics → R2: {color_value(chosen_metrics['R2'], 'r2')}  RMSE: {chosen_metrics['RMSE']:.4f}")

        # compute final train/test metrics for pretty print (try/except guarded)
        try:
            if self.best_model is not None:
                y_train_pred_final = self.best_model.predict(X_train)
                y_test_pred_final = self.best_model.predict(X_test)

                train_mse_final = mean_squared_error(y_train, y_train_pred_final)
                train_rmse_final = float(np.sqrt(train_mse_final))
                train_mae_final = float(mean_absolute_error(y_train, y_train_pred_final))
                train_r2_final = float(r2_score(y_train, y_train_pred_final))

                test_mse_final = mean_squared_error(y_test, y_test_pred_final)
                test_rmse_final = float(np.sqrt(test_mse_final))
                test_mae_final = float(mean_absolute_error(y_test, y_test_pred_final))
                test_r2_final = float(r2_score(y_test, y_test_pred_final))

                final_train_metrics = {"RMSE": train_rmse_final, "MAE": train_mae_final, "R2": train_r2_final}
                final_test_metrics = {"RMSE": test_rmse_final, "MAE": test_mae_final, "R2": test_r2_final}
                final_overfit = (
                    max(train_mse_final, test_mse_final) / min(train_mse_final, test_mse_final)
                    if min(train_mse_final, test_mse_final) > 0
                    else float("inf")
                )
            else:
                final_train_metrics, final_test_metrics, final_overfit = {}, {}, overfit_value
        except Exception as e:
            logger.warning(f"Could not compute final metrics: {e}")
            final_train_metrics, final_test_metrics, final_overfit = {}, {}, overfit_value

        # Pretty print minimal professional summary
        pretty_print_summary(
            best_model_name,
            final_train_metrics,
            final_test_metrics,
            final_overfit,
            self.tuning_enabled,
        )

        return self.best_model, self.best_metrics

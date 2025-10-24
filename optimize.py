import json
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def get_best_model(experiment_id: str):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    return best_model


def optimize_model():
    df = pd.read_csv("water_potability.csv")
    X = df.drop(columns=["Potability"])
    y = df["Potability"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=10,
    )

    experiment_name = f"XGBoost_Optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 10.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
        }

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", XGBClassifier(
                    random_state=10,
                    **params,
                )),
            ]
        )

        run_name = f"XGBoost_trial_{trial.number}_lr_{params['learning_rate']:.3f}_depth_{params['max_depth']}"

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.log_params(params)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_valid)
            metrics = {
                "valid_f1": f1_score(y_valid, y_pred),
            }
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(pipeline, "model")

            return metrics["valid_f1"]

    study = optuna.create_study(
        study_name="xgboost_water_potability",
        direction="maximize",
    )
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    best_model = get_best_model(experiment_id)

    with open(os.path.join("models", "best_model.pkl"), "wb") as model_file:
        pickle.dump(best_model, model_file)

    versions = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "mlflow": mlflow.__version__,
        "optuna": optuna.__version__,
    }
    with open(os.path.join("models", "library_versions.json"), "w", encoding="utf-8") as version_file:
        json.dump(versions, version_file, indent=4)

    best_run_id = mlflow.search_runs(experiment_id).sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    
    with mlflow.start_run(run_id=best_run_id):
        importances = best_model.named_steps["classifier"].feature_importances_
        order = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.barh(X_train.columns[order], importances[order])
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("Feature Importance")
        plt.tight_layout()
        feature_plot_path = os.path.join("plots", "feature_importance.png")
        plt.savefig(feature_plot_path, dpi=150)
        plt.close()

        best_params_path = os.path.join("plots", "best_model_params.json")
        with open(best_params_path, "w", encoding="utf-8") as params_file:
            json.dump(study.best_trial.params, params_file, indent=4)

        history_fig = plot_optimization_history(study).figure
        history_path = os.path.join("plots", "optimization_history.png")
        history_fig.savefig(history_path, dpi=150, bbox_inches="tight")
        plt.close(history_fig)

        importance_fig = plot_param_importances(study).figure
        importance_path = os.path.join("plots", "param_importances.png")
        importance_fig.savefig(importance_path, dpi=150, bbox_inches="tight")
        plt.close(importance_fig)

        mlflow.log_artifact(feature_plot_path, "plots")
        mlflow.log_artifact(history_path, "plots")
        mlflow.log_artifact(importance_path, "plots")
        mlflow.log_artifact(best_params_path, "plots")
        mlflow.log_artifact(os.path.join("models", "best_model.pkl"), "models")
        mlflow.log_artifact(os.path.join("models", "library_versions.json"), "models")

    return best_model


if __name__ == "__main__":
    optimize_model()

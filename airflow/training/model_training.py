import joblib
import mlflow
import optuna
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

RANDOM_STATE = 10

def calculate_scale_pos_weight(y_train):
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = neg / max(1, pos)
    return scale_pos_weight


def optimize_hyperparameters_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight,
        }
        
        clf = LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_score)
        
        trial.report(pr_auc, step=0)
        return pr_auc
    
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def train_lgbm_model(X_train, y_train, best_params=None):
    if best_params is None:
        scale_pos_weight = calculate_scale_pos_weight(y_train)
        best_params = {
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight,
        }
    
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_score = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'pr_auc': average_precision_score(y_val, y_score),
        'f1_score': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
    }
    
    return metrics


def log_model_to_mlflow(model, prep_pipeline, params, metrics, model_name="lgbm_model"):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        full_pipeline = Pipeline(steps=[
            ('prep', prep_pipeline),
            ('clf', model),
        ])
        
        mlflow.sklearn.log_model(full_pipeline, model_name)
        run_id = mlflow.active_run().info.run_id
        
    return run_id


def load_best_model_from_mlflow(experiment_name="Default"):
    mlflow.set_experiment(experiment_name)
    
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.pr_auc DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        return None
    
    best_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{best_run_id}/lgbm_model"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model, best_run_id


def save_model_locally(model, prep_pipeline, filepath):
    full_pipeline = Pipeline(steps=[
        ('prep', prep_pipeline),
        ('clf', model),
    ])
    joblib.dump(full_pipeline, filepath)


def load_model_locally(filepath):
    return joblib.load(filepath)


def retrain_pipeline(X_train, y_train, X_val, y_val, prep_pipeline, 
                     use_optuna=True, n_trials=20, log_to_mlflow=True):
    
    if use_optuna:
        best_params, best_pr_auc = optimize_hyperparameters_optuna(
            X_train, y_train, X_val, y_val, n_trials=n_trials
        )
    else:
        scale_pos_weight = calculate_scale_pos_weight(y_train)
        best_params = {
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'scale_pos_weight': scale_pos_weight,
        }
    
    model = train_lgbm_model(X_train, y_train, best_params)
    metrics = evaluate_model(model, X_val, y_val)
    
    if log_to_mlflow:
        run_id = log_model_to_mlflow(model, prep_pipeline, best_params, metrics)
        return model, best_params, metrics, run_id
    
    return model, best_params, metrics, None
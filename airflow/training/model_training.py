import joblib
import optuna
import warnings
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# Suprimir warnings molestos
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=FutureWarning)

RANDOM_STATE = 10


def optimize_hyperparameters_optuna(X_train, y_train, X_val, y_val, n_trials=20):
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
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        }
        
        clf = LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, y_score)
        
        trial.report(ap, step=0)
        return ap
    
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study.best_value


def train_lgbm_model(X_train, y_train, best_params=None):
    if best_params is None:
        best_params = {
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'scale_pos_weight': 1.0,
        }
    
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model


def find_optimal_threshold(y_val, y_score):
    thresholds = np.arange(0.01, 0.95, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    return best_threshold


def evaluate_model(model, X_val, y_val, threshold=0.5):
    y_score = model.predict_proba(X_val)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    
    metrics = {
        'pr_auc': average_precision_score(y_val, y_score),
        'f1_score': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'threshold': threshold
    }
    
    return metrics


def save_model_locally(model, prep_pipeline, filepath):
    full_pipeline = Pipeline(steps=[
        ('prep', prep_pipeline),
        ('clf', model),
    ])
    joblib.dump(full_pipeline, filepath)


def load_model_locally(filepath):
    return joblib.load(filepath)


def retrain_pipeline(X_train, y_train, X_val, y_val, prep_pipeline, 
                     use_optuna=True, n_trials=20, log_to_mlflow=False):
    
    if use_optuna:
        best_params, best_metric = optimize_hyperparameters_optuna(
            X_train, y_train, X_val, y_val, n_trials=n_trials
        )
    else:
        best_params = {
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'scale_pos_weight': 1.0,
        }
    
    model = train_lgbm_model(X_train, y_train, best_params)
    
    # Find optimal threshold
    y_score_val = model.predict_proba(X_val)[:, 1]
    best_threshold = find_optimal_threshold(y_val, y_score_val)
    
    metrics = evaluate_model(model, X_val, y_val, threshold=best_threshold)
    
    return model, best_params, metrics
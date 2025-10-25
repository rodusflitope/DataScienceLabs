from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import datetime
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    DATA_PATH,
    MODELS_PATH,
    PREDICTIONS_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    CLIENTES_FILENAME,
    PRODUCTOS_FILENAME,
    TRANSACCIONES_FILENAME,
)
from training.data_processing import (
    load_data, clean_data, split_temporal_data,
    WeeklyDatasetBuilder, build_preprocessing_pipeline, prepare_datasets
)
from training.model_training import retrain_pipeline
from training.drift_detection import detect_drift_in_transactions
from training.prediction import generate_predictions_for_next_week
from training.interpretability import log_shap_to_mlflow
import mlflow

args = {
    'owner': 'airflow',
    'retries': 1
}
mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")

def extract_data_task(**context):
    print("Extrayendo datos")
    clientes, productos, transacciones = load_data(
        DATA_PATH,
        CLIENTES_FILENAME,
        PRODUCTOS_FILENAME,
        TRANSACCIONES_FILENAME,
    )
    
    with open(f'{MODELS_PATH}/raw_clientes.pkl', 'wb') as f:
        pickle.dump(clientes, f)
    with open(f'{MODELS_PATH}/raw_productos.pkl', 'wb') as f:
        pickle.dump(productos, f)
    with open(f'{MODELS_PATH}/raw_transacciones.pkl', 'wb') as f:
        pickle.dump(transacciones, f)

def clean_data_task(**context):
    with open(f'{MODELS_PATH}/raw_clientes.pkl', 'rb') as f:
        clientes = pickle.load(f)
    with open(f'{MODELS_PATH}/raw_productos.pkl', 'rb') as f:
        productos = pickle.load(f)
    with open(f'{MODELS_PATH}/raw_transacciones.pkl', 'rb') as f:
        transacciones = pickle.load(f)
    
    clientes_clean, productos_clean, transacciones_clean = clean_data(
        clientes, productos, transacciones
    )
    
    with open(f'{MODELS_PATH}/clean_clientes.pkl', 'wb') as f:
        pickle.dump(clientes_clean, f)
    with open(f'{MODELS_PATH}/clean_productos.pkl', 'wb') as f:
        pickle.dump(productos_clean, f)
    with open(f'{MODELS_PATH}/clean_transacciones.pkl', 'wb') as f:
        pickle.dump(transacciones_clean, f)

def detect_drift_task(**context):
    with open(f'{MODELS_PATH}/clean_transacciones.pkl', 'rb') as f:
        new_transacciones = pickle.load(f)
    
    if os.path.exists(f'{MODELS_PATH}/reference_transacciones.pkl'):
        with open(f'{MODELS_PATH}/reference_transacciones.pkl', 'rb') as f:
            reference_transacciones = pickle.load(f)
        
        with open(f'{MODELS_PATH}/clean_clientes.pkl', 'rb') as f:
            clientes = pickle.load(f)
        with open(f'{MODELS_PATH}/clean_productos.pkl', 'rb') as f:
            productos = pickle.load(f)
        
        ref_merged = reference_transacciones.merge(clientes, on='customer_id', how='left')
        ref_merged = ref_merged.merge(productos, on='product_id', how='left')
        
        new_merged = new_transacciones.merge(clientes, on='customer_id', how='left')
        new_merged = new_merged.merge(productos, on='product_id', how='left')
        
        drift_detected = detect_drift_in_transactions(ref_merged, new_merged, threshold=0.05)
        
        context['ti'].xcom_push(key='drift_detected', value=drift_detected)
        
        if drift_detected:
            return 'build_weekly_dataset'
        else:
            return 'skip_training'
    else:
        context['ti'].xcom_push(key='drift_detected', value=True)
        return 'build_weekly_dataset'

def build_weekly_dataset_task(**context):
    with open(f'{MODELS_PATH}/clean_clientes.pkl', 'rb') as f:
        clientes = pickle.load(f)
    with open(f'{MODELS_PATH}/clean_productos.pkl', 'rb') as f:
        productos = pickle.load(f)
    with open(f'{MODELS_PATH}/clean_transacciones.pkl', 'rb') as f:
        transacciones = pickle.load(f)
    
    train_trans, val_trans, test_trans = split_temporal_data(
        transacciones, train_ratio=0.70, val_ratio=0.15
    )
    
    builder = WeeklyDatasetBuilder()
    
    train_dataset = builder.transform({
        'transacciones': train_trans,
        'clientes': clientes,
        'productos': productos
    })
    
    val_dataset = builder.transform({
        'transacciones': val_trans,
        'clientes': clientes,
        'productos': productos
    })
    
    with open(f'{MODELS_PATH}/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(f'{MODELS_PATH}/val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)

def train_model_task(**context):
    with open(f'{MODELS_PATH}/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(f'{MODELS_PATH}/val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    
    prep_pipeline = build_preprocessing_pipeline()
    
    X_train, y_train = prepare_datasets(train_dataset, prep_pipeline, fit=True)
    X_val, y_val = prepare_datasets(val_dataset, prep_pipeline, fit=False)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model, best_params, metrics, run_id = retrain_pipeline(
        X_train, y_train, X_val, y_val, prep_pipeline,
        use_optuna=True, n_trials=10, log_to_mlflow=True
    )
    
    with mlflow.start_run(run_id=run_id):
        feature_cols = [
            'region_id', 'customer_type', 'brand', 'category', 'sub_category',
            'segment', 'package', 'size', 'num_deliver_per_week', 
            'num_visit_per_week', 'week'
        ]
        X_sample = train_dataset[feature_cols].sample(min(200, len(train_dataset)))
        
        log_shap_to_mlflow(model, X_sample)
    with open(f'{MODELS_PATH}/prep_pipeline.pkl', 'wb') as f:
        pickle.dump(prep_pipeline, f)
    with open(f'{MODELS_PATH}/clean_transacciones.pkl', 'rb') as f:
        transacciones = pickle.load(f)
    with open(f'{MODELS_PATH}/reference_transacciones.pkl', 'wb') as f:
        pickle.dump(transacciones, f)
    context['ti'].xcom_push(key='run_id', value=run_id)
    context['ti'].xcom_push(key='metrics', value=metrics)

def generate_predictions_task(**context):
    with open(f'{MODELS_PATH}/clean_clientes.pkl', 'rb') as f:
        clientes = pickle.load(f)
    with open(f'{MODELS_PATH}/clean_productos.pkl', 'rb') as f:
        productos = pickle.load(f)
    with open(f'{MODELS_PATH}/clean_transacciones.pkl', 'rb') as f:
        transacciones = pickle.load(f)
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        order_by=["metrics.pr_auc DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise ValueError(f"No runs found in experiment {MLFLOW_EXPERIMENT_NAME}")
    
    best_run_id = runs.iloc[0]['run_id']
    model_uri = f"runs:/{best_run_id}/lgbm_model"
    full_pipeline = mlflow.sklearn.load_model(model_uri)
    
    predictions = generate_predictions_for_next_week(
        clientes, productos, transacciones, 
        full_pipeline, prep_pipeline=None
    )
    
    predictions_file = f'{PREDICTIONS_PATH}/predictions_next_week.parquet'
    predictions.to_parquet(predictions_file, index=False)
    
    context['ti'].xcom_push(key='predictions_file', value=predictions_file)
    context['ti'].xcom_push(key='n_predictions', value=len(predictions))

with DAG(
    dag_id='ml_pipeline',
    default_args=args,
    description='Pipeline ML con detección de drift y reentrenamiento',
    start_date=datetime(2024, 1, 1),
    schedule=None) as dag:

    extract_data = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_task
    )

    clean_data_op = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data_task
    )

    detect_drift = BranchPythonOperator(
        task_id='detect_drift',
        python_callable=detect_drift_task
    )

    skip_training = EmptyOperator(
        task_id='skip_training'
    )

    build_dataset = PythonOperator(
        task_id='build_weekly_dataset',
        python_callable=build_weekly_dataset_task
    )

    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task
    )

    join_task = EmptyOperator(
        task_id='join_paths',
        trigger_rule='none_failed_min_one_success'
    )

    generate_predictions = PythonOperator(
        task_id='generate_predictions',
        python_callable=generate_predictions_task
    )

    extract_data >> clean_data_op >> detect_drift
    detect_drift >> [build_dataset, skip_training]
    build_dataset >> train_model >> join_task
    skip_training >> join_task
    join_task >> generate_predictions

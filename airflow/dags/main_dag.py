from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from pathlib import Path
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    DATA_PATH,
    MODELS_PATH,
    PREDICTIONS_PATH,
    CLIENTES_FILENAME,
    PRODUCTOS_FILENAME,
    TRANSACCIONES_FILENAME,
    OPTUNA_N_TRIALS,
)
from training.data_processing import (
    load_data, clean_data, split_temporal_data,
    WeeklyDatasetBuilder, build_preprocessing_pipeline, prepare_datasets
)
from training.model_training import retrain_pipeline
from training.drift_detection import detect_drift_in_transactions
from training.prediction import generate_predictions_for_next_week

MODELS_DIR = Path(MODELS_PATH)
PREDICTIONS_DIR = Path(PREDICTIONS_PATH)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

def extract_data_task(**context):
    print("Extrayendo datos")
    clientes, productos, transacciones = load_data(
        DATA_PATH,
        CLIENTES_FILENAME,
        PRODUCTOS_FILENAME,
        TRANSACCIONES_FILENAME,
    )
    
    with open(MODELS_DIR / 'raw_clientes.pkl', 'wb') as f:
        pickle.dump(clientes, f)
    with open(MODELS_DIR / 'raw_productos.pkl', 'wb') as f:
        pickle.dump(productos, f)
    with open(MODELS_DIR / 'raw_transacciones.pkl', 'wb') as f:
        pickle.dump(transacciones, f)

def clean_data_task(**context):
    with open(MODELS_DIR / 'raw_clientes.pkl', 'rb') as f:
        clientes = pickle.load(f)
    with open(MODELS_DIR / 'raw_productos.pkl', 'rb') as f:
        productos = pickle.load(f)
    with open(MODELS_DIR / 'raw_transacciones.pkl', 'rb') as f:
        transacciones = pickle.load(f)
    
    clientes_clean, productos_clean, transacciones_clean = clean_data(
        clientes, productos, transacciones
    )
    
    with open(MODELS_DIR / 'clean_clientes.pkl', 'wb') as f:
        pickle.dump(clientes_clean, f)
    with open(MODELS_DIR / 'clean_productos.pkl', 'wb') as f:
        pickle.dump(productos_clean, f)
    with open(MODELS_DIR / 'clean_transacciones.pkl', 'wb') as f:
        pickle.dump(transacciones_clean, f)

def detect_drift_task(**context):
    with open(MODELS_DIR / 'clean_transacciones.pkl', 'rb') as f:
        new_transacciones = pickle.load(f)
    
    if os.path.exists(MODELS_DIR / 'reference_transacciones.pkl'):
        with open(MODELS_DIR / 'reference_transacciones.pkl', 'rb') as f:
            reference_transacciones = pickle.load(f)
        
        with open(MODELS_DIR / 'clean_clientes.pkl', 'rb') as f:
            clientes = pickle.load(f)
        with open(MODELS_DIR / 'clean_productos.pkl', 'rb') as f:
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
    with open(MODELS_DIR / 'clean_clientes.pkl', 'rb') as f:
        clientes = pickle.load(f)
    with open(MODELS_DIR / 'clean_productos.pkl', 'rb') as f:
        productos = pickle.load(f)
    with open(MODELS_DIR / 'clean_transacciones.pkl', 'rb') as f:
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
    
    with open(MODELS_DIR / 'train_dataset.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(MODELS_DIR / 'val_dataset.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)

def train_model_task(**context):
    with open(MODELS_DIR / 'train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(MODELS_DIR / 'val_dataset.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    
    prep_pipeline = build_preprocessing_pipeline()
    
    X_train, y_train = prepare_datasets(train_dataset, prep_pipeline, fit=True)
    X_val, y_val = prepare_datasets(val_dataset, prep_pipeline, fit=False)
    
    model, best_params, metrics = retrain_pipeline(
        X_train, y_train, X_val, y_val, prep_pipeline,
        use_optuna=True, n_trials=OPTUNA_N_TRIALS, log_to_mlflow=False
    )
    
    with open(MODELS_DIR / 'prep_pipeline.pkl', 'wb') as f:
        pickle.dump(prep_pipeline, f)
    with open(MODELS_DIR / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(MODELS_DIR / 'clean_transacciones.pkl', 'rb') as f:
        transacciones = pickle.load(f)
    with open(MODELS_DIR / 'reference_transacciones.pkl', 'wb') as f:
        pickle.dump(transacciones, f)
    
    context['ti'].xcom_push(key='metrics', value=metrics)

def generate_predictions_task(**context):
    with open(MODELS_DIR / 'clean_clientes.pkl', 'rb') as f:
        clientes = pickle.load(f)
    with open(MODELS_DIR / 'clean_productos.pkl', 'rb') as f:
        productos = pickle.load(f)
    with open(MODELS_DIR / 'clean_transacciones.pkl', 'rb') as f:
        transacciones = pickle.load(f)
    
    # Cargar modelo desde archivo local
    with open(MODELS_DIR / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(MODELS_DIR / 'prep_pipeline.pkl', 'rb') as f:
        prep_pipeline = pickle.load(f)
    
    from sklearn.pipeline import Pipeline
    full_pipeline = Pipeline(steps=[
        ('prep', prep_pipeline),
        ('clf', model),
    ])
    
    predictions = generate_predictions_for_next_week(
        clientes, productos, transacciones, 
        full_pipeline, prep_pipeline=None
    )
    
    predictions_file = PREDICTIONS_DIR / 'predictions_next_week.parquet'
    predictions.to_parquet(predictions_file, index=False)
    
    context['ti'].xcom_push(key='predictions_file', value=str(predictions_file))
    context['ti'].xcom_push(key='n_predictions', value=len(predictions))

with DAG(
    dag_id='ml_pipeline',
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

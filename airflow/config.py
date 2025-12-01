import os

DATA_PATH = os.environ.get('ML_PIPELINE_DATA_PATH', '/opt/airflow/data')
MODELS_PATH = os.environ.get('ML_PIPELINE_MODELS_PATH', '/opt/airflow/storage/models')
PREDICTIONS_PATH = os.environ.get('ML_PIPELINE_PREDICTIONS_PATH', '/opt/airflow/storage/predictions')
CLIENTES_FILENAME = os.environ.get('ML_PIPELINE_CLIENTES_FILE', 'clientes.parquet')
PRODUCTOS_FILENAME = os.environ.get('ML_PIPELINE_PRODUCTOS_FILE', 'productos.parquet')
TRANSACCIONES_FILENAME = os.environ.get('ML_PIPELINE_TRANSACCIONES_FILE', 'transacciones.parquet')

OPTUNA_N_TRIALS = int(os.environ.get('ML_PIPELINE_OPTUNA_N_TRIALS', '20'))
TOP_K_PREDICTIONS = os.environ.get('ML_PIPELINE_TOP_K_PREDICTIONS')
if TOP_K_PREDICTIONS is not None:
    TOP_K_PREDICTIONS = int(TOP_K_PREDICTIONS)

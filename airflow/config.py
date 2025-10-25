import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.environ.get('ML_PIPELINE_DATA_PATH', os.path.join(BASE_PATH, 'data'))
MODELS_PATH = os.environ.get('ML_PIPELINE_MODELS_PATH', os.path.join(BASE_PATH, 'airflow', 'models'))
PREDICTIONS_PATH = os.environ.get('ML_PIPELINE_PREDICTIONS_PATH', os.path.join(BASE_PATH, 'airflow', 'predictions'))
MLFLOW_TRACKING_URI = os.environ.get('ML_PIPELINE_MLFLOW_TRACKING_URI', os.path.join(BASE_PATH, 'airflow', 'mlruns'))
MLFLOW_EXPERIMENT_NAME = os.environ.get('ML_PIPELINE_MLFLOW_EXPERIMENT', 'customer_product_prediction')
CLIENTES_FILENAME = os.environ.get('ML_PIPELINE_CLIENTES_FILE', 'clientes.parquet')
PRODUCTOS_FILENAME = os.environ.get('ML_PIPELINE_PRODUCTOS_FILE', 'productos.parquet')
TRANSACCIONES_FILENAME = os.environ.get('ML_PIPELINE_TRANSACCIONES_FILE', 'transacciones.parquet')

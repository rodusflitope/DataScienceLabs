from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# Nuestras funciones del punto 2.1
from hiring_dynamic_functions import (
    create_folders,
    load_and_merge,
    split_data,
    train_model,
    evaluate_models,
)

# ---------------------------
# URLs de datos
# ---------------------------
URL_DATA_1 = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
URL_DATA_2 = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"

# ---------------------------
# Helpers para entrenamientos (evita pasar objetos por XCom)
# ---------------------------
def train_rf_callable(**kwargs):
    from sklearn.ensemble import RandomForestClassifier
    est = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    return train_model(estimator=est, **kwargs)

def train_lr_callable(**kwargs):
    from sklearn.linear_model import LogisticRegression
    est = LogisticRegression(max_iter=1000, n_jobs=None)  # n_jobs no en todos los backends
    return train_model(estimator=est, **kwargs)

def train_gb_callable(**kwargs):
    from sklearn.ensemble import GradientBoostingClassifier
    est = GradientBoostingClassifier(random_state=42)
    return train_model(estimator=est, **kwargs)

# ---------------------------
# Branching por fecha
#   - Antes de 2024-11-01  -> solo data_1
#   - Desde  2024-11-01    -> data_1 y data_2
# ---------------------------
def decide_download(**kwargs):
    ds = kwargs["ds"]  # 'YYYY-MM-DD'
    if ds < "2024-11-01":
        return "download_data1"
    else:
        return ["download_data1", "download_data2"]

with DAG(
    dag_id="hiring_dynamic",
    start_date=datetime(2024, 10, 1),
    schedule_interval="0 15 5 * *",   # día 5 a las 15:00 UTC
    catchup=True,                      # backfill habilitado
    tags=["lab9", "dynamic", "parallel"],
    default_args={"owner": "airflow"},
) as dag:

    start = EmptyOperator(task_id="start")

    mk_dirs = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
    )

    # Decide qué descargar según la fecha de ejecución
    branch = BranchPythonOperator(
        task_id="branching_downloads",
        python_callable=decide_download,
    )

    # Descarga a la carpeta raw de la corrida actual (ruta absoluta en contenedor)
    download_data1 = BashOperator(
        task_id="download_data1",
        bash_command=(
            "mkdir -p /opt/airflow/data/{{ ds_nodash }}/raw && "
            f"curl -L -o /opt/airflow/data/{{{{ ds_nodash }}}}/raw/data_1.csv {URL_DATA_1}"
        ),
    )

    download_data2 = BashOperator(
        task_id="download_data2",
        bash_command=(
            "mkdir -p /opt/airflow/data/{{ ds_nodash }}/raw && "
            f"curl -L -o /opt/airflow/data/{{{{ ds_nodash }}}}/raw/data_2.csv {URL_DATA_2}"
        ),
    )

    # Concatena si al menos UNO se descargó
    merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # corre si al menos 1 upstream tuvo éxito
    )

    # Hold-out 80/20 (estratificado dentro de la función)
    split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"test_size": 0.20, "random_state": 42},
    )

    # --- Entrenamientos en paralelo ---
    train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_rf_callable,
    )
    train_lr = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_lr_callable,
    )
    train_gb = PythonOperator(
        task_id="train_gradient_boosting",
        python_callable=train_gb_callable,
    )

    # Evalúa SOLO cuando los 3 entrenamientos terminaron con éxito
    select_best = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # explícito (por claridad)
    )

    end = EmptyOperator(task_id="end")

    # --------- Dependencias ---------
    # start >> mk_dirs >> branch >> [download_data1, download_data2] >> merge >> split >> [train_rf, train_lr, train_gb] >> select_best >> end

    start >> mk_dirs >> branch
    branch >> download_data1
    branch >> download_data2
    [download_data1, download_data2] >> merge >> split
    split >> [train_rf, train_lr, train_gb] >> select_best >> end

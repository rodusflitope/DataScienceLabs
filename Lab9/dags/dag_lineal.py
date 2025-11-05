from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
# from airflow.utils.dates import days_ago

# %% 
# Importa funciones
from hiring_functions import (
    create_folders,
    split_data,
    preprocess_and_train,
    gradio_interface,
)

# %%
# URL de datos

with DAG(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1), # inicio octubre 2024
    schedule_interval=None,   # ejecución manual
    catchup=False,            # sin backfill
    tags=["lab9", "airflow", "hiring"],
) as dag:

    start_pipeline = EmptyOperator(task_id="start_pipeline")

    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,  # kwargs se pasan automáticamente
    )

    download_data = BashOperator(
    task_id="download_data",
    bash_command=(
        "mkdir -p /opt/airflow/data/{{ ds_nodash }}/raw && "
        "curl -L -o /opt/airflow/data/{{ ds_nodash }}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    ),
)
    

    split_data_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"test_size": 0.20, "random_state": 42},
    )

    preprocess_and_train_task = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={"random_state": 42, "n_estimators": 300},
    )

    # Levanta Gradio usando la ruta del modelo que retorna t3 vía XCom
    gradio_ui = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
        op_kwargs={
            "model_path": "{{ ti.xcom_pull(task_ids='preprocess_and_train') }}"
        },
    )

    # Dependencias 
    start_pipeline >> create_folders_task >> download_data >> split_data_task >> preprocess_and_train_task >> gradio_ui

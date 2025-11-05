# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

ENV AIRFLOW_HOME=/opt/airflow
ENV PATH="$AIRFLOW_HOME/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
 && rm -rf /var/lib/apt/lists/*

ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.10
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
 && pip install --no-cache-dir pandas scikit-learn joblib gradio

WORKDIR ${AIRFLOW_HOME}
RUN mkdir -p dags logs plugins data

COPY dags/ dags/
COPY plugins/ plugins/

EXPOSE 8080

CMD ["bash", "-lc", "airflow standalone"]

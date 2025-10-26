FROM apache/airflow:2.10.2-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgomp1 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow
COPY airflow/requirements.txt /tmp/reqs.txt
RUN pip install --no-cache-dir -r /tmp/reqs.txt

WORKDIR /opt/airflow

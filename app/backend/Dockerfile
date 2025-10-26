FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV MODEL_PATH=/app/models/model.pkl
ENV PREP_PIPELINE_PATH=/app/models/prep_pipeline.pkl
ENV CLIENTES_PATH=/app/models/clean_clientes.pkl
ENV PRODUCTOS_PATH=/app/models/clean_productos.pkl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

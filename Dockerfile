FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY models/ models/

EXPOSE 8000

VOLUME ["/app/data"]

CMD ["python", "main.py"]

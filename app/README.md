# SodAI Drinks - Aplicación Web

Esta aplicación web permite realizar predicciones sobre si un cliente comprará un producto específico en una semana determinada.

## Estructura

```
app/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
└── docker-compose.yml
```

## Componentes

### Backend (FastAPI)
- **Puerto:** 8000
- **Endpoints:**
  - `GET /`: Mensaje de bienvenida
  - `GET /health`: Verificación de salud
  - `POST /predict`: Realiza una predicción individual
  - `POST /predict/batch`: Realiza predicciones en batch
  - `GET /customers`: Lista los primeros 100 clientes
  - `GET /products`: Lista los primeros 100 productos

### Frontend (Gradio)
- **Puerto:** 7860
- Interfaz web interactiva para realizar predicciones
- Campos de entrada: Customer ID, Product ID, Week
- Muestra resultado de predicción y probabilidad

## Instrucciones de uso

### 1. Levantar la aplicación con Docker Compose

```bash
cd app
docker-compose up --build
```

### 2. Acceder a la aplicación

- **Frontend (Gradio):** http://localhost:7860
- **Backend (FastAPI):** http://localhost:8000
- **Documentación API:** http://localhost:8000/docs

### 3. Detener la aplicación

```bash
docker-compose down
```

## Requisitos previos

- Docker instalado
- Docker Compose instalado
- Modelos entrenados en `../airflow/storage/models/`

## Volúmenes

La aplicación monta el directorio `../airflow/storage/models/` como volumen para acceder a:
- `model.pkl`: Modelo entrenado
- `clean_clientes.pkl`: Datos de clientes
- `clean_productos.pkl`: Datos de productos

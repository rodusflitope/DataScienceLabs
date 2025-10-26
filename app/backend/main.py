from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.pkl")
PREP_PIPELINE_PATH = os.getenv("PREP_PIPELINE_PATH", "/app/models/prep_pipeline.pkl")
CLIENTES_PATH = os.getenv("CLIENTES_PATH", "/app/models/clean_clientes.pkl")
PRODUCTOS_PATH = os.getenv("PRODUCTOS_PATH", "/app/models/clean_productos.pkl")

model = None
prep_pipeline = None
clientes_df = None
productos_df = None

@app.on_event("startup")
def load_artifacts():
    global model, prep_pipeline, clientes_df, productos_df
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(PREP_PIPELINE_PATH, 'rb') as f:
            prep_pipeline = pickle.load(f)
        with open(CLIENTES_PATH, 'rb') as f:
            clientes_df = pickle.load(f)
        with open(PRODUCTOS_PATH, 'rb') as f:
            productos_df = pickle.load(f)
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class PredictionRequest(BaseModel):
    customer_id: int
    product_id: int
    week: int

class PredictionResponse(BaseModel):
    customer_id: int
    product_id: int
    week: int
    prediction: int
    prediction_proba: float

@app.get("/")
def read_root():
    return {"message": "SodAI Drinks API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or prep_pipeline is None or clientes_df is None or productos_df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    cliente_info = clientes_df[clientes_df['customer_id'] == request.customer_id]
    producto_info = productos_df[productos_df['product_id'] == request.product_id]
    
    if cliente_info.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    if producto_info.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    
    input_data = pd.DataFrame({
        'region_id': [cliente_info['region_id'].values[0]],
        'customer_type': [cliente_info['customer_type'].values[0]],
        'brand': [producto_info['brand'].values[0]],
        'category': [producto_info['category'].values[0]],
        'sub_category': [producto_info['sub_category'].values[0]],
        'segment': [producto_info['segment'].values[0]],
        'package': [producto_info['package'].values[0]],
        'size': [producto_info['size'].values[0]],
        'num_deliver_per_week': [cliente_info['num_deliver_per_week'].values[0]],
        'num_visit_per_week': [cliente_info['num_visit_per_week'].values[0]],
        'week': [request.week]
    })
    
    X_transformed = prep_pipeline.transform(input_data)
    prediction_proba = model.predict_proba(X_transformed)[0, 1]
    prediction = int(model.predict(X_transformed)[0])
    
    return PredictionResponse(
        customer_id=request.customer_id,
        product_id=request.product_id,
        week=request.week,
        prediction=prediction,
        prediction_proba=float(prediction_proba)
    )

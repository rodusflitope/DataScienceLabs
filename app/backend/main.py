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
METADATA_PATH = os.getenv("METADATA_PATH", "/app/models/metadata.pkl")

model = None
prep_pipeline = None
clientes_df = None
productos_df = None
next_week = None

@app.on_event("startup")
def load_artifacts():
    global model, prep_pipeline, clientes_df, productos_df, next_week
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(PREP_PIPELINE_PATH, 'rb') as f:
            prep_pipeline = pickle.load(f)
        with open(CLIENTES_PATH, 'rb') as f:
            clientes_df = pickle.load(f)
        with open(PRODUCTOS_PATH, 'rb') as f:
            productos_df = pickle.load(f)
        
        # Load metadata with next_week
        try:
            with open(METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
                next_week = metadata['next_week']
                print(f"Loaded metadata. Next week to predict: {next_week}")
        except FileNotFoundError:
            print("Warning: metadata.pkl not found, using default next_week=78")
            next_week = 78
        
        print(f"All artifacts loaded successfully. Next week: {next_week}")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class PredictionRequest(BaseModel):
    customer_id: int
    product_id: int

class PredictionResponse(BaseModel):
    customer_id: int
    product_id: int
    week: int
    prediction: int
    prediction_proba: float

class ProductInfo(BaseModel):
    product_id: int
    product_name: str
    brand: str
    category: str
    sub_category: str

class AggregateRequest(BaseModel):
    product_id: int

class AggregateResponse(BaseModel):
    product_id: int
    product_name: str
    week: int
    total_customers: int
    expected_buyers: int
    expected_percentage: float

@app.get("/")
def read_root():
    return {"message": "SodAI Drinks API"}

@app.get("/products")
def get_products():
    if productos_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    products = []
    for _, row in productos_df.iterrows():
        products.append({
            "product_id": int(row['product_id']),
            "product_name": f"{row['brand']} - {row['category']} - {row['sub_category']}",
            "brand": row['brand'],
            "category": row['category'],
            "sub_category": row['sub_category']
        })
    return {"products": products}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or prep_pipeline is None or clientes_df is None or productos_df is None or next_week is None:
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
        'week': [next_week]
    })
    
    X_transformed = prep_pipeline.transform(input_data)
    prediction_proba = model.predict_proba(X_transformed)[0, 1]
    prediction = int(model.predict(X_transformed)[0])
    
    return PredictionResponse(
        customer_id=request.customer_id,
        product_id=request.product_id,
        week=next_week,
        prediction=prediction,
        prediction_proba=float(prediction_proba)
    )

@app.post("/predict/aggregate", response_model=AggregateResponse)
def predict_aggregate(request: AggregateRequest):
    if model is None or prep_pipeline is None or clientes_df is None or productos_df is None or next_week is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    producto_info = productos_df[productos_df['product_id'] == request.product_id]
    
    if producto_info.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    
    total_customers = len(clientes_df)
    expected_buyers = 0
    
    for _, cliente in clientes_df.iterrows():
        input_data = pd.DataFrame({
            'region_id': [cliente['region_id']],
            'customer_type': [cliente['customer_type']],
            'brand': [producto_info['brand'].values[0]],
            'category': [producto_info['category'].values[0]],
            'sub_category': [producto_info['sub_category'].values[0]],
            'segment': [producto_info['segment'].values[0]],
            'package': [producto_info['package'].values[0]],
            'size': [producto_info['size'].values[0]],
            'num_deliver_per_week': [cliente['num_deliver_per_week']],
            'num_visit_per_week': [cliente['num_visit_per_week']],
            'week': [next_week]
        })
        
        X_transformed = prep_pipeline.transform(input_data)
        prediction = model.predict(X_transformed)[0]
        expected_buyers += int(prediction)
    
    product_name = f"{producto_info['brand'].values[0]} - {producto_info['category'].values[0]} - {producto_info['sub_category'].values[0]}"
    
    return AggregateResponse(
        product_id=request.product_id,
        product_name=product_name,
        week=next_week,
        total_customers=total_customers,
        expected_buyers=expected_buyers,
        expected_percentage=float(expected_buyers / total_customers * 100)
    )

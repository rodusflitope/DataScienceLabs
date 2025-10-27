from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.pkl")
PREP_PIPELINE_PATH = os.getenv("PREP_PIPELINE_PATH", "/app/models/prep_pipeline.pkl")
CLIENTES_PATH = os.getenv("CLIENTES_PATH", "/app/models/clean_clientes.pkl")
PRODUCTOS_PATH = os.getenv("PRODUCTOS_PATH", "/app/models/clean_productos.pkl")
TRANSACCIONES_PATH = os.getenv("TRANSACCIONES_PATH", "/app/models/clean_transacciones.pkl")

model = None
prep_pipeline = None
clientes_df = None
productos_df = None
transacciones_df = None
next_week = None

@app.on_event("startup")
def load_artifacts():
    global model, prep_pipeline, clientes_df, productos_df, transacciones_df, next_week
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(PREP_PIPELINE_PATH, 'rb') as f:
            prep_pipeline = pickle.load(f)
        with open(CLIENTES_PATH, 'rb') as f:
            clientes_df = pickle.load(f)
        with open(PRODUCTOS_PATH, 'rb') as f:
            productos_df = pickle.load(f)
        with open(TRANSACCIONES_PATH, 'rb') as f:
            transacciones_df = pickle.load(f)
        
        transacciones_df['purchase_date'] = pd.to_datetime(transacciones_df['purchase_date'])
        fecha_inicio = transacciones_df['purchase_date'].min()
        transacciones_df['week'] = ((transacciones_df['purchase_date'] - fecha_inicio).dt.days // 7).astype(int)
        last_week = transacciones_df['week'].max()
        next_week = last_week + 1
        
        print(f"Loaded artifacts. Next week for recommendations: {next_week}")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

class RecommendationRequest(BaseModel):
    customer_id: int

class ProductRecommendation(BaseModel):
    product_id: int
    product_name: str
    brand: str
    category: str
    sub_category: str
    prediction_proba: float

class RecommendationResponse(BaseModel):
    customer_id: int
    week: int
    recommendations: list[ProductRecommendation]

@app.get("/")
def read_root():
    return {"message": "SodAI Drinks Recommendation System API"}

@app.get("/customers")
def get_customers():
    if clientes_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    customers = []
    for _, row in clientes_df.head(100).iterrows():
        customers.append({
            "customer_id": int(row['customer_id']),
            "gender": row.get('gender', 'N/A'),
            "age": int(row.get('age', 0)) if pd.notna(row.get('age', 0)) else 0,
            "region": row.get('region', 'N/A')
        })
    return {"customers": customers}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    if model is None or prep_pipeline is None or clientes_df is None or productos_df is None or next_week is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    customer_id = request.customer_id
    
    if customer_id not in clientes_df['customer_id'].values:
        raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found")
    
    cliente_info = clientes_df[clientes_df['customer_id'] == customer_id]
    customer_history = transacciones_df[transacciones_df['customer_id'] == customer_id]['product_id'].unique()
    
    predictions = []
    for _, product_row in productos_df.iterrows():
        product_id = int(product_row['product_id'])
        
        input_data = pd.DataFrame({
            'region_id': [cliente_info['region_id'].values[0]],
            'customer_type': [cliente_info['customer_type'].values[0]],
            'brand': [product_row['brand']],
            'category': [product_row['category']],
            'sub_category': [product_row['sub_category']],
            'segment': [product_row['segment']],
            'package': [product_row['package']],
            'size': [product_row['size']],
            'num_deliver_per_week': [cliente_info['num_deliver_per_week'].values[0]],
            'num_visit_per_week': [cliente_info['num_visit_per_week'].values[0]],
            'week': [next_week]
        })
        
        try:
            features = prep_pipeline.transform(input_data)
            proba = model.predict_proba(features)[0, 1]
            
            already_purchased = product_id in customer_history
            
            predictions.append({
                'product_id': product_id,
                'product_name': f"{product_row['brand']} - {product_row['category']} - {product_row['sub_category']}",
                'brand': product_row['brand'],
                'category': product_row['category'],
                'sub_category': product_row['sub_category'],
                'prediction_proba': float(proba),
                'already_purchased': already_purchased
            })
        except Exception as e:
            print(f"Error predicting for product {product_id}: {e}")
            continue
    
    if len(predictions) == 0:
        raise HTTPException(status_code=500, detail="Could not generate predictions for any product")
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('prediction_proba', ascending=False)
    
    top_5 = predictions_df.head(5)
    
    recommendations = []
    for _, row in top_5.iterrows():
        recommendations.append(ProductRecommendation(
            product_id=int(row['product_id']),
            product_name=row['product_name'],
            brand=row['brand'],
            category=row['category'],
            sub_category=row['sub_category'],
            prediction_proba=row['prediction_proba']
        ))
    
    return RecommendationResponse(
        customer_id=customer_id,
        week=next_week,
        recommendations=recommendations
    )

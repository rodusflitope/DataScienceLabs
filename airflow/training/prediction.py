import pandas as pd

def generate_predictions_for_next_week(clientes, productos, transacciones, 
                                       model, prep_pipeline=None, 
                                       last_week=None, threshold=0.5, top_k=None,
                                       frequency_df=None):
    
    if last_week is None:
        transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])
        fecha_inicio = transacciones['purchase_date'].min()
        transacciones['week'] = ((transacciones['purchase_date'] - fecha_inicio).dt.days // 7).astype(int)
        last_week = transacciones['week'].max()
    
    next_week = last_week + 1
    
    clientes_activos = transacciones['customer_id'].unique()
    productos_vendidos = transacciones['product_id'].unique()
    
    prediction_universe = pd.MultiIndex.from_product(
        [clientes_activos, productos_vendidos, [next_week]],
        names=['customer_id', 'product_id', 'week']
    ).to_frame(index=False)
    
    prediction_data = (
        prediction_universe
        .merge(clientes, on='customer_id', how='left')
        .merge(productos, on='product_id', how='left')
    )
    
    # Merge frequency feature if available
    if frequency_df is not None:
        prediction_data = prediction_data.merge(
            frequency_df, 
            on=['customer_id', 'product_id'], 
            how='left'
        )
        prediction_data['purchase_frequency'] = prediction_data['purchase_frequency'].fillna(0)
    else:
        # If not provided, try to calculate from transacciones (less ideal but fallback)
        freq = transacciones.groupby(['customer_id', 'product_id']).size().reset_index(name='purchase_frequency')
        prediction_data = prediction_data.merge(freq, on=['customer_id', 'product_id'], how='left')
        prediction_data['purchase_frequency'] = prediction_data['purchase_frequency'].fillna(0)
    
    feature_cols = [
        'region_id', 'customer_type', 'brand', 'category', 'sub_category',
        'segment', 'package', 'size', 'num_deliver_per_week', 
        'num_visit_per_week', 'week', 'purchase_frequency'
    ]
    
    X_pred = prediction_data[feature_cols]
    
    if prep_pipeline is not None:
        X_pred_transformed = prep_pipeline.transform(X_pred)
        predictions_proba = model.predict_proba(X_pred_transformed)[:, 1]
    else:
        predictions_proba = model.predict_proba(X_pred)[:, 1]
        
    # Apply threshold
    predictions = (predictions_proba >= threshold).astype(int)
    
    prediction_data['prediction'] = predictions
    prediction_data['prediction_proba'] = predictions_proba
    
    # Filter only positive predictions
    prediction_data = prediction_data[prediction_data['prediction'] == 1].copy()
    
    # If top_k is specified and we have more predictions than top_k, keep only top_k
    if top_k is not None and len(prediction_data) > top_k:
        prediction_data = prediction_data.sort_values('prediction_proba', ascending=False).head(top_k)
    
    return prediction_data


def load_model_and_predict(clientes, produtos, transacciones, model_path):
    import joblib
    
    full_pipeline = joblib.load(model_path)
    
    # Note: This function might need to be updated to load threshold if available
    # For now, using default 0.5
    predictions = generate_predictions_for_next_week(
        clientes, produtos, transacciones, 
        full_pipeline, prep_pipeline=None
    )
    
    return predictions

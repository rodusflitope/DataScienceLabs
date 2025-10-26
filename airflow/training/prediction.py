import pandas as pd

def generate_predictions_for_next_week(clientes, productos, transacciones, 
                                       model, prep_pipeline=None, 
                                       last_week=None):
    
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
    
    feature_cols = [
        'region_id', 'customer_type', 'brand', 'category', 'sub_category',
        'segment', 'package', 'size', 'num_deliver_per_week', 
        'num_visit_per_week', 'week'
    ]
    
    X_pred = prediction_data[feature_cols]
    
    if prep_pipeline is not None:
        X_pred_transformed = prep_pipeline.transform(X_pred)
        predictions_proba = model.predict_proba(X_pred_transformed)[:, 1]
        predictions = model.predict(X_pred_transformed)
    else:
        predictions_proba = model.predict_proba(X_pred)[:, 1]
        predictions = model.predict(X_pred)
    
    prediction_data['prediction'] = predictions
    prediction_data['prediction_proba'] = predictions_proba
    
    return prediction_data


def load_model_and_predict(clientes, produtos, transacciones, model_path):
    import joblib
    
    full_pipeline = joblib.load(model_path)
    
    predictions = generate_predictions_for_next_week(
        clientes, produtos, transacciones, 
        full_pipeline, prep_pipeline=None
    )
    
    return predictions

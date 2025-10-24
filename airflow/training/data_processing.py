import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from config import CLIENTES_FILENAME, PRODUCTOS_FILENAME, TRANSACCIONES_FILENAME

# Constantes globales
RANDOM_STATE = 10
RATIO_NEG_POS = 2


def load_data(
    data_path: str,
    clientes_filename: str = CLIENTES_FILENAME,
    productos_filename: str = PRODUCTOS_FILENAME,
    transacciones_filename: str = TRANSACCIONES_FILENAME,
):
    print("Cargando datos desde:", data_path)
    clientes = pd.read_parquet(os.path.join(data_path, clientes_filename))
    productos = pd.read_parquet(os.path.join(data_path, productos_filename))
    transacciones = pd.read_parquet(os.path.join(data_path, transacciones_filename))
    
    return clientes, productos, transacciones

# Preprocesamiento puro como eliminar los dupes (que era 1) y columnas irrelevantes (según lo que decidimos en la parte 1)
def clean_data(clientes, productos, transacciones):
    print("Limpiando datos")
    # Eliminar coordenadas geográficas (X, Y) porque tenemos region_id
    clientes_clean = clientes.drop(columns=['X', 'Y'], errors='ignore')
    
    # Eliminar duplicados
    clientes_clean = clientes_clean.drop_duplicates()
    productos_clean = productos.drop_duplicates()
    transacciones_clean = transacciones.drop_duplicates()
    
    # Eliminar columna 'items' porque tiene valores inconsistentes
    transacciones_clean = transacciones_clean.drop(columns=['items'], errors='ignore')
    
    return clientes_clean, productos_clean, transacciones_clean


class WeeklyDatasetBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, ratio_neg_pos: int = RATIO_NEG_POS, random_state: int = RANDOM_STATE):
        self.ratio_neg_pos = ratio_neg_pos
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X debe ser un dict con keys: 'transacciones', 'clientes', 'productos'
        """
        transacciones = X['transacciones'].copy()
        clientes = X['clientes']
        productos = X['productos']

        # Convertir purchase_date a datetime y calcular semana
        transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])
        fecha_inicio = transacciones['purchase_date'].min()
        transacciones['week'] = ((transacciones['purchase_date'] - fecha_inicio).dt.days // 7).astype(int)

        # Casos positivos (compras reales)
        positivos = (
            transacciones.groupby(['customer_id', 'product_id', 'week'])
            .size()
            .reset_index(name='count')
        )
        positivos['purchased'] = 1

        # Crear universo completo de combinaciones cliente-producto-semana
        semanas_totales = range(transacciones['week'].min(), transacciones['week'].max() + 1)
        clientes_activos = transacciones['customer_id'].unique()
        productos_vendidos = transacciones['product_id'].unique()
        
        universe = pd.MultiIndex.from_product(
            [clientes_activos, productos_vendidos, semanas_totales],
            names=['customer_id', 'product_id', 'week']
        ).to_frame(index=False)

        # Merge para identificar casos negativos
        dataset_completo = universe.merge(
            positivos, 
            on=['customer_id', 'product_id', 'week'], 
            how='left'
        )
        dataset_completo['purchased'] = dataset_completo['purchased'].fillna(0)

        # Negative sampling estratificado por cliente
        casos_positivos = dataset_completo[dataset_completo['purchased'] == 1]
        casos_negativos = dataset_completo[dataset_completo['purchased'] == 0]

        negativos_sample = []
        for cliente in clientes_activos:
            neg_cliente = casos_negativos[casos_negativos['customer_id'] == cliente]
            pos_cliente = casos_positivos[casos_positivos['customer_id'] == cliente]
            
            # Samplear negativos en proporción a positivos
            n_sample = min(len(pos_cliente) * self.ratio_neg_pos, len(neg_cliente))
            if n_sample > 0:
                neg_sample = neg_cliente.sample(n=int(n_sample), random_state=self.random_state)
                negativos_sample.append(neg_sample)

        # Combinar positivos y negativos
        if len(negativos_sample) > 0:
            casos_negativos_sample = pd.concat(negativos_sample, ignore_index=True)
            dataset_completo = pd.concat([casos_positivos, casos_negativos_sample], ignore_index=True)
        else:
            dataset_completo = casos_positivos.copy()

        # Merge con datos de clientes y productos
        dataset_final = (
            dataset_completo
            .merge(clientes, on='customer_id', how='left')
            .merge(productos, on='product_id', how='left')
        )

        # Eliminar columna 'count' si existe
        if 'count' in dataset_final.columns:
            dataset_final = dataset_final.drop(columns=['count'])
        
        return dataset_final


def add_week_cyclical(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    w = (Xc['week'] % 52).astype(float)
    Xc['week_sin'] = np.sin(2 * np.pi * w / 52.0)
    Xc['week_cos'] = np.cos(2 * np.pi * w / 52.0)
    return Xc.drop(columns=['week'])


def build_preprocessing_pipeline():
    # Columnas categóricas y numéricas
    cat_cols = [
        'region_id',
        'customer_type',
        'brand',
        'category',
        'sub_category',
        'segment',
        'package',
    ]
    
    num_base_cols = [
        'size',
        'num_deliver_per_week',
        'num_visit_per_week',
    ]
    
    # Después de add_week_cyclical, tendremos week_sin y week_cos
    num_cols = num_base_cols + ['week_sin', 'week_cos']
    
    # Transformador de tiempo
    add_time = FunctionTransformer(add_week_cyclical)
    
    # Procesamiento de columnas
    preprocess = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )
    
    # Pipeline completo
    prep_pipeline = Pipeline(steps=[
        ('add_time', add_time),
        ('preprocess', preprocess),
    ])
    
    return prep_pipeline


def prepare_datasets(dataset_final, prep_pipeline, fit=True):
    feature_cols = [
        'region_id', 'customer_type', 'brand', 'category', 'sub_category',
        'segment', 'package', 'size', 'num_deliver_per_week', 
        'num_visit_per_week', 'week'
    ]
    
    X = dataset_final[feature_cols]
    y = dataset_final['purchased']
    
    if fit:
        X_transformed = prep_pipeline.fit_transform(X)
    else:
        X_transformed = prep_pipeline.transform(X)
    
    return X_transformed, y

#Para evitar el data leakage en la división de los datos
def split_temporal_data(transacciones, train_ratio=0.70, val_ratio=0.15):
    transacciones_sorted = transacciones.sort_values('purchase_date')
    total = len(transacciones_sorted)
    
    idx_train = int(total * train_ratio)
    idx_val = int(total * (train_ratio + val_ratio))
    
    train_trans = transacciones_sorted.iloc[:idx_train]
    val_trans = transacciones_sorted.iloc[idx_train:idx_val]
    test_trans = transacciones_sorted.iloc[idx_val:]
    
    return train_trans, val_trans, test_trans
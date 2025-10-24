import pandas as pd
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weeks', type=int, default=1)
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent / 'data'
    transacciones = pd.read_parquet(base_path / 'transacciones.parquet')
    
    transacciones['purchase_date'] = pd.to_datetime(transacciones['purchase_date'])
    transacciones = transacciones.sort_values('purchase_date')
    
    max_date = transacciones['purchase_date'].max()
    cutoff_date = max_date - pd.Timedelta(weeks=args.weeks)
    
    transacciones_old = transacciones[transacciones['purchase_date'] < cutoff_date].copy()
    transacciones_new = transacciones[transacciones['purchase_date'] >= cutoff_date].copy()
    
    transacciones_old.to_parquet(base_path / 'transacciones_backup.parquet', index=False)
    transacciones_new.to_parquet(base_path / 'transacciones_new.parquet', index=False)

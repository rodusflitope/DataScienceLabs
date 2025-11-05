from __future__ import annotations
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
from typing import Tuple, Optional, Iterable

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


def _run_dir_from_kwargs(kwargs: Optional[dict], base_dir: str = "data") -> Path:
    """
    Devuelve la carpeta de ejecución como data/YYYYMMDD usando kwargs de Airflow
    (ds_nodash o logical_date). Si no hay kwargs, usa fecha actual.
    """
    run_stamp = None
    if kwargs:
        run_stamp = kwargs.get("ds_nodash")
        logical_date = kwargs.get("logical_date")
        if run_stamp is None and logical_date is not None:
            run_stamp = logical_date.strftime("%Y%m%d")
    if run_stamp is None:
        run_stamp = datetime.now().strftime("%Y%m%d")
    return Path(base_dir) / run_stamp


def _io_paths(run_dir: Path) -> dict:
    """Rutas usadas en el pipeline."""
    return {
        "raw": run_dir / "raw",
        "preprocessed": run_dir / "preprocessed",
        "splits": run_dir / "splits",
        "models": run_dir / "models",
    }


# =========================
# 1) create_folders
# =========================
def create_folders(**kwargs) -> str:
    """
    Crea carpeta de ejecución data/<FECHA>/ con subcarpetas:
    raw, preprocessed, splits, models.
    Retorna la ruta absoluta (string).
    """
    run_dir = _run_dir_from_kwargs(kwargs)
    paths = _io_paths(run_dir)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    print(f"[create_folders] Run dir: {run_dir.resolve()}")
    print("[create_folders] Subcarpetas: raw, preprocessed, splits, models creadas.")
    return str(run_dir.resolve())


# =========================
# 2) load_and_merge
# =========================
def load_and_merge(**kwargs) -> str:
    """
    Lee data_1.csv y, si existe, data_2.csv desde data/<FECHA>/raw,
    concatena (vertical) y guarda merged.csv en preprocessed.
    Retorna la ruta al merged.csv.
    """
    run_dir = _run_dir_from_kwargs(kwargs)
    paths = _io_paths(run_dir)

    f1 = paths["raw"] / "data_1.csv"
    f2 = paths["raw"] / "data_2.csv"

    if not f1.exists():
        raise FileNotFoundError(f"No se encontró {f1}. Descargue primero data_1.csv.")

    dfs = [pd.read_csv(f1)]
    if f2.exists():
        dfs.append(pd.read_csv(f2))
        print(f"[load_and_merge] data_2.csv detectado y agregado ({f2}).")
    else:
        print("[load_and_merge] No se encontró data_2.csv; se usará solo data_1.csv.")

    merged = pd.concat(dfs, axis=0, ignore_index=True)
    out_path = paths["preprocessed"] / "merged.csv"
    merged.to_csv(out_path, index=False)

    print(f"[load_and_merge] merged.csv guardado en: {out_path.resolve()} (n={len(merged)})")
    return str(out_path.resolve())


# =========================
# 3) split_data
# =========================
def split_data(test_size: float = 0.20, random_state: int = 42, **kwargs) -> Tuple[str, str]:
    """
    Lee preprocessed/merged.csv y realiza hold-out estratificado 80/20 sobre HiringDecision.
    Guarda train.csv y test.csv en splits/. Retorna (train_path, test_path).
    """
    run_dir = _run_dir_from_kwargs(kwargs)
    paths = _io_paths(run_dir)
    merged_path = paths["preprocessed"] / "merged.csv"

    if not merged_path.exists():
        raise FileNotFoundError("No se encontró preprocessed/merged.csv. Ejecute load_and_merge().")

    df = pd.read_csv(merged_path)
    target = "HiringDecision"
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna objetivo '{target}'.")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_path = paths["splits"] / "train.csv"
    test_path = paths["splits"] / "test.csv"

    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    print(f"[split_data] Train: {train_path.resolve()} (n={len(X_train)})")
    print(f"[split_data] Test : {test_path.resolve()} (n={len(X_test)})")
    return str(train_path.resolve()), str(test_path.resolve())


# =========================
# 4) train_model
# =========================
def train_model(estimator, **kwargs) -> str:
    """
    Recibe un estimador de clasificación de sklearn (p.ej., RandomForestClassifier()).
    - Lee splits/train.csv
    - Aplica Pipeline con ColumnTransformer:
        * Num: imputación media + StandardScaler
        * Cat: imputación moda + OneHotEncoder(handle_unknown='ignore')
    - Entrena el pipeline y guarda .joblib en models/ con nombre identificable.

    Retorna la ruta del modelo entrenado.
    """
    from sklearn.base import ClassifierMixin

    if not hasattr(estimator, "fit"):
        raise ValueError("El parámetro 'estimator' debe ser un estimador sklearn con método .fit().")

    run_dir = _run_dir_from_kwargs(kwargs)
    paths = _io_paths(run_dir)

    train_path = paths["splits"] / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("No se encontró splits/train.csv. Ejecute split_data().")

    df_train = pd.read_csv(train_path)
    target = "HiringDecision"
    if target not in df_train:
        raise ValueError(f"'{target}' no está en train.csv.")

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    # Categóricas explícitas (incluyendo PreviousCompanies, como acordamos)
    categorical = ["Gender", "EducationLevel", "RecruitmentStrategy", "PreviousCompanies"]
    # Numéricas = resto
    numeric = X_train.columns.difference(categorical)

    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, list(numeric)),
            ("cat", categorical_tf, categorical),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[
        ("preprocess", preproc),
        ("model", estimator),
    ])

    pipe.fit(X_train, y_train)

    # Nombre identificable del modelo
    model_name = estimator.__class__.__name__
    stamp = datetime.now().strftime("%H%M%S")
    model_path = paths["models"] / f"model_{model_name}_{stamp}.joblib"

    joblib.dump(pipe, model_path)
    print(f"[train_model] Modelo entrenado y guardado en: {model_path.resolve()}")
    return str(model_path.resolve())


# =========================
# 5) evaluate_models
# =========================
def evaluate_models(**kwargs) -> str:
    """
    Lee todos los .joblib de models/, evalúa en splits/test.csv con accuracy y
    selecciona el mejor. Imprime el nombre del mejor y su accuracy.
    Guarda el mejor como models/best_model.joblib y retorna su ruta.
    """
    run_dir = _run_dir_from_kwargs(kwargs)
    paths = _io_paths(run_dir)

    test_path = paths["splits"] / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("No se encontró splits/test.csv. Ejecute split_data().")

    df_test = pd.read_csv(test_path)
    target = "HiringDecision"
    if target not in df_test:
        raise ValueError(f"'{target}' no está en test.csv.")

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    model_files = sorted(paths["models"].glob("model_*.joblib"))
    if len(model_files) == 0:
        raise FileNotFoundError("No hay modelos en models/. Entrene con train_model().")

    # Evaluación de todos los modelos
    scores = []
    for mp in model_files:
        pipe = joblib.load(mp)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores.append((acc, mp))

    # Selección del mejor
    best_acc, best_path = max(scores, key=lambda t: t[0])

    # Guardar copia estandarizada
    best_model = joblib.load(best_path)
    best_out = paths["models"] / "best_model.joblib"
    joblib.dump(best_model, best_out)

    print(f"[evaluate_models] Mejor modelo: {best_path.name} | accuracy(test)={best_acc:.4f}")
    print(f"[evaluate_models] Guardado como: {best_out.resolve()}")
    return str(best_out.resolve())

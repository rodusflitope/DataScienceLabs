# %%
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd

# %%
import gradio as gr

# %%
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# %%
# ---------------------------
# Helpers
# ---------------------------
def _run_dir_from_kwargs(kwargs: dict, base_dir: str = "data") -> Path:
    """
    Devuelve la carpeta de ejecución como: data/YYYYMMDD
    Usa 'ds_nodash' (Airflow) o 'logical_date' si está disponible.
    """
    run_stamp = None
    if kwargs is not None:
        # Airflow 2.x suele dar ambas: ds_nodash y logical_date
        run_stamp = kwargs.get("ds_nodash", None) 
        logical_date = kwargs.get("logical_date", None)
        if run_stamp is None and logical_date is not None:
            run_stamp = logical_date.strftime("%Y%m%d") # logical_date es datetime; lo formateamos

    if run_stamp is None:
        # Fallback si se ejecuta fuera de Airflow
        run_stamp = datetime.now().strftime("%Y%m%d")

    return Path(base_dir) / run_stamp


# ---------------------------
# 1) create_folders
# ---------------------------
def create_folders(**kwargs) -> str:

    """
    Crea carpeta data/<FECHA>/ y subcarpetas: raw, splits, models.
    Retorna la ruta total de la carpeta de ejecución (tipo string)
    para poder usarla luego si quieres (por ejemplo XCom).
    """
    run_dir = _run_dir_from_kwargs(kwargs)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "splits").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    print(f"[create_folders] Carpeta de ejecución: {run_dir.resolve()}")
    print(f"[create_folders] Subcarpetas: raw/, splits/, models/ creadas.")
    # Devolver como string (útil usarlo con XCom push automático)
    return str(run_dir.resolve())


# ---------------------------
# 2) split_data
# ---------------------------
def split_data(test_size: float = 0.20, random_state: int = 42, **kwargs) -> tuple[str, str]:
    """
    Lee data_1.csv desde data/<FECHA>/raw/, realiza hold-out estratificado 80/20
    sobre HiringDecision y guarda train.csv y test.csv en data/<FECHA>/splits/.

    Retorna rutas absolutas (train_path, test_path).
    """
    run_dir = _run_dir_from_kwargs(kwargs)
    raw_csv = run_dir / "raw" / "data_1.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"No se encontró {raw_csv}. Asegúrese de colocar data_1.csv en {run_dir/'raw'}."
        )

    df = pd.read_csv(raw_csv)

    # Variable objetivo
    target = "HiringDecision"
    if target not in df.columns:
        raise ValueError(f"No se encuentra la columna objetivo '{target}' en {raw_csv}.")

    # Hold-out estratificado
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    splits_dir = run_dir / "splits"
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"

    # Guardar CSV
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    print(f"[split_data] Train guardado en: {train_path.resolve()}  (n={len(X_train)})")
    print(f"[split_data] Test  guardado en: {test_path.resolve()}  (n={len(X_test)})")
    return str(train_path.resolve()), str(test_path.resolve())


# ---------------------------
# 3) preprocess_and_train
# ---------------------------
def preprocess_and_train(random_state: int = 42, n_estimators: int = 300, **kwargs) -> str:
    """
    - Lee train.csv y test.csv desde data/<FECHA>/splits/
    - Arma un Pipeline con ColumnTransformer:
        * Numéricas: imputación media + StandardScaler
        * Categóricas: imputación moda + OneHotEncoder
    - Entrena RandomForest
    - Imprime Accuracy y F1 (clase positiva=1) en test
    - Guarda el pipeline entrenado en data/<FECHA>/models/model.joblib

    Retorna la ruta del modelo guardado.
    """
    run_dir = _run_dir_from_kwargs(kwargs)
    splits_dir = run_dir / "splits"
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("No se encuentran train.csv y/o test.csv. Ejecutar primero split_data().")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target = "HiringDecision"
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    # Definición de columnas (según enunciado)
    categorical = ["Gender", "EducationLevel", "RecruitmentStrategy", "PreviousCompanies"]

    # El resto (excluyendo target y las categóricas) serán numéricas
    numeric = X_train.columns.difference(categorical)

    # Preprocesadores
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, list(numeric)),
            ("cat", categorical_tf, categorical)
        ],
        remainder="drop"
    )

    # Modelo
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocess", preproc),
        ("model", clf)
    ])

    # Entrenamiento
    pipe.fit(X_train, y_train)

    # Evaluación en test
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, pos_label=1, average="binary")

    print(f"[preprocess_and_train] Accuracy (test): {acc:.4f}")
    print(f"[preprocess_and_train] F1 clase positiva=1 (test): {f1_pos:.4f}")

    # Guardado del modelo
    model_path = run_dir / "models" / "model.joblib"
    joblib.dump(pipe, model_path)
    print(f"[preprocess_and_train] Modelo guardado en: {model_path.resolve()}")

    return str(model_path.resolve())


# ---------------------------
# 4) Interfaz Gradio
# ---------------------------

def predict(file, model_path: str):
    """
    Carga el pipeline entrenado (joblib), lee un JSON subido por el usuario
    (archivo con un solo registro o una lista de registros) y devuelve la predicción.
    """
    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)  # gradio pasa la ruta temporal del archivo
    predictions = pipeline.predict(input_data)

    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]
    return {'Predicción': labels[0]}


def gradio_interface(model_path: str | None = None, base_dir: str = "data", **kwargs):
    """
    Interfaz Gradio que sube un JSON y usa predict(file, model_path).
    - Si model_path es None: busca el modelo del run actual (usando kwargs de Airflow),
      y si no hay kwargs (ejecución local), toma el más reciente encontrado en data/*/models/model.joblib.
    """

    def _run_dir_from_kwargs_local(_kwargs: dict | None, _base_dir: str = "data") -> Path | None:
        if _kwargs:
            ds_nodash = _kwargs.get("ds_nodash")
            logical_date = _kwargs.get("logical_date")
            if ds_nodash:
                return Path(_base_dir) / ds_nodash
            if logical_date:
                return Path(_base_dir) / logical_date.strftime("%Y%m%d")
        return None

    def _latest_model(p_base: str) -> Path | None:
        base = Path(p_base)
        candidates = sorted(
            [p for p in base.glob("*") if p.is_dir() and p.name.isdigit()],
            key=lambda p: p.name,
            reverse=True
        )
        for run_dir in candidates:
            mp = run_dir / "models" / "model.joblib"
            if mp.exists():
                return mp
        return None

    # Resolver model_path si no lo pasan explícito
    if model_path is None:
        run_dir = _run_dir_from_kwargs_local(kwargs, base_dir)
        if run_dir is not None:
            candidate = run_dir / "models" / "model.joblib"
            if candidate.exists():
                model_path = str(candidate.resolve())
        if model_path is None:
            latest = _latest_model(base_dir)
            if latest is None:
                raise FileNotFoundError("No se encontró ningún model.joblib en data/*/models/. "
                                        "Ejecutar preprocess_and_train() primero.")
            model_path = str(latest.resolve())

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Subir un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Subir un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
# %%

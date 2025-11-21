# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



"""Paso 1: utilidades de carga, limpieza y modelado."""

import os
import json
import gzip
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, GridSearchCV

from tqdm import tqdm  # se mantiene la dependencia, aunque no se use


# 1. Limpieza y carga de datos


def limpiar_dataset_csv(ruta_csv: str | Path) -> pd.DataFrame:
    """
    Lee un CSV y aplica las transformaciones básicas:
      - Renombrar la columna objetivo a 'default'
      - Eliminar 'ID' si existe
      - Normalizar EDUCATION (numérica, valores >4 se agrupan en 4)
      - Eliminar filas con valores faltantes
    """
    ruta_csv = Path(ruta_csv)
    df = pd.read_csv(ruta_csv)

    # Renombrar objetivo si viene con el nombre original del dataset
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})

    # Eliminar ID si está presente
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # EDUCATION a numérico y compactar categorías >4 en 4
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = pd.to_numeric(df["EDUCATION"], errors="coerce")
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    # Borrar filas con datos faltantes
    df = df.dropna().reset_index(drop=True)
    return df


def cargar_y_procesar(
    train_csv: str | Path = "files/input/train_data.csv",
    test_csv: str | Path = "files/input/test_data.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los CSV de train y test y aplica la limpieza básica.
    """
    df_train = limpiar_dataset_csv(train_csv)
    df_test = limpiar_dataset_csv(test_csv)
    return df_train, df_test


# 2. Separar en X/y y alinear columnas


def dividir_en_xy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = "default",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """
    Separa características (X) y variable objetivo (y) en train y test.

    - Si df_test no trae la columna objetivo, y_test será None.
    - Se garantiza que X_train y X_test tengan exactamente las mismas
      columnas y en el mismo orden.
    """
    if target_col not in df_train.columns:
        raise ValueError(
            f"La columna objetivo '{target_col}' no está en df_train. "
            f"Columnas disponibles: {list(df_train.columns)}"
        )

    # Train
    X_train = df_train.drop(columns=[target_col]).copy()
    y_train = pd.to_numeric(df_train[target_col], errors="coerce").astype("Int64")

    # Test (si trae la etiqueta)
    if target_col in df_test.columns:
        X_test = df_test.drop(columns=[target_col]).copy()
        y_test = pd.to_numeric(df_test[target_col], errors="coerce").astype("Int64")
    else:
        X_test = df_test.copy()
        y_test = None

    # Columnas comunes y alineación
    columnas_comunes = X_train.columns.intersection(X_test.columns)
    if len(columnas_comunes) == 0:
        raise ValueError(
            "X_train y X_test no comparten ninguna columna. "
            "Revisar el preprocesamiento."
        )

    faltantes_en_test = [c for c in X_train.columns if c not in X_test.columns]
    sobrantes_en_test = [c for c in X_test.columns if c not in X_train.columns]

    if faltantes_en_test or sobrantes_en_test:
        print("[Aviso] Ajustando columnas entre train y test:")
        if faltantes_en_test:
            print(" - Solo en train, faltan en test:", faltantes_en_test)
        if sobrantes_en_test:
            print(" - Solo en test, no están en train:", sobrantes_en_test)

    X_train = X_train[columnas_comunes].copy()
    X_test = X_test[columnas_comunes].copy()

    return X_train, y_train, X_test, y_test


# 3. Pipeline de preprocesamiento + clasificador

def crear_pipeline_rf(categorical_cols, numeric_cols) -> Pipeline:
    """
    Construye un Pipeline con:
      - Imputación + OneHotEncoder para variables categóricas
      - Imputación para numéricas
      - ColumnTransformer que une ambos bloques
      - Estandarización
      - PCA (todas las componentes)
      - Selección de k=23 características
      - SVC con kernel RBF
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numeric_cols),
        ]
    )

    clf = SVC(kernel="rbf", probability=True, random_state=42)

    modelo = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
            ("select", SelectKBest(score_func=f_classif, k=23)),
            ("clf", clf),
        ]
    )

    return modelo



# 4. Utilidades para modelo y métricas

MODEL_PATHS = [
    "files/models/model.pkl.gz",
]

OUTPUT_JSON = "files/output/metrics.json"

POS_LABEL = None        # puede sobrescribirse si se desea forzar la etiqueta positiva
STRICT_EXAMPLE_TYPO = False
TYPO_LABEL = "1"


def find_model_path() -> str:
    """
    Busca el primer modelo existente en MODEL_PATHS.
    """
    for ruta in MODEL_PATHS:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(
        "No se encontró ningún modelo en las rutas: " + ", ".join(MODEL_PATHS)
    )


def load_model(path: str):
    """
    Carga el modelo desde 'path'. Intenta primero con joblib y,
    si falla, recurre a pickle + gzip.
    """
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def infer_setting(y: pd.Series | np.ndarray) -> dict:
    """
    A partir de las clases presentes en y, decide si:
      - es binario (average='binary' con una etiqueta positiva)
      - es multiclase (average='weighted')
    """
    clases = np.unique(y)
    n_clases = len(clases)

    if n_clases <= 2:
        # Caso binario
        if POS_LABEL is not None:
            etiqueta_positiva = POS_LABEL
        else:
            # Heurística para caso 0/1
            try:
                valores_num = np.array(clases, dtype=float)
                if set(valores_num.tolist()) == {0.0, 1.0}:
                    etiqueta_positiva = 1
                else:
                    etiqueta_positiva = clases[-1]
            except Exception:
                etiqueta_positiva = clases[-1]

        return {"average": "binary", "pos_label": etiqueta_positiva}

    # Si hay más de dos clases, se toma ponderado
    return {"average": "weighted", "pos_label": None}


def compute_metrics(model, X, y, dataset_name: str) -> dict:
    """
    Calcula balanced accuracy, precisión, recall y f1 para un conjunto dado.
    """
    y_pred = model.predict(X)

    bacc = balanced_accuracy_score(y, y_pred)

    cfg = infer_setting(y)
    average = cfg["average"]
    pos_label = cfg["pos_label"]

    if average == "binary":
        prec = precision_score(
            y, y_pred, average=average, pos_label=pos_label, zero_division=0
        )
        rec = recall_score(
            y, y_pred, average=average, pos_label=pos_label, zero_division=0
        )
        f1 = f1_score(
            y, y_pred, average=average, pos_label=pos_label, zero_division=0
        )
    else:
        prec = precision_score(y, y_pred, average=average, zero_division=0)
        rec = recall_score(y, y_pred, average=average, zero_division=0)
        f1 = f1_score(y, y_pred, average=average, zero_division=0)

    return {
        "dataset": dataset_name,
        "precision": float(prec),
        "balanced_accuracy": float(bacc),
        "recall": float(rec),
        "f1_score": float(f1),
    }


def ensure_output_dir(path: str) -> None:
    """
    Crea la carpeta contenedora de 'path' si no existe.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)


def build_cm_entry(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    dataset_name: str,
) -> dict:
    """
    Construye un diccionario con la matriz de confusión en formato:
      {
        "type": "cm_matrix",
        "dataset": "...",
        "true_label1": {
            "predicted_label1": n11,
            ...
        },
        ...
      }
    """
    etiquetas = np.unique(
        np.concatenate([np.unique(y_true), np.unique(y_pred)])
    )
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)

    salida = {"type": "cm_matrix", "dataset": dataset_name}

    for i, etiqueta_real in enumerate(etiquetas):
        fila = {}
        for j, etiqueta_pred in enumerate(etiquetas):
            etiqueta_pred_str = str(etiqueta_pred)

            if STRICT_EXAMPLE_TYPO and etiqueta_pred_str == TYPO_LABEL:
                key = f"predicte_{etiqueta_pred_str}"
            else:
                key = f"predicted_{etiqueta_pred_str}"

            fila[key] = int(cm[i, j])

        salida[f"true_{str(etiqueta_real)}"] = fila

    return salida


def main_metricas():
    """
    Calcula métricas de train y test usando el modelo ya entrenado
    y las guarda en formato JSON Lines.
    Requiere que existan X_train, y_train, X_test, y_test en el ámbito global.
    """
    model_path = find_model_path()
    model = load_model(model_path)

    train_metrics = compute_metrics(model, X_train, y_train, "train")
    test_metrics = compute_metrics(model, X_test, y_test, "test")

    ensure_output_dir(OUTPUT_JSON)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics, ensure_ascii=False) + "\n")
        f.write(json.dumps(test_metrics, ensure_ascii=False) + "\n")

    print(f"Métricas guardadas en: {OUTPUT_JSON}")
    print("Ejemplo de métricas:")
    print(train_metrics)
    print(test_metrics)


def main_matriz_confusion():
    """
    Calcula las matrices de confusión para train y test y las añade
    al archivo JSON Lines de salida.
    """
    model_path = find_model_path()
    model = load_model(model_path)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_entry = build_cm_entry(y_train, y_pred_train, "train")
    test_entry = build_cm_entry(y_test, y_pred_test, "test")

    ensure_output_dir(OUTPUT_JSON)
    with open(OUTPUT_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(train_entry, ensure_ascii=False) + "\n")
        f.write(json.dumps(test_entry, ensure_ascii=False) + "\n")

    print(f"Matrices de confusión añadidas a: {OUTPUT_JSON}")
    print(train_entry)
    print(test_entry)


# 5. Búsqueda de hiperparámetros, guardado de modelo y ejecución

if __name__ == "__main__":
    # Rutas por defecto (ajustar si es necesario)
    ruta_train = (
        "files/input/train_data.csv/train_default_of_credit_card_clients.csv"
    )
    ruta_test = (
        "files/input/test_data.csv/test_default_of_credit_card_clients.csv"
    )

    # Carga y limpieza de datos
    df_train, df_test = cargar_y_procesar(ruta_train, ruta_test)
    X_train, y_train, X_test, y_test = dividir_en_xy(df_train, df_test)

    print("Datos de entrenamiento y prueba preparados.")

    # Definir columnas categóricas y numéricas
    cat_cols_explicit = (
        ["SEX", "EDUCATION", "MARRIAGE"]
        + [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
    )
    categorical_cols = [c for c in cat_cols_explicit if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    pipeline_base = crear_pipeline_rf(categorical_cols, numeric_cols)

    # Espacio de búsqueda de hiperparámetros
    param_dist = {
        "clf__C": np.logspace(-3, 3, 7),
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf"],
    }

    # Generar combinaciones aleatorias y adaptarlas a GridSearchCV
    n_iter = 50
    muestras = list(
        ParameterSampler(param_dist, n_iter=n_iter, random_state=42)
    )
    # Cada dict se convierte a un dict con listas, como requiere GridSearchCV
    param_grid = [
        {
            k: (v if isinstance(v, (list, tuple, np.ndarray)) else [v])
            for k, v in combinacion.items()
        }
        for combinacion in muestras
    ]

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline_base,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=2,
        error_score="raise",
    )

    print(f"Iniciando búsqueda de {n_iter} combinaciones con 10-fold CV...")
    grid.fit(X_train, y_train)

    print(
        "\nMejor balanced accuracy (CV): {:.4f}".format(grid.best_score_)
    )
    print("Mejores hiperparámetros encontrados:")
    print(grid.best_params_)

    # El objeto final será el GridSearchCV completo
    model = grid

    # Guardar el modelo (incluye best_estimator_)
    modelo_dir = Path("files/models")
    modelo_dir.mkdir(parents=True, exist_ok=True)
    modelo_path = modelo_dir / "model.pkl.gz"

    with gzip.open(modelo_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Modelo guardado en: {modelo_path}")

    # Verificación rápida
    print(str(type(model)))
    assert "GridSearchCV" in str(type(model))

    # Cálculo y guardado de métricas + matrices de confusión
    main_metricas()
    main_matriz_confusion()

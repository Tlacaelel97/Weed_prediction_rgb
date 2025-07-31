#!/usr/bin/env python3
"""
Script para entrenar un modelo de detección de maleza usando bandas RGB y ExG
en GPU con cuML.
"""
import argparse
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer

# sklearn para las métricas en CPU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# cuML y cuDF para GPU
import cudf
import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRF

import joblib  # funciona para modelos cuML también

def parse_args():
    p = argparse.ArgumentParser(description="Entrena RF RGB+ExG en GPU (cuML)")
    p.add_argument("--raster",       required=True, help="Ruta al TIFF multibanda RGB")
    p.add_argument("--weed_csv",     required=True, help="CSV con coords de maleza")
    p.add_argument("--notweed_csv",  required=True, help="CSV coords de no-maleza")
    p.add_argument("--output",       default="rf_rgb_model_cuml.pkl",
                   help="Dónde guardar el modelo cuML")
    p.add_argument("--test_size",    type=float, default=0.2,   help="Fracción test")
    p.add_argument("--n_estimators", type=int,   default=100,   help="n árboles RF")
    p.add_argument("--max_depth",    type=int,   default=None,  help="Profundidad máxima")
    p.add_argument("--min_samples_leaf", type=int, default=1,  help="Muestras mín. por hoja")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed")
    return p.parse_args()

def load_labels(weed_csv, notweed_csv):
    w  = pd.read_csv(weed_csv)
    nw = pd.read_csv(notweed_csv)
    for df in (w,nw):
        if 'x' in df.columns and 'y' in df.columns:
            df.rename(columns={'x':'lon','y':'lat'}, inplace=True)
    w['label'], nw['label'] = 1, 0
    df = pd.concat([w, nw], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def extract_features(df, raster_path):
    # Transformación de coordenadas lon/lat → CRS del ráster
    with rasterio.open(raster_path) as src:
        crs = src.crs
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    xs, ys = transformer.transform(df['lon'].values, df['lat'].values)
    coords = list(zip(xs, ys))

    # Muestreo de bandas
    with rasterio.open(raster_path) as src:
        samples = np.array(list(src.sample(coords)))  # (N,3)
    r, g, b = samples[:,0], samples[:,1], samples[:,2]
    exg      = 2*g - r - b
    X_np     = np.column_stack([r, g, b, exg]).astype(np.float32)
    y_np     = df['label'].values.astype(np.int32)
    return X_np, y_np

def main():
    args = parse_args()
    df   = load_labels(args.weed_csv, args.notweed_csv)
    X_np, y_np = extract_features(df, args.raster)

    # Split en CPU (numpy)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_np, y_np,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_np
    )

    # Pasar a cuDF/cuPy para entrenamiento en GPU
    gdf_tr = cudf.DataFrame({
        'R':   cp.asarray(X_tr[:,0]),
        'G':   cp.asarray(X_tr[:,1]),
        'B':   cp.asarray(X_tr[:,2]),
        'ExG': cp.asarray(X_tr[:,3]),
    })
    y_tr_gpu = cudf.Series(cp.asarray(y_tr))

    # Configurar y entrenar cuML RandomForest
    rf_gpu = cuRF(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed
    )
    print("Entrenando RF en GPU con cuML…")
    rf_gpu.fit(gdf_tr, y_tr_gpu)

    # Inferencia sobre test (GPU)
    gdf_te = cudf.DataFrame({
        'R':   cp.asarray(X_te[:,0]),
        'G':   cp.asarray(X_te[:,1]),
        'B':   cp.asarray(X_te[:,2]),
        'ExG': cp.asarray(X_te[:,3]),
    })
    preds_gpu = rf_gpu.predict(gdf_te)
    y_pred    = preds_gpu.to_numpy().astype(int)  # pasa de GPU a CPU

    # Métricas en CPU
    print("Evaluando en el conjunto de test…")
    print("Matriz de confusión:\n", confusion_matrix(y_te, y_pred))
    print("Classification Report:\n", classification_report(y_te, y_pred, digits=4))

    # Guardar modelo (pickleable con joblib)
    joblib.dump(rf_gpu, args.output)
    print(f"Modelo GPU guardado en {args.output}")

if __name__ == "__main__":
    main()

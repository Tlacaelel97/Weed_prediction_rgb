# train_rgb_model.py
"""
Script para entrenar un modelo de detección de maleza usando bandas RGB y ExG sobre CPU.
"""
import argparse
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def parse_args():
    p = argparse.ArgumentParser(description="Entrena RF sobre RGB+ExG")
    p.add_argument("--raster", required=True, help="Ruta al TIFF multibanda RGB")
    p.add_argument("--weed_csv", required=True, help="CSV con coords de maleza")
    p.add_argument("--notweed_csv", required=True, help="CSV coords de no-maleza")
    p.add_argument("--output", default="rf_rgb_model.pkl", help="Ruta para guardar el modelo")
    p.add_argument("--test_size", type=float, default=0.2, help="Fracción test")
    p.add_argument("--n_estimators", type=int, default=100, help="n árboles RF")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def load_labels(weed_csv, notweed_csv):
    w = pd.read_csv(weed_csv)
    nw = pd.read_csv(notweed_csv)
    for df in (w, nw):
        if 'x' in df.columns and 'y' in df.columns:
            df.rename(columns={'x':'lon','y':'lat'}, inplace=True)
    w['label'], nw['label'] = 1, 0
    df = pd.concat([w, nw], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def extract_features(df, raster_path):
    # Transformer y coords
    with rasterio.open(raster_path) as src:
        crs = src.crs
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    lons = df['lon'].values
    lats = df['lat'].values
    xs, ys = transformer.transform(lons, lats)
    coords = list(zip(xs, ys))
    # Muestreo masivo
    with rasterio.open(raster_path) as src:
        samples = np.array(list(src.sample(coords)))  # shape (N,3)
    # R, G, B y ExG
    r, g, b = samples[:,0], samples[:,1], samples[:,2]
    exg = 2*g - r - b
    X = np.column_stack([r, g, b, exg])
    y = df['label'].values
    return X, y


def main():
    args = parse_args()
    df = load_labels(args.weed_csv, args.notweed_csv)
    X, y = extract_features(df, args.raster)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        random_state=args.seed
    )
    print("Entrenando...")
    rf.fit(X_train, y_train)
    print("Evaluando en test set...")
    y_pred = rf.predict(X_test)
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(rf, args.output)
    print(f"Modelo guardado en {args.output}")

if __name__ == "__main__":
    main()

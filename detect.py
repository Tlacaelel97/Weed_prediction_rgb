# detect_rgb.py
"""
Script para generar un raster de predicción con el modelo entrenado (CPU).
"""
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import joblib

from sklearn.ensemble import RandomForestClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Predicción RGB+ExG CPU")
    p.add_argument("--raster", required=True, help="Ruta al TIFF multibanda RGB")
    p.add_argument("--model", required=True, help="Ruta al modelo RF .pkl")
    p.add_argument("--output", default="weed_pred.tif", help="Ruta GeoTIFF salida")
    p.add_argument("--threshold_exg", type=int, default=20, help="Umbral ExG para prefiltrar")
    return p.parse_args()


def main():
    args = parse_args()
    rf: RandomForestClassifier = joblib.load(args.model)
    with rasterio.open(args.raster) as src:
        profile = src.profile.copy()
        profile.update({'count':1,'dtype':'uint8','nodata':255,'compress':'lzw'})
        with rasterio.open(args.output, 'w', **profile) as dst:
            for _, window in src.block_windows(1):
                # Leer R/G/B en bloque y cálculo ExG
                r = src.read(1, window=window).astype(np.int16)
                g = src.read(2, window=window).astype(np.int16)
                b = src.read(3, window=window).astype(np.int16)
                exg = 2*g - r - b
                # Máscara verde
                mask = exg > args.threshold_exg
                pred = np.zeros(r.shape, dtype=np.uint8)
                if mask.any():
                    feats = np.stack([r[mask], g[mask], b[mask], exg[mask]], axis=1)
                    pred[mask] = rf.predict(feats).astype(np.uint8)
                dst.write(pred, 1, window=window)
    print(f"Predicción completada en {args.output}")

if __name__ == "__main__":
    main()

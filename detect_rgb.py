#!/usr/bin/env python3
"""
Script para generar un raster de predicción con el modelo entrenado (CPU o GPU).
Si el modelo es de cuML (RandomForestClassifier cuML), hará la inferencia en GPU;
si no, caerá en scikit-learn y usará CPU.
"""
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import joblib

# Intentamos importar cuDF/cuPy para GPU
try:
    import cudf, cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRF
    GPU_READY = True
except ImportError:
    GPU_READY = False

# scikit-learn para CPU
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

def parse_args():
    p = argparse.ArgumentParser(description="Predicción RGB+ExG (CPU/GPU)")
    p.add_argument("--raster",        required=True, help="Ruta al TIFF multibanda RGB")
    p.add_argument("--model",         required=True, help="Ruta al modelo RF (.pkl)")
    p.add_argument("--output",        default="weed_pred.tif", help="Ruta GeoTIFF salida")
    p.add_argument("--threshold_exg", type=int,   default=20,  help="Umbral ExG para prefiltrar")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Carga el modelo (puede ser sklearn o cuML)
    rf = joblib.load(args.model)
    use_gpu = GPU_READY and isinstance(rf, cuRF)
    engine  = "GPU (cuML)" if use_gpu else "CPU (scikit-learn)"
    print(f"Cargando modelo desde '{args.model}' → ejecutando en {engine}")

    # 2) Abre el ráster de entrada y prepara el perfil de salida
    with rasterio.open(args.raster) as src:
        profile = src.profile.copy()
        profile.update({
            'count':    1,
            'dtype':    'uint8',
            'nodata':   255,
            'compress': 'lzw'
        })

        with rasterio.open(args.output, 'w', **profile) as dst:
            # 3) Itera bloque a bloque
            for _, window in src.block_windows(1):
                # lee canales R/G/B y calcula ExG
                r = src.read(1, window=window).astype(np.int16)
                g = src.read(2, window=window).astype(np.int16)
                b = src.read(3, window=window).astype(np.int16)
                exg = 2*g - r - b

                # prefiltro por verde
                mask_prefilt = exg > args.threshold_exg
                pred = np.zeros(r.shape, dtype=np.uint8)

                if mask_prefilt.any():
                    # 4) Prepara características de los píxeles candidatos
                    coords = np.where(mask_prefilt)
                    feats_cpu = np.stack([
                        r[coords], g[coords], b[coords], exg[coords]
                    ], axis=1).astype(np.float32)

                    if use_gpu:
                        # -> GPU branch: convertimos a cuDF
                        gdf = cudf.DataFrame({
                            'R':   cp.asarray(feats_cpu[:,0]),
                            'G':   cp.asarray(feats_cpu[:,1]),
                            'B':   cp.asarray(feats_cpu[:,2]),
                            'ExG': cp.asarray(feats_cpu[:,3]),
                        })
                        # inferencia en GPU
                        try:
                            preds_gpu = rf.predict(gdf)           # cudf.Series
                        except NotFittedError:
                            raise RuntimeError("El modelo cuML no parece estar ajustado.")
                        preds = preds_gpu.to_numpy().astype(np.uint8)
                    else:
                        # -> CPU branch: scikit-learn
                        preds = rf.predict(feats_cpu).astype(np.uint8)

                    # 5) Escribe las predicciones de vuelta al array completo
                    pred[coords] = preds

                # 6) Escribe el bloque en el GeoTIFF de salida
                dst.write(pred, 1, window=window)

    print(f"Predicción completada: '{args.output}'")

if __name__ == "__main__":
    main()

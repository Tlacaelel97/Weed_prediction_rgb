#!/usr/bin/env python3
"""
invert_mask.py

Invierte una máscara binaria (0 fono, 255 maleza) para que:
 - Negro (0) = maleza
 - Blanco (255) = fondo
Acepta GeoTIFF o PNG de entrada.
"""
import argparse
import os
import numpy as np

def invert_png(input_path, output_path):
    import cv2
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"No pude leer '{input_path}'")
    # Asumimos 0 de fondo y 255 de maleza → invertimos
    inv = np.where(mask>0, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, inv)
    print(f"PNG invertido guardado en '{output_path}'")

def invert_tif(input_path, output_path):
    import rasterio
    with rasterio.open(input_path) as src:
        data = src.read(1).astype(np.uint8)
        inv  = np.where(data>0, 0, 255).astype(np.uint8)
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(inv, 1)
    print(f"GeoTIFF invertido guardado en '{output_path}'")

def parse_args():
    p = argparse.ArgumentParser(
        description="Invierte máscaras binarias (255→0, 0→255)"
    )
    p.add_argument(
        "-i","--input", required=True,
        help="Ruta al archivo de máscara (PNG o GeoTIFF)"
    )
    p.add_argument(
        "-o","--output", required=True,
        help="Ruta de salida (misma extensión que input)"
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".png", ".jpg", ".tif", ".tiff"):
        # elegimos método según extensión de salida
        if ext in (".png", ".jpg"):
            invert_png(args.input, args.output)
        else:
            invert_tif(args.input, args.output)
    else:
        raise ValueError(f"Extensión '{ext}' no soportada. Usa .png o .tif")

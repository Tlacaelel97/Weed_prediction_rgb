#!/usr/bin/env python3
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import argparse
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Recorta automáticamente la región activa de una máscara PNG y la aplica al TIFF (Pillow)"
    )
    p.add_argument("-t","--tif",  required=True, help="Ruta al TIFF original")
    p.add_argument("-m","--mask", required=True, help="Ruta a la máscara PNG")
    p.add_argument("-o","--out",  default="crop", help="Prefijo de salida (sin extensión)")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Abrir la máscara en “modo lazy”
    mask = Image.open(args.mask)
    gray = mask.convert("L")
    bbox = gray.getbbox()
    if bbox is None:
        print("La máscara está vacía. Nada que recortar.")
        return
    x1, y1, x2, y2 = bbox
    pad = 5
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(mask.width, x2 + pad), min(mask.height, y2 + pad)

    # 2) Recortar y guardar máscara
    crop_mask = mask.crop((x1, y1, x2, y2))
    out_mask  = f"{args.out}_mask.png"
    crop_mask.save(out_mask)

    # 3) Recortar y guardar TIFF original (solo esa ventana)
    tif = Image.open(args.tif)
    crop_tif = tif.crop((x1, y1, x2, y2))
    out_tif  = f"{args.out}_tif.png"
    crop_tif.save(out_tif)

    print("Región:", x1, y1, x2, y2)
    print("→", out_mask)
    print("→", out_tif)

if __name__=="__main__":
    main()

#!/usr/bin/env python3
import os
import argparse
from PIL import Image

# Desactivar límite de tamaño en Pillow
Image.MAX_IMAGE_PIXELS = None

def parse_args():
    p = argparse.ArgumentParser(
        description="Genera tiles emparejados de un TIFF y su máscara PNG"
    )
    p.add_argument("-t","--tif",      required=True, help="Ruta al TIFF original")
    p.add_argument("-m","--mask",     required=True, help="Ruta a la máscara PNG")
    p.add_argument("-o","--out_dir",  required=True, help="Directorio de salida")
    p.add_argument("--tile_size", type=int, default=512,
                   help="Tamaño de lado de cada tile (px)")
    p.add_argument("--overlap",  type=int, default=0,
                   help="Solapamiento entre tiles (px)")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Carga lazy del TIFF y la máscara
    tif  = Image.open(args.tif)
    mask = Image.open(args.mask)

    if tif.size != mask.size:
        raise SystemExit("ERROR: TIFF y máscara tienen resoluciones distintas")

    W, H = tif.size
    S     = args.tile_size
    O     = args.overlap
    out_img_dir  = os.path.join(args.out_dir, "tiles_img")
    out_mask_dir = os.path.join(args.out_dir, "tiles_mask")
    os.makedirs(out_img_dir,  exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    idx = 0
    for y in range(0, H, S - O):
        for x in range(0, W, S - O):
            x2 = min(x + S, W)
            y2 = min(y + S, H)
            box = (x, y, x2, y2)

            tile_img  = tif.crop(box)
            tile_mask = mask.crop(box)

            fname = f"tile_{idx:05d}_{x}_{y}.png"
            tile_img.save( os.path.join(out_img_dir,  fname) )
            tile_mask.save(os.path.join(out_mask_dir, fname) )

            idx += 1

    print(f"→ Generados {idx} tiles de {S}×{S}px con overlap={O}px")
    print(f"  Imágenes en: {out_img_dir}")
    print(f"  Máscaras en:  {out_mask_dir}")

if __name__ == "__main__":
    main()

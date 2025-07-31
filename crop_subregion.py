# crop_pair_subregion.py

import cv2
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser(
        description="Selecciona un sub-ROI y lo aplica simultáneamente a un TIFF y su máscara PNG"
    )
    p.add_argument("-t", "--tif",  required=True, help="Ruta al TIFF original")
    p.add_argument("-m", "--mask", required=True, help="Ruta a la máscara PNG")
    p.add_argument("-o", "--out",  default="subcrop", help="Prefijo de salida (sin extensión)")
    return p.parse_args()

def main():
    args = parse_args()

    # Carga ambas imágenes
    img_tif  = cv2.imread(args.tif,  cv2.IMREAD_COLOR)
    img_mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    if img_tif is None:
        print(f"ERROR: no pude cargar TIFF '{args.tif}'"); return
    if img_mask is None:
        print(f"ERROR: no pude cargar máscara '{args.mask}'"); return

    # Crea ventana y muestra la versión ligera (la máscara)
    win = "Selecciona sub-ROI (arrastrar + ENTER)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img_mask)
    cv2.waitKey(1)  # fuerza creación de la ventana

    # Selección interactiva
    x, y, w, h = cv2.selectROI(win, img_mask, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("No seleccionaste nada. Abortando."); return

    # Recorta y guarda la máscara
    crop_mask = img_mask[int(y):int(y+h), int(x):int(x+w)]
    out_mask  = f"{args.out}_mask.png"
    cv2.imwrite(out_mask, crop_mask)

    # Recorta y guarda el TIFF
    crop_tif  = img_tif[int(y):int(y+h), int(x):int(x+w)]
    out_tif   = f"{args.out}_tif.png"
    cv2.imwrite(out_tif, crop_tif)

    print("▶ Sub-recorte guardado:")
    print("   Máscara:", out_mask)
    print("   TIFF:   ", out_tif)
    print(f"  (x={x}, y={y}, w={w}, h={h})")

if __name__ == "__main__":
    main()

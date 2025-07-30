**README del Proyecto: Detección de Maleza con Dron y Random Forest**

---

## Descripción del Proyecto

Este proyecto implementa un sistema de detección de maleza en tiempo real a bordo de un dron DJI Agras T50. La cámara, montada con una inclinación de 45 °, captura vídeo en resolución UHD (3840 × 2160 px). A través de un pipeline geométrico—que recorta la imagen a una franja y un trapecio de visión frontal—y un clasificador **Random Forest** entrenado con datos anotados manualmente, se localiza la maleza y se calcula el instante preciso para disparar fertilizante.

---

## Flujo de Trabajo

1. **Extracción de fotogramas**

   * Se lee el archivo de vídeo y se extrae un fotograma cada *N* cuadros (configurable).
   * Cada imagen se guarda en la carpeta `video_frames/` como `frame_000001.png`, `frame_000006.png`, etc.

2. **Anotación de datos**

   * Con `annotation.py` se revisan manualmente los fotogramas:

     * **Clic izquierdo**: marca píxeles de maleza (`label=1`).
     * **Clic central** (rueda del mouse): marca píxeles de fondo (`label=0`).
     * **n**: siguiente imagen. **q**: finalizar.
   * Al concluir, se genera `annotations.csv`:

     ```csv
     filename,x,y,label
     frame_000001.png,123,456,1
     frame_000001.png,789,321,0
     ```

3. **Entrenamiento del modelo**

   * `train/train.py` realiza:

     1. Carga de `video_frames/` y `annotations.csv`.
     2. Extracción de características por píxel: canales BGR, índice ExG (`2·G–R–B`), HSV.
     3. División estratificada (80 % entrenamiento, 20 % test).
     4. **GridSearchCV** (5 folds) sobre:

        * `n_estimators`: {100, 200, 300}
        * `max_depth`: {None, 10, 20}
        * `min_samples_leaf`: {1, 2, 5}
     5. Evaluación con matriz de confusión y reporte de clasificación.
     6. Guardado en `rf_video_model.pkl`.

   * **Comando**:

     ```bash
     python train/train.py \
       --frames_dir video_frames \
       --annotations_csv annotations.csv \
       --output rf_video_model.pkl \
       --test_size 0.2 \
       --n_estimators 100 200 300 \
       --max_depth None 10 20 \
       --min_samples_leaf 1 2 5
     ```

---

## Inferencia y Visualización Optimizada

El script principal `main.py` ahora incorpora múltiples optimizaciones para acercarse a un procesamiento casi en tiempo real sin cambiar el modelo ni reentrenar:

* **Frame skipping**: procesa solo 1 de cada 10 cuadros.
* **Downscale**: reduce la ROI a 1/4 de su tamaño en cada dimensión (1/16 píxeles).
* **Patch sampling**: clasificación por bloques de 4×4 píxeles en la ROI reducida.
* **Morfología reforzada**: apertura y cierre con kernel de 7×7, apertura de 3×3 y filtrado de blobs <200 px.

### Ejecución de `main.py`

```bash
# Solo visualizar:
python main.py -i input_video.mp4 -m rf_video_model.pkl

# Visualizar y guardar resultado:
python main.py \
  -i input_video.mp4 \
  -m rf_video_model.pkl \
  -o output_with_detections.mp4
```

#### Argumentos

* `-i, --input_video`: ruta al vídeo de entrada (MP4).
* `-m, --model`: ruta al modelo entrenado (`.pkl`).
* `-o, --output_video`: ruta al MP4 de salida (opcional).

En modo visualización, cada ROI se procesa con skipping, downscale, patch sampling y morfología antes de superponer detecciones en rojo semitransparente y dibujar el trapecio.

---

## Simulación de Vuelo (`simulate.py`)

Además, `simulate.py` implementa el mismo pipeline optimizado en un bucle de simulación “en vivo”, incluyendo:

* Sincronización con FPS del vídeo.
* Programación de alertas basadas en velocidad, altura y latencia.
* Patch sampling y morfología reforzada.

```bash
python simulate.py \
  -i input_video.mp4 \
  -m rf_video_model.pkl \
  --speed_kmh 25 \
  --altitude 3 \
  --latency_ms 50
```

---

## Estructura de Directorios

```
├── video_frames/            # Fotogramas extraídos
├── annotations.csv          # Etiquetas generadas
├── train/                   # Entrenamiento
│   ├── train.py             # GridSearch y métricas
│   └── annotation.py        # Script interactivo de anotación
├── rf_video_model.pkl       # Modelo RF resultante
├── main.py                  # Inferencia optimizada + visualización
├── simulate.py              # Simulación de vuelo en vivo
├── optimize.py              # Profiling y optimizaciones multilínea
└── README.md                # Este documento
```

---

## Requisitos

* Python ≥ 3.8
* OpenCV
* pandas, numpy, scikit-learn
* joblib

---

## Resultado Actual

![Primeros resultados](image.png)

---

## Lógica de Optimización Interna

Para alcanzar un desempeño cercano al tiempo real, el pipeline aprovecha varias palancas:

* **Frame skipping**: solo se procesan 1 de cada 10 cuadros (`SKIP=10`), reduciendo la carga de procesamiento a un 10 %.
* **Downscale**: la región de interés (ROI) se reduce 4× en cada dimensión, de modo que el número total de píxeles cae a 1/16 antes de cualquier cálculo.
* **Patch sampling**: en la ROI reducida, se muestrea cada bloque de 4 × 4 píxeles en lugar de procesar píxel a píxel, ensamblando luego una máscara por defecto de esos bloques.
* **Filtrado morfológico en C++**: las operaciones de apertura y cierre (`cv2.morphologyEx`) y de extracción de contornos se ejecutan en el backend en C++, garantizando alta velocidad.
* **Hilos separados** (en `optimize.py`): la captura de vídeo (`cv2.VideoCapture`) y la inferencia se distribuyen entre dos hilos con una cola segura (`queue.Queue`), evitando bloqueos y aprovechando que las pruebas de OpenCV liberan el GIL.

Esta combinación permite reducir los tiempos por frame de varios segundos a decenas de milisegundos, sin modificar el modelo ni reanotar datos.

![Primeros resultados](image.png)

---

## Generación de Heatmap de Predicción

Para visualizar de forma más intuitiva tu máscara de predicción, se ha añadido el script `heatmap_overlay.py`:

```bash
python heatmap_overlay.py \
  --tif data/IN/campo_original.tif \
  --mask data/IN/prediccion.png \
  --output data/OUT/overlay_heatmap.png
```

Este script:

1. Carga el TIFF RGB original y la máscara PNG.
2. Normaliza la máscara a 0–255.
3. Aplica un colormap (`COLORMAP_JET`).
4. Superpone en semitransparencia (alpha=0.5).
5. Guarda el resultado sin mostrar ventanas.

```python
import cv2, numpy as np, argparse
p = argparse.ArgumentParser()
p.add_argument("--tif", required=True)
p.add_argument("--mask", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
orig      = cv2.imread(args.tif, cv2.IMREAD_COLOR)
mask      = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
mask_norm = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heatmap   = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
overlay   = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
cv2.imwrite(args.output, overlay)
```

---

## Recorte Interactivo Emparejado

Se ha creado `crop_pair.py` para seleccionar interactivamente una ROI sobre el TIFF y recortar el mismo rectángulo en la máscara PNG:

```bash
python crop_pair.py \
  --tif data/IN/campo_original.tif \
  --png data/IN/prediccion_mask.png \
  --out data/OUT/recorte1
```

1. Se abre la imagen TIFF y se dibuja ROI con el ratón.
2. Al confirmar (Enter), se aplica el mismo recorte a la máscara PNG.
3. Guarda `recorte1_tif_crop.png` y `recorte1_png_crop.png`, garantizando idéntica región.

---

## Recorte Interactivo de Sub‑regiones Emparejadas

Para casos en que ya dispongas de un par recortado (`*_tif.png` y `*_mask.png`), puedes perfilar detalles con `crop_pair_subregion.py` (alias `crop_subregion.py`). Este script permite dibujar un rectángulo sobre la máscara y aplicar idéntica selección al TIFF:

```bash
python crop_pair_subregion.py \
  -t data/OUT/recorte1_tif.png \
  -m data/OUT/recorte1_mask.png \
  -o data/OUT/recorte1_box4
```

1. Se muestra **solo** la máscara (`recorte1_mask.png`).
2. Clic-arrastre para marcar la sub‑región y pulsa **Enter** o **Espacio**.
3. Se generan:

   * `data/OUT/recorte1_box4_mask.png`
   * `data/OUT/recorte1_box4_tif.png`

De esta forma obtienes sub‑recortes exactamente coincidentes en ambas imágenes.

# Weed Prediciton

Este proyecto consiste unicamente en doferentes utilidades para el entrenamiento de modelos de Ia para la deteccion de maleza

Para entrenar el modelo simplementa hacemos

```bash
python train_rgb_model.py \
  --raster "data/IN/Lalo Campos_Abril.tif" \
  --weed_csv data/IN/cleaned_weed_2.csv \
  --notweed_csv data/IN/cleaned_no_weed_2.csv \
  --output rf_rgb_model_cuml.pkl \
  --test_size 0.2 \
  --n_estimators 200 \
  --max_depth 20 \
  --min_samples_leaf 2 \
  --seed 42
```

Parar realizar detecciones sobre un archivo tif:

```
python detect_rgb.py \
  --raster "data/IN/Lalo Campos_Abril.tif" \
  --model models/rf_rgb_model_cuml.pkl \
  --output data/OUT/weed_pred_gpu.tif \
  --threshold_exg 20
Cargando modelo desde 'models/rf_rgb_model_cuml.pkl' → ejecutando en GPU (cuML)
Predicción completada: 'data/OUT/weed_pred_gpu.tif'
```
Para convertir los archivos tif de la prediccion a shapefile podemos hacer lo siguiente

```
python raster_to_shapefile.py \
  -i data/OUT/weed_pred.tif \
  -o data/OUT/weed_polygons.shp
```

# Creacion de mascaras y tiles a partir de las detecciones

## Creacion de mascara

Basta con ejecutar:

```bash
python tif2HM.py
```

Es importante asegurarse de especificar en el script las rutas de entrada y de salida de 
los tif y png

## Extraccion de datos geoespaciales

gdalinfo ruta/a/tif

## Transformar mascara png a geotif

Para ser esto se debe de incluir la información extraida en el paso anterior, con el objetivo de añadr los datos geográficos. Se debe usar upper left seguido de lower right
```bash
gdal_translate \
  -of GTiff \
  -a_srs EPSG:4326 \
  -a_ullr -102.3442102 21.4011496 -102.3369427 21.3944741 \
  data/OUT/overlay_heatmap.png \
  overlay_heatmap_geo.tif
```

## Reproyectar a WebMercator (EPSG:3857) para encajar el esquema

```bash
gdalwarp \
  -t_srs EPSG:3857 \
  -r near \
  -co COMPRESS=LZW \
  overlay_heatmap_geo.tif \
  overlay_heatmap_3857.tif
```

## Teselear en {z}/{x}/{y}.png

```bash
gdal2tiles.py \
  -z 14-22 \
  -w none \
  overlay_heatmap_3857.tif \
  data/OUT/tiles_mask
```
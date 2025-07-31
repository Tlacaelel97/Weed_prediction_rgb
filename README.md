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
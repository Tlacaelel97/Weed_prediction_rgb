#!/usr/bin/env python3
"""
raster_to_shapefile.py

Convierte un GeoTIFF binario de maleza (valor=1) en un Shapefile de polígonos.
"""
import argparse
import rasterio
from rasterio.features import shapes
import fiona
from shapely.geometry import shape, mapping

def parse_args():
    p = argparse.ArgumentParser(
        description="Raster binario → Shapefile de polígonos (valor=1)"
    )
    p.add_argument(
        "-i", "--input_raster", required=True,
        help="Ruta al GeoTIFF de entrada (uint8, 0= fondo, 1= maleza)"
    )
    p.add_argument(
        "-o", "--output_shp", default="weed_polygons.shp",
        help="Ruta al Shapefile de salida"
    )
    p.add_argument(
        "--value", type=int, default=1,
        help="Valor raster a vectorizar (por defecto 1)"
    )
    return p.parse_args()

def raster_to_shapefile(raster_path, shp_path, value):
    # Abre el raster
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        mask = band == value

        # Genera (geom, val) para cada región conectada igual al valor
        shp_gen = (
            (shape(geom), val)
            for geom, val in shapes(band, mask=mask, transform=src.transform)
            if val == value
        )

        # Prepara esquema y CRS para fiona
        schema = {
            "geometry": "Polygon",
            "properties": {"value": "int"},
        }
        crs = src.crs

        # Escribe el Shapefile
        with fiona.open(
            shp_path, "w",
            driver="ESRI Shapefile",
            crs=crs,
            schema=schema
        ) as shp:
            for geom, val in shp_gen:
                shp.write({
                    "geometry": mapping(geom),
                    "properties": {"value": int(val)},
                })

if __name__ == "__main__":
    args = parse_args()
    raster_to_shapefile(args.input_raster, args.output_shp, args.value)
    print(f"✔ Shapefile creado en: {args.output_shp}")

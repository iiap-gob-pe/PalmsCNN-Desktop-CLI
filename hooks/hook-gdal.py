# hooks/hook-gdal.py
from PyInstaller.utils.hooks import collect_data_files

# Incluir datos de GDAL
datas = collect_data_files('osgeo')
hiddenimports = [
    'osgeo',
    'osgeo.gdal',
    'osgeo.ogr', 
    'osgeo.osr',
    'osgeo.gdal_array',
    'osgeo.gdalconst'
]
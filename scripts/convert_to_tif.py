import os
import sys
from osgeo import gdal
from PIL import Image

def convert_to_tif(input_path, output_path=None):
    """Convertir cualquier imagen a TIFF"""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_converted.tif"
    
    print(f"Convirtiendo {input_path} a {output_path}")
    
    try:
        # Intentar con GDAL primero
        src_ds = gdal.Open(input_path)
        if src_ds:
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.CreateCopy(output_path, src_ds, 0)
            dst_ds = None
            src_ds = None
        else:
            # Usar PIL como respaldo
            img = Image.open(input_path)
            img.save(output_path, format='TIFF')
            
        print(f"Conversión exitosa: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error en conversión: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_to_tif(sys.argv[1])
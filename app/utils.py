import os
import json
from datetime import datetime

def ensure_dir(directory):
    """Asegurar que existe el directorio"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_timestamp():
    """Obtener timestamp para nombres de archivo"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def validate_image_file(file_path):
    """Validar que el archivo es una imagen válida"""
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    ext = os.path.splitext(file_path)[1].lower()
    return ext in valid_extensions and os.path.exists(file_path)

def format_results(results):
    """Formatear resultados para mostrar en interfaz"""
    if not results:
        return "No se obtuvieron resultados"
    
    # Aquí puedes personalizar el formato según la salida de tus scripts
    return str(results)
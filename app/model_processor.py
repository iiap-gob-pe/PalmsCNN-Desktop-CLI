"""
Módulo para procesar tiles individuales con los modelos de IA
Permite importar la lógica de segmentacion.py e instancias.py
"""

import os
import sys
import numpy as np
from osgeo import gdal
import onnxruntime as ort

class TileModelProcessor:
    """
    Procesador de tiles que aplica los modelos de segmentación y conteo
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.setup_models()
        
    def setup_models(self):
        """Configurar modelos ONNX"""
        try:
            # Modelo de segmentación
            seg_model_path = self.config.get("models", {}).get("segmentacion", 
                            "models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx")
            self.seg_session = ort.InferenceSession(seg_model_path)
            self.seg_input_name = self.seg_session.get_inputs()[0].name
            
            # Modelo de instancias
            inst_model_path = self.config.get("models", {}).get("instancias",
                            "models/model_converted.onnx")
            self.inst_session = ort.InferenceSession(inst_model_path)
            self.inst_input_names = [inp.name for inp in self.inst_session.get_inputs()]
            
            print("Modelos ONNX cargados correctamente")
            
        except Exception as e:
            print(f"Error cargando modelos: {e}")
            raise
    
    def preprocess_tile(self, tile_data):
        """
        Preprocesar tile para el modelo de segmentación
        Similar a la lógica en segmentacion.py
        """
        # Convertir a float32 y normalizar
        if tile_data.dtype != np.float32:
            tile_data = tile_data.astype(np.float32)
        
        # Normalizar a [-1, 1]
        if tile_data.max() > 1:
            tile_data = tile_data / 127.5 - 1.0
        
        # Asegurar 3 bandas para RGB
        if len(tile_data.shape) == 2:
            tile_data = np.stack([tile_data] * 3, axis=-1)
        elif tile_data.shape[2] > 3:
            tile_data = tile_data[:, :, :3]
        
        return tile_data
    
    def segment_tile(self, tile_data):
        """
        Aplicar segmentación semántica a un tile
        """
        try:
            # Preprocesar tile
            processed_tile = self.preprocess_tile(tile_data)
            
            # Añadir dimensión de batch
            batch_tile = np.expand_dims(processed_tile, axis=0)
            
            # Ejecutar modelo
            outputs = self.seg_session.run(None, {self.seg_input_name: batch_tile})
            segmentation_mask = np.argmax(outputs[0][0], axis=-1).astype(np.uint8)
            
            return segmentation_mask
            
        except Exception as e:
            print(f"Error en segmentación de tile: {e}")
            return None
    
    def process_instances_tile(self, rgb_tile, segmentation_mask):
        """
        Aplicar conteo de instancias a un tile
        Similar a la lógica en instancias.py
        """
        try:
            # Preparar máscara para el modelo de instancias
            ss_mask = np.zeros_like(segmentation_mask, dtype=np.float32)
            
            # Mapear clases a códigos (como en instancias.py)
            class_to_ss = {-128: 1, -96: 2, -64: 3}  # Mauritia, Euterpe, Oenocarpus
            
            for ss_code, class_id in class_to_ss.items():
                ss_mask[segmentation_mask == class_id] = ss_code
            
            # Combinar RGB con máscara (4 canales)
            combined_input = np.concatenate([rgb_tile, ss_mask[..., np.newaxis]], axis=-1)
            
            # Preparar entrada para el modelo
            batch_input = np.expand_dims(combined_input.astype(np.float32), axis=0)
            ss_batch = np.expand_dims(segmentation_mask.astype(np.float32), axis=0)
            
            # Ejecutar modelo de instancias
            if len(self.inst_input_names) >= 2:
                inputs = {
                    self.inst_input_names[0]: batch_input,
                    self.inst_input_names[1]: ss_batch
                }
            else:
                inputs = {self.inst_input_names[0]: batch_input}
            
            outputs = self.inst_session.run(None, inputs)
            instances_result = outputs[0][0]
            
            return instances_result
            
        except Exception as e:
            print(f"Error en conteo de instancias de tile: {e}")
            return None
    
    def process_complete_tile(self, rgb_tile):
        """
        Procesamiento completo de un tile: segmentación + instancias
        """
        try:
            # 1. Segmentación
            segmentation_mask = self.segment_tile(rgb_tile)
            if segmentation_mask is None:
                return None
            
            # 2. Instancias
            instances_result = self.process_instances_tile(rgb_tile, segmentation_mask)
            
            # Para simplificar, retornamos la máscara de segmentación
            # En una implementación completa, combinaríamos ambos resultados
            return segmentation_mask
            
        except Exception as e:
            print(f"Error procesando tile completo: {e}")
            return None

# Instancia global
tile_model_processor = None

def initialize_tile_processor(config):
    """Inicializar el procesador de tiles global"""
    global tile_model_processor
    tile_model_processor = TileModelProcessor(config)
    return tile_model_processor
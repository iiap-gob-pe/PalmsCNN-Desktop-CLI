"""
OPTIMIZADOR DE MEMORIA MEJORADO - Cálculo preciso de uso de memoria
Implementación exacta de lo que el ingeniero recomendó
"""

import psutil
import os
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PreciseMemoryCalculator:
    """
    Calculador preciso de memoria para procesamiento por tiles
    Basado en las recomendaciones del ingeniero
    """
    
    def __init__(self, safety_margin: float = 0.7):
        self.safety_margin = safety_margin
        self.memory_history = []
        
    def get_available_memory_mb(self) -> float:
        """
        Retorna la memoria RAM disponible en MB.
        """
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        
        logger.info(f"Memoria disponible del sistema: {available_mb:.2f} MB")
        return available_mb
    
    def estimate_tile_memory_usage_mb(self, 
                                    tile_size: Tuple[int, int] = (512, 512), 
                                    channels: int = 3, 
                                    dtype: type = np.uint8,
                                    model_overhead_mb: float = 150) -> float:
        """
        Estima PRECISAMENTE el uso de memoria para procesar un solo tile.
        """
        bytes_per_pixel = np.dtype(dtype).itemsize
        
        tile_height, tile_width = tile_size
        tile_pixels = tile_height * tile_width
        tile_memory_bytes = tile_pixels * channels * bytes_per_pixel
        tile_memory_mb = tile_memory_bytes / (1024 * 1024)
        
        output_memory_bytes = tile_pixels * 1 * 4
        output_memory_mb = output_memory_bytes / (1024 * 1024)
        
        intermediate_memory_mb = tile_memory_mb * 2.5
        
        total_estimated_mb = (tile_memory_mb + 
                            output_memory_mb + 
                            intermediate_memory_mb + 
                            model_overhead_mb)
        
        logger.info(f"CÁLCULO DETALLADO DE MEMORIA PARA TILE {tile_size}:")
        logger.info(f"   • Tile entrada ({channels} canales, {dtype.__name__}): {tile_memory_mb:.2f} MB")
        logger.info(f"   • Tile salida (máscara float32): {output_memory_mb:.2f} MB")
        logger.info(f"   • Tensores intermedios: {intermediate_memory_mb:.2f} MB")
        logger.info(f"   • Overhead modelo: {model_overhead_mb:.2f} MB")
        logger.info(f"   • TOTAL ESTIMADO: {total_estimated_mb:.2f} MB")
        
        return total_estimated_mb
    
    def calculate_optimal_tile_size(self, 
                                  target_memory_mb: Optional[float] = None,
                                  initial_tile_size: Tuple[int, int] = (512, 512), 
                                  channels: int = 3, 
                                  dtype: type = np.uint8, 
                                  model_overhead_mb: float = 150,
                                  safety_margin_percent: float = 0.8) -> Optional[Tuple[int, int]]:
        """
        Calcula un tamaño de tile óptimo basado en la memoria disponible.
        """
        if target_memory_mb is None:
            available_mb = self.get_available_memory_mb()
            target_memory_mb = available_mb * safety_margin_percent
            logger.info(f"Memoria disponible: {available_mb:.2f} MB. "
                       f"Objetivo para tiles: {target_memory_mb:.2f} MB")
        
        current_tile_size = initial_tile_size
        iteration = 0
        max_iterations = 10
        
        logger.info("BUSCANDO TAMAÑO ÓPTIMO DE TILE...")
        
        while iteration < max_iterations:
            iteration += 1
            
            estimated_mb = self.estimate_tile_memory_usage_mb(
                current_tile_size, channels, dtype, model_overhead_mb
            )
            
            if estimated_mb <= target_memory_mb:
                logger.info(f"TILE {current_tile_size} VIABLE: "
                           f"{estimated_mb:.2f} MB <= {target_memory_mb:.2f} MB")
                return current_tile_size
            
            current_height, current_width = current_tile_size
            new_side = max(64, current_height // 2)
            
            if new_side == current_height:
                logger.warning(f"Tile mínimo alcanzado: {current_tile_size}. "
                              f"Estimado: {estimated_mb:.2f} MB, Objetivo: {target_memory_mb:.2f} MB")
                return current_tile_size
            
            current_tile_size = (new_side, new_side)
            logger.info(f"Reduciendo tile a {current_tile_size} "
                       f"(estimado: {estimated_mb:.2f} MB > objetivo: {target_memory_mb:.2f} MB)")
        
        logger.error(f"No se pudo encontrar tamaño de tile óptimo después de {max_iterations} iteraciones")
        return None
    
    def analyze_image_processing(self, image_path: str, 
                               model_overhead_mb: float = 150) -> Dict:
        """
        Análisis completo para procesamiento de imagen específica.
        """
        try:
            image_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    width, height = src.width, src.height
                    bands = src.count
            except ImportError:
                from osgeo import gdal
                dataset = gdal.Open(image_path)
                if dataset:
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize
                    bands = dataset.RasterCount
                    dataset = None
                else:
                    width, height, bands = 0, 0, 0
            
            available_mb = self.get_available_memory_mb()
            target_memory_mb = available_mb * self.safety_margin
            
            optimal_tile_size = self.calculate_optimal_tile_size(
                target_memory_mb=target_memory_mb,
                model_overhead_mb=model_overhead_mb
            )
            
            optimal_tile_memory = self.estimate_tile_memory_usage_mb(
                optimal_tile_size, model_overhead_mb=model_overhead_mb
            ) if optimal_tile_size else 0
            
            if optimal_tile_size and optimal_tile_size[0] >= 512:
                strategy = "procesamiento_directo"
                recommendation = "La imagen puede procesarse directamente"
            else:
                strategy = "procesamiento_por_tiles"
                recommendation = "Usar procesamiento por tiles para evitar sobrecarga de memoria"
            
            analysis = {
                'image_info': {
                    'path': image_path,
                    'size_mb': image_size_mb,
                    'dimensions': f"{width}x{height}",
                    'bands': bands,
                    'memory_mb': (width * height * bands * 4) / (1024 * 1024)
                },
                'system_info': {
                    'available_memory_mb': available_mb,
                    'target_memory_mb': target_memory_mb,
                    'safety_margin': self.safety_margin
                },
                'optimal_tile_size': optimal_tile_size,
                'tile_memory_estimate_mb': optimal_tile_memory,
                'strategy': strategy,
                'recommendation': recommendation,
                'model_overhead_used_mb': model_overhead_mb
            }
            
            logger.info("ANÁLISIS COMPLETO DE PROCESAMIENTO:")
            logger.info(f"   • Imagen: {os.path.basename(image_path)}")
            logger.info(f"   • Tamaño: {image_size_mb:.2f} MB")
            logger.info(f"   • Dimensiones: {width}x{height} ({bands} bandas)")
            logger.info(f"   • Memoria imagen completa: {analysis['image_info']['memory_mb']:.2f} MB")
            logger.info(f"   • Tile óptimo: {optimal_tile_size}")
            logger.info(f"   • Memoria por tile: {optimal_tile_memory:.2f} MB")
            logger.info(f"   • Estrategia: {strategy}")
            logger.info(f"   • Recomendación: {recommendation}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error en análisis de imagen: {e}")
            return {'error': str(e)}
    
    def get_processing_plan(self, image_path: str) -> Dict:
        """
        Genera un plan de procesamiento para la imagen.
        """
        analysis = self.analyze_image_processing(image_path)
        
        if 'error' in analysis:
            return {
                'image_info': {
                    'image_size_mb': 0,
                    'strategy': 'error',
                    'recommended_tile_size': 512
                },
                'steps': ['Error en análisis'],
                'warnings': [analysis['error']]
            }
        
        strategy = analysis['strategy']
        tile_size = analysis['optimal_tile_size'][0] if analysis['optimal_tile_size'] else 512
        
        if strategy == "procesamiento_directo":
            steps = [
                "PROCESAMIENTO_DIRECTO: Imagen cabe en memoria",
                "Cargar imagen completa en RAM",
                "Aplicar modelos de segmentación",
                "Generar resultados completos"
            ]
            warnings = []
        else:
            steps = [
                f"PROCESAMIENTO_POR_TILES: Tile size {tile_size}px",
                "Dividir imagen en tiles optimizados",
                "Procesar tiles por lotes controlados",
                "Reensamblar resultados finales"
            ]
            warnings = ["Imagen grande - Usando modo conservador de memoria"]
        
        return {
            'image_info': {
                'image_size_mb': analysis['image_info']['size_mb'],
                'strategy': strategy,
                'recommended_tile_size': tile_size
            },
            'steps': steps,
            'warnings': warnings
        }

# Instancia global para uso fácil
memory_calculator = PreciseMemoryCalculator()

# Funciones de conveniencia para uso directo
def get_available_memory_mb():
    return memory_calculator.get_available_memory_mb()

def estimate_tile_memory_usage_mb(tile_size=(512, 512), channels=3, dtype=np.uint8, model_overhead_mb=150):
    return memory_calculator.estimate_tile_memory_usage_mb(
        tile_size, channels, dtype, model_overhead_mb
    )

def calculate_optimal_tile_size(target_memory_mb=None, initial_tile_size=(512, 512), 
                              channels=3, dtype=np.uint8, model_overhead_mb=150, 
                              safety_margin_percent=0.8):
    return memory_calculator.calculate_optimal_tile_size(
        target_memory_mb, initial_tile_size, channels, dtype, 
        model_overhead_mb, safety_margin_percent
    )

def analyze_image_processing(image_path, model_overhead_mb=150):
    return memory_calculator.analyze_image_processing(image_path, model_overhead_mb)

def get_processing_plan(image_path):
    return memory_calculator.get_processing_plan(image_path)

# Alias para mantener compatibilidad
memory_optimizer = memory_calculator
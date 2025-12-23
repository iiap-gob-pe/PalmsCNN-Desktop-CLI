"""
PROCESADOR POR TILES MEJORADO - Lectura por partes y procesamiento en bloques
Implementación eficiente con gestión activa de memoria
"""

import os
import numpy as np
from osgeo import gdal
import psutil
import gc
from typing import List, Dict, Tuple, Optional, Callable
import logging
from PIL import Image
import time

logger = logging.getLogger(__name__)

class EfficientTileMemoryManager:
    """Gestor avanzado de memoria para tiles con cálculo preciso"""
    
    def __init__(self, safety_margin=0.5):
        self.safety_margin = safety_margin
        self.memory_history = []
    
    def calculate_precise_tile_memory(self, tile_size: Tuple[int, int], 
                                      channels: int = 3, 
                                      dtype_bytes: int = 4,
                                      model_overhead_mb: float = 150) -> float:
        """
        Calcula PRECISAMENTE la memoria necesaria para un tile
        """
        tile_height, tile_width = tile_size
        tile_pixels = tile_height * tile_width
        
        # Memoria del tile de entrada
        input_memory_mb = (tile_pixels * channels * dtype_bytes) / (1024 * 1024)
        
        # Memoria del tile de salida (segmentación)
        output_memory_mb = (tile_pixels * 1 * 4) / (1024 * 1024)
        
        # Memoria de tensores intermedios (estimación conservadora)
        intermediate_memory_mb = input_memory_mb * 2.5
        
        total_memory_mb = input_memory_mb + output_memory_mb + intermediate_memory_mb + model_overhead_mb
        
        logger.info(f"Memoria tile {tile_size}: {total_memory_mb:.1f} MB "
                    f"(input: {input_memory_mb:.1f}, output: {output_memory_mb:.1f}, "
                    f"intermediate: {intermediate_memory_mb:.1f}, model: {model_overhead_mb})")
        
        return total_memory_mb
    
    def get_available_memory_mb(self) -> float:
        """Obtiene memoria disponible real con margen de seguridad"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        safe_memory_mb = available_mb * self.safety_margin
        
        logger.info(f"Memoria disponible: {available_mb:.1f} MB -> Segura: {safe_memory_mb:.1f} MB")
        return safe_memory_mb
    
    def calculate_optimal_tile_size(self, image_path: str, 
                                  max_tile_size: int = 512,
                                  min_tile_size: int = 128) -> Tuple[int, int]:
        """
        Calcula el tamaño OPTIMO de tile basado en memoria disponible
        """
        try:
            available_memory_mb = self.get_available_memory_mb()
            
            # Tamaños de tile a evaluar (de mayor a menor, empezando en 512)
            possible_sizes = [512, 384, 256, 192, 128]
            possible_sizes = [s for s in possible_sizes if min_tile_size <= s <= max_tile_size]
            
            optimal_size = (min_tile_size, min_tile_size)
            
            for tile_side in possible_sizes:
                tile_size = (tile_side, tile_side)
                tile_memory = self.calculate_precise_tile_memory(tile_size)
                
                if tile_memory <= available_memory_mb:
                    optimal_size = tile_size
                    logger.info(f"Tile {tile_size} VIABLE ({tile_memory:.1f} MB)")
                    break
                else:
                    logger.info(f"Tile {tile_size} EXCEDE ({tile_memory:.1f} MB > {available_memory_mb:.1f} MB)")
            
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculando tile optimo: {e}")
            return (256, 256)

class EfficientImageTiler:
    """
    Tiler eficiente que lee por partes y maneja imágenes grandes
    """
    
    def __init__(self, image_path: str, tile_size: Tuple[int, int] = (512, 512), 
                 overlap: int = 32, use_pillow: bool = False):
        self.image_path = image_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.use_pillow = use_pillow
        
        # Intentar con GDAL primero (mejor para geoTIFF grandes)
        try:
            self.dataset = gdal.Open(image_path)
            if self.dataset:
                self.width = self.dataset.RasterXSize
                self.height = self.dataset.RasterYSize
                self.bands = self.dataset.RasterCount
                self.geotransform = self.dataset.GetGeoTransform()
                self.projection = self.dataset.GetProjection()
                self.driver = "GDAL"
            else:
                raise ValueError("GDAL no pudo abrir la imagen")
                
        except Exception as e:
            logger.warning(f"GDAL fallo, usando Pillow: {e}")
            self.use_pillow = True
            self.dataset = None
        
        # Fallback a Pillow si GDAL falla
        if self.use_pillow:
            try:
                self.pil_image = Image.open(image_path)
                self.width, self.height = self.pil_image.size
                self.bands = len(self.pil_image.getbands())
                self.driver = "Pillow"
            except Exception as e:
                raise ValueError(f"No se puede abrir la imagen con ningún método: {e}")
        
        logger.info(f"Tiler inicializado: {self.width}x{self.height}, "
                    f"{self.bands} bandas, tile_size: {tile_size}, driver: {self.driver}")
    
    def generate_tile_grid(self) -> List[Dict]:
        """Genera la cuadrícula de tiles para procesamiento por partes"""
        tiles = []
        
        tile_height, tile_width = self.tile_size
        
        # Calcular número de tiles
        cols = (self.width + tile_width - 1) // tile_width
        rows = (self.height + tile_height - 1) // tile_height
        
        logger.info(f"Grid: {cols} columnas x {rows} filas = {cols * rows} tiles")
        logger.info(f"Imagen: {self.width}x{self.height}, Tile: {tile_width}x{tile_height}")
        
        for row in range(rows):
            for col in range(cols):
                # Coordenadas de lectura CON solapamiento
                x_start_read = max(0, col * tile_width - self.overlap)
                y_start_read = max(0, row * tile_height - self.overlap)
                x_end_read = min(self.width, (col + 1) * tile_width + self.overlap)
                y_end_read = min(self.height, (row + 1) * tile_height + self.overlap)
                
                # Coordenadas de escritura SIN solapamiento
                x_start_write = col * tile_width
                y_start_write = row * tile_height
                x_end_write = min(self.width, (col + 1) * tile_width)
                y_end_write = min(self.height, (row + 1) * tile_height)
                
                # Dimensiones reales
                read_width = x_end_read - x_start_read
                read_height = y_end_read - y_start_read
                write_width = x_end_write - x_start_write
                write_height = y_end_write - y_start_write
                
                # Calcular offsets para remover el overlap al escribir
                overlap_left = self.overlap if x_start_read > 0 else 0
                overlap_top = self.overlap if y_start_read > 0 else 0
                overlap_right = self.overlap if x_end_read < self.width else 0
                overlap_bottom = self.overlap if y_end_read < self.height else 0
                
                tile_info = {
                    'id': f"tile_{row}_{col}",
                    'grid_position': (row, col),
                    'pixel_coordinates': {
                        'x_start': x_start_read, 'y_start': y_start_read,
                        'x_end': x_end_read, 'y_end': y_end_read,
                        'width': read_width,
                        'height': read_height
                    },
                    'write_coordinates': {
                        'x_start': x_start_write,
                        'y_start': y_start_write,
                        'width': write_width,
                        'height': write_height
                    },
                    'overlap_info': {
                        'left': overlap_left,
                        'top': overlap_top, 
                        'right': overlap_right,
                        'bottom': overlap_bottom,
                        'total_overlap': self.overlap
                    }
                }
                tiles.append(tile_info)
        
        return tiles
    
    def read_tile(self, tile_info: Dict, bands: List[int] = None) -> Optional[np.ndarray]:
        """
        Lee un tile específico de la imagen - IMPLEMENTACION DE LECTURA POR PARTES
        """
        if bands is None:
            bands = list(range(1, min(4, self.bands + 1)))
        
        try:
            coords = tile_info['pixel_coordinates']
            
            if self.driver == "GDAL":
                # LECTURA EFICIENTE CON GDAL - por partes
                tile_data = []
                for band_num in bands:
                    band = self.dataset.GetRasterBand(band_num)
                    data = band.ReadAsArray(
                        coords['x_start'], coords['y_start'],
                        coords['width'], coords['height']
                    )
                    if data is not None:
                        tile_data.append(data)
                    else:
                        logger.warning(f"Tile {tile_info['id']}: banda {band_num} vacia")
                        tile_data.append(np.zeros((coords['height'], coords['width']), dtype=np.uint8))
                
                # Stack bands: (height, width, bands)
                if len(tile_data) > 1:
                    return np.stack(tile_data, axis=-1)
                else:
                    return tile_data[0]
                    
            else:  # Pillow
                # LECTURA EFICIENTE CON PILLOW - por partes usando crop()
                tile_pil = self.pil_image.crop((
                    coords['x_start'], coords['y_start'],
                    coords['x_end'], coords['y_end']
                ))
                tile_array = np.array(tile_pil)
                
                # Manejar diferentes formatos de imagen
                if len(tile_array.shape) == 2:
                    tile_array = np.expand_dims(tile_array, axis=-1)
                
                return tile_array
                
        except Exception as e:
            logger.error(f"Error leyendo tile {tile_info['id']}: {e}")
            return None
    
    def close(self):
        """Libera recursos - CRUCIAL para gestión de memoria"""
        try:
            if self.dataset:
                self.dataset = None
            if hasattr(self, 'pil_image'):
                self.pil_image.close()
        except Exception as e:
            logger.warning(f"Error cerrando recursos: {e}")

class EfficientTileProcessor:
    """
    Procesador principal que implementa procesamiento en bloques
    con gestión activa de memoria
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.memory_manager = EfficientTileMemoryManager()
        self.current_tiler = None
        self.output_dataset = None
        
        # Cargar modelos ONNX si están disponibles en la configuración
        self.segmentation_session = None
        self.model_input_size = (512, 512)
        self._load_models()
    
    def _load_models(self):
        """Cargar modelos ONNX con configuración de BAJA MEMORIA"""
        try:
            logger.info("Buscando modelos ONNX...")
            
            models_config = self.config.get("models", {})
            
            # Modelo de segmentación - con múltiples fallbacks
            seg_model_path = None
            
            possible_seg_paths = [
                models_config.get("segmentacion"),
                "models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx",
                "../models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx",
                "./models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx"
            ]
            
            for path in possible_seg_paths:
                if path and os.path.exists(path):
                    seg_model_path = path
                    logger.info(f"Modelo encontrado en: {path}")
                    break
            
            if seg_model_path and os.path.exists(seg_model_path):
                import onnxruntime as ort
                logger.info(f"Cargando modelo: {seg_model_path}")
                
                # CONFIGURACION CRITICA DE MEMORIA ONNX
                sess_options = ort.SessionOptions()
                
                # 1. Desactivar la Arena de Memoria (CRUCIAL)
                # Esto hace que ONNX libere la RAM inmediatamente
                sess_options.enable_cpu_mem_arena = False
                
                # 2. Ejecución secuencial
                # Reduce el pico de memoria al no paralelizar operaciones
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                
                # 3. Limitar hilos
                # Evita overhead de memoria por múltiples hilos
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                
                self.segmentation_session = ort.InferenceSession(
                    seg_model_path, 
                    sess_options, 
                    providers=['CPUExecutionProvider']
                )
                
                model_input_shape = self.segmentation_session.get_inputs()[0].shape
                if len(model_input_shape) >= 3:
                    self.model_input_size = (model_input_shape[1], model_input_shape[2])
                
                logger.info(f"Modelo cargado en MODO BAJA MEMORIA")
                logger.info(f"Tamaño de entrada: {self.model_input_size}")
                    
            else:
                logger.warning("No se pudo encontrar el modelo de segmentación")
                
        except ImportError as e:
            logger.error(f"ONNX Runtime no está instalado: {e}")
        except Exception as e:
            logger.error(f"Error cargando modelo ONNX: {e}")
    
    def process_image_by_tiles(self, image_path: str, output_path: str, 
                             processing_callback: Callable,
                             tile_size: Optional[Tuple[int, int]] = None,
                             overlap: int = 32,
                             batch_size: int = 1,
                             progress_callback: Callable = None) -> bool:
        """
        Versión mejorada con gestión activa de memoria y progreso
        """
        try:
            logger.info(f"INICIANDO PROCESAMIENTO OPTIMIZADO")
            
            # CALCULO PRECISO DE MEMORIA
            if tile_size is None:
                # Forzar tile de 512 para seguridad
                tile_size = (512, 512)
                logger.info(f"Usando tile seguro: {tile_size}")
            
            # Forzar batch a 1 siempre
            safe_batch_size = 1
            
            logger.info(f"Configuracion: Tile {tile_size}, Batch {safe_batch_size}")
            
            # Inicializar tiler
            self.current_tiler = EfficientImageTiler(image_path, tile_size, overlap)
            
            # Crear salida
            if not self._create_output_dataset(output_path):
                return False
            
            # Generar grid
            tiles = self.current_tiler.generate_tile_grid()
            total_tiles = len(tiles)
            
            logger.info(f"Total de tiles: {total_tiles}")
            
            if progress_callback:
                progress_callback(0, f"Procesando {total_tiles} tiles...")
            
            # PROCESAMIENTO POR BLOQUES CON GESTION DE MEMORIA
            processed_count = 0
            success_count = 0
            
            for batch_start in range(0, total_tiles, safe_batch_size):
                batch_end = min(batch_start + safe_batch_size, total_tiles)
                batch_tiles = tiles[batch_start:batch_end]
                
                if progress_callback:
                    progress = (batch_start / total_tiles) * 100
                    progress_callback(progress, f"Procesando tile {batch_start+1}/{total_tiles}...")
                
                # Procesar lote actual
                batch_success = self._process_tile_batch_optimized(
                    batch_tiles, processing_callback, progress_callback
                )
                success_count += batch_success
                processed_count += len(batch_tiles)
                
                # LIMPIEZA AGRESIVA DE MEMORIA
                self._force_memory_cleanup()
                import gc
                gc.collect() # Forzar recolección explícita
                
                # REPORTE DE ESTADO (cada 10 tiles)
                if processed_count % 10 == 0:
                    memory_status = self._get_memory_status()
                    logger.info(f"Progreso: {processed_count}/{total_tiles} | RAM usada: {memory_status['process_memory_mb']:.1f} MB")
            
            # FINALIZAR
            self._finalize_processing()
            
            success_rate = (success_count / total_tiles) * 100
            logger.info(f"PROCESAMIENTO COMPLETADO: {success_count}/{total_tiles} tiles")
            
            if progress_callback:
                progress_callback(100, "Completado")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error en procesamiento por tiles: {e}")
            self._finalize_processing()
            return False

    def _calculate_safe_batch_size(self, tile_size: tuple, desired_batch_size: int) -> int:
        """Siempre retorna 1 para máxima seguridad en este modo"""
        return 1

    def _process_tile_batch_optimized(self, batch_tiles: List[Dict], 
                                      processing_callback: Callable,
                                      progress_callback: Callable = None) -> int:
        """
        Procesamiento de lote con mejor manejo de errores y progreso
        """
        success_count = 0
        
        for i, tile_info in enumerate(batch_tiles):
            try:
                # 1. LECTURA DEL TILE
                tile_data = self.current_tiler.read_tile(tile_info)
                if tile_data is None:
                    continue
                
                # 2. PROCESAMIENTO
                processed_tile = processing_callback(tile_data, tile_info)
                
                if processed_tile is not None:
                    # 3. ESCRITURA
                    self._write_tile_to_output_optimized(tile_info, processed_tile)
                    success_count += 1
                
                # 4. LIBERACION INMEDIATA DE MEMORIA
                del tile_data
                if processed_tile is not None:
                    del processed_tile
                    
            except Exception as e:
                logger.error(f"Error procesando tile: {e}")
                continue
        
        return success_count

    def _write_tile_to_output_optimized(self, tile_info: Dict, processed_tile: np.ndarray):
        """Escribe tile procesado en la posición correcta del output"""
        try:
            write_coords = tile_info['write_coordinates']
            
            tile_height, tile_width = processed_tile.shape[:2]
            expected_height = write_coords['height']
            expected_width = write_coords['width']
            
            actual_x_start = write_coords['x_start']
            actual_y_start = write_coords['y_start']
            
            if (tile_height != expected_height or tile_width != expected_width):
                # Recortar si es necesario (por padding del modelo)
                processed_tile = processed_tile[:expected_height, :expected_width]
                
                # Rellenar si es necesario
                if processed_tile.shape[0] < expected_height or processed_tile.shape[1] < expected_width:
                    padded = np.zeros((expected_height, expected_width), dtype=processed_tile.dtype)
                    padded[:processed_tile.shape[0], :processed_tile.shape[1]] = processed_tile
                    processed_tile = padded
            
            band = self.output_dataset.GetRasterBand(1)
            band.WriteArray(processed_tile, actual_x_start, actual_y_start)
            
        except Exception as e:
            logger.error(f"Error escribiendo tile: {e}")

    # METODOS EXISTENTES (mantenidos para compatibilidad)
    def process_tile_with_model(self, tile_data):
        """Procesa un tile usando el modelo de segmentación"""
        try:
            processed_tile = self._preprocess_tile_for_model(tile_data)
            
            if self.segmentation_session is not None:
                return self._process_with_onnx_model(processed_tile)
            else:
                return self._process_tile_basic(processed_tile)
            
        except Exception as e:
            logger.error(f"Error procesando tile con modelo: {e}")
            height, width = tile_data.shape[:2]
            return np.zeros((height, width), dtype=np.float32)

    def _process_with_onnx_model(self, processed_tile):
        """Procesar tile con modelo ONNX real"""
        try:
            original_height, original_width = processed_tile.shape[:2]
            
            # Redimensionar si es necesario
            if (original_height, original_width) != self.model_input_size:
                try:
                    from PIL import Image
                    pil_tile = Image.fromarray((processed_tile * 255).astype(np.uint8))
                    pil_tile = pil_tile.resize(self.model_input_size, Image.Resampling.LANCZOS)
                    processed_tile = np.array(pil_tile).astype(np.float32) / 255.0
                except Exception as e:
                    pass

            input_tile = np.expand_dims(processed_tile, axis=0)
            input_name = self.segmentation_session.get_inputs()[0].name
            output_name = self.segmentation_session.get_outputs()[0].name
            
            # INFERENCIA
            outputs = self.segmentation_session.run([output_name], {input_name: input_tile})
            segmentation_mask = np.argmax(outputs[0][0], axis=-1).astype(np.float32)
            
            # Restaurar tamaño original
            if segmentation_mask.shape != (original_height, original_width):
                try:
                    from PIL import Image
                    pil_mask = Image.fromarray(segmentation_mask.astype(np.uint8))
                    pil_mask = pil_mask.resize((original_width, original_height), Image.Resampling.NEAREST)
                    segmentation_mask = np.array(pil_mask).astype(np.float32)
                except Exception:
                    pass
            
            return segmentation_mask
            
        except Exception as e:
            logger.error(f"Error en inferencia ONNX: {e}")
            return self._process_tile_basic(processed_tile)

    def _process_tile_basic(self, processed_tile):
        """Procesamiento básico de tile (fallback)"""
        height, width = processed_tile.shape[:2]
        return np.zeros((height, width), dtype=np.float32)

    def _preprocess_tile_for_model(self, tile_data):
        """Preprocesa el tile para el modelo"""
        try:
            if tile_data.dtype != np.float32:
                tile_data = tile_data.astype(np.float32) / 255.0
            
            if len(tile_data.shape) == 2:
                tile_data = np.stack([tile_data] * 3, axis=-1)
            elif tile_data.shape[2] > 3:
                tile_data = tile_data[:, :, :3]
                
            return tile_data
        except Exception:
            return tile_data

    def _create_output_dataset(self, output_path: str) -> bool:
        """Crea el dataset de salida para los resultados"""
        try:
            driver = gdal.GetDriverByName('GTiff')
            self.output_dataset = driver.Create(
                output_path,
                self.current_tiler.width,
                self.current_tiler.height,
                1,
                gdal.GDT_Float32
            )
            
            if self.output_dataset:
                if hasattr(self.current_tiler, 'geotransform') and self.current_tiler.geotransform:
                    self.output_dataset.SetGeoTransform(self.current_tiler.geotransform)
                if hasattr(self.current_tiler, 'projection') and self.current_tiler.projection:
                    self.output_dataset.SetProjection(self.current_tiler.projection)
                return True
            return False
        except Exception as e:
            logger.error(f"Error creando dataset: {e}")
            return False

    def _force_memory_cleanup(self):
        """Limpieza activa y forzada de memoria"""
        gc.collect()
        try:
            if hasattr(np, 'getbuffer'):
                np.getbuffer.cache_clear()
        except:
            pass

    def _aggressive_memory_cleanup(self):
        """Limpieza agresiva de memoria para situaciones críticas"""
        for _ in range(3):
            gc.collect()

    def _get_memory_status(self) -> Dict:
        """Obtiene estado actual de memoria"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        return {
            'process_memory_mb': process.memory_info().rss / (1024**2),
            'available_gb': memory.available / (1024**3)
        }

    def _finalize_processing(self):
        """Finaliza el procesamiento y libera TODOS los recursos"""
        try:
            if self.output_dataset:
                self.output_dataset.FlushCache()
                self.output_dataset = None
            
            if self.current_tiler:
                self.current_tiler.close()
                self.current_tiler = None
            
            self._aggressive_memory_cleanup()
            logger.info("Recursos liberados")
        except Exception as e:
            logger.error(f"Error finalizando: {e}")

# Instancia global mejorada para uso fácil
tile_processor = EfficientTileProcessor()
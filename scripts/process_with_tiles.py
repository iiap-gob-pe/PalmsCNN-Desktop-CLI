import sys
import os
import argparse
import numpy as np
from osgeo import gdal, gdal_array
import onnxruntime as rt
import psutil
import time
import gc
import math
import cv2
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Habilitar excepciones de GDAL
gdal.UseExceptions()

# ============================================================================
# CONFIGURACIÓN DE LOGGING Y ARGUMENTOS
# ============================================================================

# Configurar logging para Windows (sin caracteres Unicode problemáticos)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('segmentacion_tiles_batch.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Segmentación por Tiles con Procesamiento por Batches para Máxima Velocidad'
)
parser.add_argument('image_path', help='Ruta a la imagen TIF')
parser.add_argument('--model', default='Redes/deeplab_keras_model_palms_iaa_all_0.003_W.onnx',
                    help='Ruta al modelo ONNX')
parser.add_argument('--output', default='output_tiles_batch',
                    help='Directorio de salida')
parser.add_argument('--tile_size', type=int, default=512,
                    help='Tamaño de los tiles (cuadrados)')
parser.add_argument('--overlap', type=int, default=64,
                    help='Solapamiento entre tiles para evitar bordes')
parser.add_argument('--scaling', choices=['none', 'normalize'], default='normalize',
                    help='Tipo de escalado de entrada')
parser.add_argument('--max_batch_size', type=int, default=8,
                    help='Tamaño máximo del batch para inferencia (4-8 recomendado)')
parser.add_argument('--min_batch_size', type=int, default=4,
                    help='Tamaño mínimo del batch para inferencia')
parser.add_argument('--min_confidence', type=float, default=0.5,
                    help='Umbral mínimo de confianza para predicciones')
parser.add_argument('--memory_safety_margin', type=float, default=0.2,
                    help='Margen de seguridad para memoria (0.2 = 20%%)')
parser.add_argument('--prefetch_tiles', type=int, default=16,
                    help='Número de tiles a precargar en memoria')
parser.add_argument('--save_npz', action='store_true',
                    help='Guardar tiles como NPZ para inspección')
parser.add_argument('--debug', action='store_true',
                    help='Modo debug con más información')

args = parser.parse_args()

# ============================================================================
# ESTRUCTURAS DE DATOS Y CONFIGURACIÓN
# ============================================================================

@dataclass
class TileInfo:
    """Información de un tile para procesamiento"""
    id: int
    row: int
    col: int
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    output_x: int
    output_y: int
    output_width: int
    output_height: int
    tile_data: Optional[np.ndarray] = None
    processed_mask: Optional[np.ndarray] = None
    read_time: float = 0.0
    infer_time: float = 0.0
    read_success: bool = True

@dataclass
class BatchInfo:
    """Información de un batch de tiles"""
    tiles: List[TileInfo]
    batch_input: np.ndarray
    start_time: float

# ============================================================================
# CLASE PARA MANEJO DE MEMORIA Y MONITOREO AVANZADO
# ============================================================================

class AdvancedMemoryMonitor:
    """Monitoriza y gestiona el uso de memoria de forma inteligente"""
    
    def __init__(self, safety_margin: float = 0.2):
        self.process = psutil.Process()
        self.system_memory = psutil.virtual_memory()
        self.start_memory = self.get_memory_mb()
        self.safety_margin = safety_margin
        self.history = []
        self.peak_memory = 0
        
        logger.info(f"Memoria total del sistema: {self.system_memory.total / (1024**3):.1f} GB")
        logger.info(f"Memoria disponible: {self.system_memory.available / (1024**3):.1f} GB")
    
    def get_memory_mb(self) -> float:
        """Obtiene uso de memoria en MB"""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_available_memory_mb(self) -> float:
        """Obtiene memoria disponible en MB"""
        return self.system_memory.available / (1024 * 1024)
    
    def get_memory_usage_percentage(self) -> float:
        """Obtiene porcentaje de uso de memoria"""
        return self.process.memory_percent()
    
    def update_stats(self):
        """Actualiza estadísticas de memoria"""
        current = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
        self.history.append(current)
        
        # Actualizar memoria del sistema
        self.system_memory = psutil.virtual_memory()
    
    def log_memory(self, label: str = "", detailed: bool = False):
        """Registra uso de memoria (sin caracteres Unicode para Windows)"""
        current = self.get_memory_mb()
        available = self.get_available_memory_mb()
        
        # Usar "delta" en lugar de Δ para Windows
        delta = current - self.start_memory
        delta_sign = "+" if delta >= 0 else ""
        
        logger.info(f"Memoria {label}: {current:.1f} MB (delta: {delta_sign}{delta:+.1f} MB)")
        
        if detailed:
            logger.info(f"  Disponible: {available:.1f} MB, Uso sistema: {self.system_memory.percent:.1f}%")
            logger.info(f"  Pico: {self.peak_memory:.1f} MB")
        
        self.update_stats()
        return current, available
    
    def can_allocate(self, size_mb: float) -> bool:
        """Verifica si se puede asignar cierta cantidad de memoria"""
        available = self.get_available_memory_mb()
        current = self.get_memory_mb()
        
        # Considerar margen de seguridad
        safe_available = available * (1.0 - self.safety_margin)
        
        can_alloc = size_mb < safe_available
        if not can_alloc:
            logger.warning(f"No se puede asignar {size_mb:.1f} MB. Disponible (seguro): {safe_available:.1f} MB")
        
        return can_alloc
    
    def estimate_batch_size(self, tile_size: int, channels: int = 3) -> int:
        """
        Estima el tamaño de batch óptimo basado en la memoria disponible
        """
        # Memoria aproximada por tile (float32)
        tile_memory_mb = (tile_size * tile_size * channels * 4) / (1024 * 1024)
        
        # Memoria para el batch completo (entrada + salida)
        # Asumir 2x para salidas del modelo y buffers
        batch_memory_per_tile = tile_memory_mb * 3
        
        # Memoria disponible segura
        safe_available = self.get_available_memory_mb() * (1.0 - self.safety_margin)
        
        # Calcular batch size máximo
        max_batch_by_memory = int(safe_available / batch_memory_per_tile)
        
        # Limitar por configuración y mínimo práctico
        max_batch = min(
            max_batch_by_memory,
            args.max_batch_size,
            16  # Límite absoluto
        )
        
        min_batch = max(args.min_batch_size, 1)
        
        batch_size = max(min_batch, min(max_batch, min_batch))
        
        logger.info(f"Estimacion batch size:")
        logger.info(f"  - Memoria por tile: {tile_memory_mb:.1f} MB")
        logger.info(f"  - Memoria disponible segura: {safe_available:.1f} MB")
        logger.info(f"  - Batch maximo teorico: {max_batch_by_memory}")
        logger.info(f"  - Batch seleccionado: {batch_size}")
        
        return batch_size

# ============================================================================
# CLASE PARA PROCESAMIENTO POR BATCHES
# ============================================================================

class BatchTileProcessor:
    """Procesa múltiples tiles simultáneamente en batches"""
    
    def __init__(self, session: rt.InferenceSession, memory_monitor: AdvancedMemoryMonitor):
        self.session = session
        self.memory_monitor = memory_monitor
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        
        # Estadísticas
        self.total_inference_time = 0.0
        self.total_tiles_processed = 0
        self.batch_count = 0
        
        # Configuración dinámica
        self.batch_size = 1  # Se ajustará dinámicamente
        self.current_batch_size = 1
    
    def process_batch(self, batch_tiles: List[TileInfo], batch_input: np.ndarray) -> List[TileInfo]:
        """
        Procesa un batch de tiles simultáneamente
        """
        if not batch_tiles:
            return []
        
        self.batch_count += 1
        batch_start = time.time()
        
        try:
            # Ejecutar inferencia en batch
            predictions = self.session.run(
                [self.output_name], 
                {self.input_name: batch_input}
            )[0]
            
            infer_time = time.time() - batch_start
            self.total_inference_time += infer_time
            
            # Procesar cada tile del batch
            processed_tiles = []
            for i, tile_info in enumerate(batch_tiles):
                if i < len(predictions):
                    pred = predictions[i]
                    
                    # Aplicar umbral de confianza
                    if args.min_confidence > 0:
                        max_probs = np.max(pred, axis=-1)
                        confidence_mask = max_probs >= args.min_confidence
                        
                        # Obtener clase con mayor probabilidad
                        mask_512 = np.argmax(pred, axis=-1).astype(np.uint8)
                        
                        # Aplicar máscara de confianza
                        mask_512[~confidence_mask] = 0
                    else:
                        mask_512 = np.argmax(pred, axis=-1).astype(np.uint8)
                    
                    # Recortar padding si es necesario
                    tile_width = tile_info.x_end - tile_info.x_start
                    tile_height = tile_info.y_end - tile_info.y_start
                    
                    if tile_width < args.tile_size or tile_height < args.tile_size:
                        final_mask = mask_512[:tile_height, :tile_width]
                    else:
                        final_mask = mask_512
                    
                    # Aplicar post-procesamiento básico
                    if not args.debug:
                        final_mask = self.apply_basic_postprocessing(final_mask)
                    
                    tile_info.processed_mask = final_mask
                    tile_info.infer_time = infer_time / len(batch_tiles)
                    self.total_tiles_processed += 1
                    processed_tiles.append(tile_info)
            
            logger.info(f"Batch {self.batch_count}: {len(processed_tiles)} tiles en {infer_time:.3f}s "
                       f"({infer_time/len(batch_tiles):.3f}s/tile)")
            
            return processed_tiles
            
        except Exception as e:
            logger.error(f"Error procesando batch: {e}")
            # Fallback: procesar tiles individualmente
            return self.process_fallback(batch_tiles, batch_input)
    
    def apply_basic_postprocessing(self, mask: np.ndarray) -> np.ndarray:
        """Post-procesamiento básico para un tile"""
        # Filtro de mediana rápido
        if mask.size > 0:
            mask = cv2.medianBlur(mask.astype(np.uint8), 3)
        return mask
    
    def process_fallback(self, batch_tiles: List[TileInfo], batch_input: np.ndarray) -> List[TileInfo]:
        """Fallback para procesamiento individual si falla el batch"""
        logger.warning("Usando fallback de procesamiento individual")
        
        processed_tiles = []
        for i, tile_info in enumerate(batch_tiles):
            try:
                # Procesar tile individualmente
                tile_start = time.time()
                individual_input = batch_input[i:i+1]  # Batch de tamaño 1
                
                predictions = self.session.run(
                    [self.output_name], 
                    {self.input_name: individual_input}
                )[0]
                
                pred = predictions[0]
                
                # Aplicar umbral de confianza
                if args.min_confidence > 0:
                    max_probs = np.max(pred, axis=-1)
                    confidence_mask = max_probs >= args.min_confidence
                    mask_512 = np.argmax(pred, axis=-1).astype(np.uint8)
                    mask_512[~confidence_mask] = 0
                else:
                    mask_512 = np.argmax(pred, axis=-1).astype(np.uint8)
                
                # Recortar si es necesario
                tile_width = tile_info.x_end - tile_info.x_start
                tile_height = tile_info.y_end - tile_info.y_start
                
                if tile_width < args.tile_size or tile_height < args.tile_size:
                    final_mask = mask_512[:tile_height, :tile_width]
                else:
                    final_mask = mask_512
                
                tile_info.processed_mask = final_mask
                tile_info.infer_time = time.time() - tile_start
                processed_tiles.append(tile_info)
                
            except Exception as e:
                logger.error(f"Error en fallback para tile {tile_info.id}: {e}")
        
        return processed_tiles
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del procesador"""
        if self.total_tiles_processed > 0:
            avg_inference = self.total_inference_time / self.total_tiles_processed
            avg_batch = self.total_tiles_processed / max(self.batch_count, 1)
        else:
            avg_inference = 0
            avg_batch = 0
            
        return {
            'total_batches': self.batch_count,
            'total_tiles': self.total_tiles_processed,
            'total_inference_time': self.total_inference_time,
            'avg_inference_time_per_tile': avg_inference,
            'avg_batch_size': avg_batch
        }

# ============================================================================
# FUNCIONES DE PRE/POST-PROCESAMIENTO MEJORADAS
# ============================================================================

def normalize_tile_batch(tile_batch: np.ndarray) -> np.ndarray:
    """
    Normaliza un batch de tiles
    """
    if tile_batch.dtype != np.float32:
        tile_batch = tile_batch.astype(np.float32)
    
    return tile_batch / 127.5 - 1.0

def prepare_batch_input(tiles: List[TileInfo], tile_size: int) -> Optional[np.ndarray]:
    """
    Prepara un batch de tiles para inferencia
    """
    # Filtrar tiles válidos (con datos)
    valid_tiles = []
    for tile_info in tiles:
        if tile_info.tile_data is not None and tile_info.read_success:
            valid_tiles.append(tile_info)
    
    if not valid_tiles:
        logger.warning("No hay tiles válidos para procesar en este batch")
        return None
    
    batch_size = len(valid_tiles)
    batch_input = np.zeros((batch_size, tile_size, tile_size, 3), dtype=np.float32)
    
    for i, tile_info in enumerate(valid_tiles):
        tile_data = tile_info.tile_data
        
        # Verificar que los datos son válidos
        if tile_data is None:
            logger.warning(f"Tile {tile_info.id} tiene datos None, saltando")
            continue
            
        # Aplicar padding si es necesario
        if tile_data.shape[0] < tile_size or tile_data.shape[1] < tile_size:
            padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            padded_tile[:tile_data.shape[0], :tile_data.shape[1], :] = tile_data
            batch_input[i] = padded_tile
        else:
            batch_input[i] = tile_data
        
        # Liberar memoria del tile individual
        tile_info.tile_data = None
    
    return batch_input

def apply_median_filter(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Filtro de mediana optimizado"""
    if mask.size == 0:
        return mask
    
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.medianBlur(mask.astype(np.uint8), kernel_size)

def apply_morphological_operations(mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Operaciones morfológicas optimizadas"""
    if mask.size == 0:
        return mask
    
    cleaned_mask = mask.copy()
    
    # Procesar cada clase excepto fondo
    for class_id in range(1, 4):
        class_mask = (mask == class_id).astype(np.uint8)
        
        if np.sum(class_mask) > 0:
            # Componentes conectados
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            
            # Eliminar componentes pequeños
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] < min_size:
                    cleaned_mask[labels == label] = 0
    
    return cleaned_mask

def blend_tile_edges(current_tile: np.ndarray, 
                     global_mask: np.ndarray,
                     x: int, y: int, 
                     width: int, height: int,
                     overlap: int) -> None:
    """
    Mezcla optimizada de bordes de tiles
    """
    h, w = current_tile.shape
    
    # Usar mezcla lineal simple para mayor velocidad
    blend_region = min(overlap // 2, h // 4, w // 4)
    
    if blend_region > 0:
        # Extraer región existente
        existing_region = global_mask[y:y+h, x:x+w]
        
        # Solo mezclar donde ambos tienen valores
        mask_current = current_tile > 0
        mask_existing = existing_region > 0
        overlap_mask = mask_current & mask_existing
        
        if np.any(overlap_mask):
            # Crear máscara de mezcla simple
            blend_mask = np.ones((h, w), dtype=np.float32)
            
            # Bordes horizontales
            for i in range(blend_region):
                alpha = i / blend_region
                blend_mask[i, :] = alpha
                blend_mask[-(i+1), :] = alpha
            
            # Bordes verticales
            for i in range(blend_region):
                alpha = i / blend_region
                blend_mask[:, i] = np.minimum(blend_mask[:, i], alpha)
                blend_mask[:, -(i+1)] = np.minimum(blend_mask[:, -(i+1)], alpha)
            
            # Mezclar
            blended = np.where(
                overlap_mask,
                current_tile.astype(np.float32) * blend_mask + 
                existing_region.astype(np.float32) * (1 - blend_mask),
                np.where(mask_current, current_tile, existing_region)
            )
            
            global_mask[y:y+h, x:x+w] = blended.astype(np.uint8)
        else:
            global_mask[y:y+h, x:x+w] = current_tile
    else:
        global_mask[y:y+h, x:x+w] = current_tile

# ============================================================================
# FUNCIÓN PARA GUARDAR BALANCED_ARGMAX.PNG
# ============================================================================

def save_balanced_argmax_image(mask: np.ndarray, output_path: Path, base_name: str):
    """
    Guarda la máscara como imagen PNG con colores específicos para balanced_argmax
    Esto es lo que el aplicativo espera para la vista de Segmentación
    """
    try:
        logger.info(f"\n[IMG] GENERANDO IMAGEN DE SEGMENTACIÓN (balanced_argmax)...")
        
        # Definir colores específicos para el aplicativo
        colors = [
            "#000000",  # Negro: Fondo
            "#ff7f00",  # Naranja: Mauritia flexuosa
            "#08F6EB",  # Cian: Euterpe precatoria  
            "#E008F6"   # Púrpura: Oenocarpus bataua
        ]
        custom_cmap = ListedColormap(colors)
        
        # Definir los límites de las clases (como en instancias_tiles.py)
        bounds = [0, 1, 2, 3, 4]  # Las clases son 0, 1, 2, 3
        norm = BoundaryNorm(bounds, custom_cmap.N)
        
        # Crear figura
        plt.figure(figsize=(12, 8))
        plt.imshow(mask, cmap=custom_cmap, norm=norm, interpolation='nearest')
        plt.axis('off')
        
        # Guardar con alta calidad
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        
        if output_path.exists():
            img_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"[OK] Imagen de segmentación guardada: {output_path} ({img_size:.2f} MB)")
        else:
            logger.error(f"[ERROR] No se pudo crear: {output_path}")
            
    except Exception as e:
        logger.error(f"Error guardando balanced_argmax.png: {e}")
        # Intentar con OpenCV como respaldo
        try:
            # Mapear colores BGR
            color_map = {
                0: [0, 0, 0],        # Negro
                1: [0, 127, 255],    # Naranja (BGR)
                2: [235, 246, 8],    # Cian (BGR)
                3: [246, 8, 224]     # Púrpura (BGR)
            }
            
            # Crear imagen RGB
            h, w = mask.shape
            rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
            
            for class_id, color in color_map.items():
                rgb_image[mask == class_id] = color
            
            # Guardar con OpenCV
            cv2.imwrite(str(output_path), rgb_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            logger.info(f"[BACKUP] Imagen guardada con OpenCV: {output_path}")
            
        except Exception as e2:
            logger.error(f"Error en backup OpenCV: {e2}")

def save_colored_mask(mask: np.ndarray, output_path: Path):
    """
    Guarda la máscara como imagen RGB coloreada (versión original)
    """
    try:
        colors = np.array([
            [0, 0, 0],        # 0: Negro - Fondo
            [0, 127, 255],    # 1: Naranja (BGR)
            [235, 246, 8],    # 2: Cian (BGR)
            [246, 8, 224]     # 3: Púrpura (BGR)
        ], dtype=np.uint8)
        
        if mask.size > 0:
            color_mask = colors[mask]
            cv2.imwrite(str(output_path), color_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            logger.info(f"Visualizacion guardada: {output_path}")
        else:
            logger.warning("Mascara vacia, no se guarda visualizacion")
            
    except Exception as e:
        logger.error(f"Error guardando mascara coloreada: {e}")

# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO CON BATCHES
# ============================================================================

def process_image_with_batch():
    """
    Procesa la imagen por tiles con procesamiento por batches para máxima velocidad
    """
    # Crear directorios de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_npz:
        npz_dir = output_dir / 'tiles_npz'
        npz_dir.mkdir(exist_ok=True)
    
    # Inicializar monitor de memoria avanzado
    memory_monitor = AdvancedMemoryMonitor(args.memory_safety_margin)
    logger.info("=" * 70)
    logger.info("INICIANDO SEGMENTACIÓN POR TILES CON PROCESAMIENTO POR BATCHES")
    logger.info("=" * 70)
    
    # 1. Cargar Modelo ONNX optimizado
    logger.info(f"Cargando modelo: {args.model}")
    try:
        # Configuración optimizada para batches
        sess_options = rt.SessionOptions()
        sess_options.enable_cpu_mem_arena = True
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Usar múltiples threads para inferencia
        sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        sess_options.inter_op_num_threads = 1
        
        session = rt.InferenceSession(
            args.model, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        logger.info(f"Modelo cargado. Forma de entrada: {input_shape}")
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise
    
    # 2. Abrir Imagen con GDAL
    logger.info(f"Abriendo imagen: {args.image_path}")
    
    # Configurar GDAL para manejar errores de TIFF
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetConfigOption('GDAL_NUM_THREADS', '2')
    
    dataset = gdal.Open(args.image_path)
    if dataset is None:
        logger.error(f"No se pudo abrir la imagen: {args.image_path}")
        sys.exit(1)
    
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    
    # Obtener nombre base del archivo
    base_name = Path(args.image_path).stem
    logger.info(f"Imagen: {width}x{height} pixels, {bands} bandas, Nombre: {base_name}")
    
    # 3. Calcular configuración de tiles
    tile_size = args.tile_size
    overlap = args.overlap
    effective_size = tile_size - 2 * overlap
    
    # Ajustar solapamiento si es necesario
    if overlap >= tile_size // 2:
        logger.warning(f"Solapamiento ({overlap}) muy grande. Reduciendo a {tile_size//4}")
        overlap = tile_size // 4
        effective_size = tile_size - 2 * overlap
    
    cols = math.ceil(width / effective_size)
    rows = math.ceil(height / effective_size)
    total_tiles = cols * rows
    
    # 4. Determinar batch size dinámico
    dynamic_batch_size = memory_monitor.estimate_batch_size(tile_size)
    
    logger.info(f"\nConfiguracion de procesamiento:")
    logger.info(f"  - Tamaño tile: {tile_size}x{tile_size}")
    logger.info(f"  - Solapamiento: {overlap} pixels")
    logger.info(f"  - Tamaño efectivo: {effective_size}x{effective_size}")
    logger.info(f"  - Tiles totales: {rows} filas x {cols} columnas = {total_tiles}")
    logger.info(f"  - Batch size dinamico: {dynamic_batch_size}")
    logger.info(f"  - Tiles a precargar: {args.prefetch_tiles}")
    
    # 5. Crear archivos de salida
    driver = gdal.GetDriverByName('GTiff')
    # CORRECCIÓN CRÍTICA: Cambiar el nombre para que coincida con lo que espera instancias_tiles.py
    out_tif = output_dir / f'{base_name}_balanced_argmax.tif'  # Cambiado de _segmented.tif
    
    tif_options = [
        'COMPRESS=LZW',
        'PREDICTOR=2',
        'BIGTIFF=IF_SAFER',
        'TILED=YES',
        'BLOCKXSIZE=256',
        'BLOCKYSIZE=256'
    ]
    
    out_ds = driver.Create(
        str(out_tif), width, height, 1, gdal.GDT_Byte,
        options=tif_options
    )
    if out_ds is None:
        logger.error("No se pudo crear el archivo de salida")
        sys.exit(1)
        
    out_ds.SetGeoTransform(dataset.GetGeoTransform())
    out_ds.SetProjection(dataset.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(0)
    
    # Máscara global en memoria
    full_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 6. Inicializar procesador de batches
    batch_processor = BatchTileProcessor(session, memory_monitor)
    batch_processor.batch_size = dynamic_batch_size
    
    # 7. Procesamiento principal con batches
    logger.info("\n" + "=" * 70)
    logger.info("INICIANDO PROCESAMIENTO POR BATCHES")
    logger.info("=" * 70)
    
    total_start = time.time()
    tile_counter = 0
    processed_pixels = 0
    
    # Estadísticas
    batch_times = []
    read_times = []
    
    # Pool de threads para lectura
    read_pool = ThreadPoolExecutor(max_workers=min(2, os.cpu_count() or 1))  # Reducido para estabilidad
    
    # Procesar por filas para mejor localidad de cache
    for row in range(rows):
        row_start_time = time.time()
        logger.info(f"\nProcesando fila {row+1}/{rows}")
        
        # Crear lista de tiles para esta fila
        row_tiles = []
        for col in range(cols):
            tile_counter += 1
            
            # Calcular coordenadas
            x_start = max(0, col * effective_size - overlap)
            y_start = max(0, row * effective_size - overlap)
            
            x_end = min(width, x_start + tile_size)
            y_end = min(height, y_start + tile_size)
            
            # Ajustar inicio si es necesario
            if x_end - x_start < tile_size:
                x_start = max(0, x_end - tile_size)
            if y_end - y_start < tile_size:
                y_start = max(0, y_end - tile_size)
            
            tile_width = x_end - x_start
            tile_height = y_end - y_start
            
            # Coordenadas de salida
            output_x = col * effective_size
            output_y = row * effective_size
            output_width = min(effective_size, width - output_x)
            output_height = min(effective_size, height - output_y)
            
            if output_width <= 0 or output_height <= 0:
                continue
            
            row_tiles.append(TileInfo(
                id=tile_counter,
                row=row,
                col=col,
                x_start=x_start,
                y_start=y_start,
                x_end=x_end,
                y_end=y_end,
                output_x=output_x,
                output_y=output_y,
                output_width=output_width,
                output_height=output_height
            ))
        
        # Leer tiles de esta fila en paralelo
        read_start = time.time()
        
        for tile_info in row_tiles:
            tile_data = read_tile_data_safe(dataset, tile_info)
            if tile_data is not None:
                tile_info.tile_data = tile_data
                tile_info.read_success = True
            else:
                tile_info.read_success = False
                logger.warning(f"Tile {tile_info.id} no se pudo leer correctamente")
        
        read_time = time.time() - read_start
        read_times.append(read_time)
        
        # Contar tiles exitosos
        successful_tiles = sum(1 for t in row_tiles if t.read_success)
        logger.info(f"  - Leidos {successful_tiles}/{len(row_tiles)} tiles en {read_time:.2f}s")
        
        # Procesar en batches solo los tiles exitosos
        successful_row_tiles = [t for t in row_tiles if t.read_success]
        
        if not successful_row_tiles:
            logger.warning(f"  - No hay tiles validos en esta fila, saltando...")
            continue
        
        batch_start = time.time()
        batch_results = []
        
        for i in range(0, len(successful_row_tiles), dynamic_batch_size):
            batch_slice = successful_row_tiles[i:i + dynamic_batch_size]
            
            if not batch_slice:
                continue
            
            # Verificar memoria antes de procesar
            current_mem, available_mem = memory_monitor.log_memory(
                f"antes de batch {i//dynamic_batch_size + 1}", 
                detailed=False
            )
            
            # Preparar batch
            batch_input = prepare_batch_input(batch_slice, tile_size)
            
            if batch_input is None:
                logger.warning(f"  - No se pudo preparar batch, saltando...")
                continue
            
            # Procesar batch
            batch_result = batch_processor.process_batch(batch_slice, batch_input)
            batch_results.extend(batch_result)
            
            # Liberar memoria inmediatamente
            del batch_input
            gc.collect()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Escribir resultados de esta fila
        write_start = time.time()
        tiles_written = 0
        
        for tile_info in batch_results:
            if tile_info.processed_mask is not None:
                try:
                    # Extraer región central
                    offset_x = tile_info.output_x - tile_info.x_start
                    offset_y = tile_info.output_y - tile_info.y_start
                    
                    # Verificar que los offsets sean válidos
                    if (offset_y >= 0 and offset_y + tile_info.output_height <= tile_info.processed_mask.shape[0] and
                        offset_x >= 0 and offset_x + tile_info.output_width <= tile_info.processed_mask.shape[1]):
                        
                        central_region = tile_info.processed_mask[
                            offset_y:offset_y + tile_info.output_height,
                            offset_x:offset_x + tile_info.output_width
                        ]
                        
                        # Mezclar con máscara global
                        blend_tile_edges(
                            central_region,
                            full_mask,
                            tile_info.output_x,
                            tile_info.output_y,
                            tile_info.output_width,
                            tile_info.output_height,
                            overlap // 2
                        )
                        
                        # Escribir al TIFF
                        out_band.WriteArray(
                            central_region, 
                            tile_info.output_x, 
                            tile_info.output_y
                        )
                        
                        tiles_written += 1
                        processed_pixels += tile_info.output_width * tile_info.output_height
                    else:
                        logger.warning(f"  - Offset invalido para tile {tile_info.id}, saltando...")
                        
                except Exception as e:
                    logger.error(f"  - Error procesando tile {tile_info.id}: {e}")
        
        write_time = time.time() - write_start
        
        # Estadísticas de la fila
        row_time = time.time() - row_start_time
        progress = (tile_counter / total_tiles) * 100
        
        logger.info(f"  - Procesados: {len(batch_results)} tiles")
        logger.info(f"  - Escritos: {tiles_written} tiles")
        logger.info(f"  - Tiempo batch: {batch_time:.2f}s")
        logger.info(f"  - Tiempo escritura: {write_time:.2f}s")
        logger.info(f"  - Tiempo total fila: {row_time:.2f}s")
        logger.info(f"  - Progreso total: {progress:.1f}%")
        
        # Liberar memoria de la fila
        for tile_info in row_tiles:
            tile_info.tile_data = None
            tile_info.processed_mask = None
        
        gc.collect()
        
        # Monitoreo de memoria cada fila
        if row % 2 == 0:
            memory_monitor.log_memory(f"despues de fila {row+1}", detailed=True)
    
    # Cerrar pool de lectura
    read_pool.shutdown(wait=True)
    
    # 8. Post-procesamiento global
    logger.info("\n" + "=" * 70)
    logger.info("POST-PROCESAMIENTO GLOBAL")
    logger.info("=" * 70)
    
    post_start = time.time()
    
    try:
        # Aplicar filtros finales a la máscara completa
        full_mask_clean = apply_median_filter(full_mask, kernel_size=5)
        full_mask_clean = apply_morphological_operations(full_mask_clean, min_size=20)
        
        # Escribir máscara final
        out_band.WriteArray(full_mask_clean, 0, 0)
        
        post_time = time.time() - post_start
        logger.info(f"Post-procesamiento completado en {post_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error en post-procesamiento: {e}")
        # Usar máscara sin post-procesar
        full_mask_clean = full_mask
    
    # 9. Generar imagen balanced_argmax.png (CRÍTICO PARA EL APLICATIVO)
    balanced_argmax_png_path = output_dir / f'{base_name}_balanced_argmax.png'
    save_balanced_argmax_image(full_mask_clean, balanced_argmax_png_path, base_name)
    
    # 10. Guardar también la visualización original
    verification_path = output_dir / f'{base_name}_verificacion.png'
    save_colored_mask(full_mask_clean, verification_path)
    
    # 11. Finalizar
    out_band.FlushCache()
    out_ds = None
    dataset = None
    
    # 12. Estadísticas finales
    total_time = time.time() - total_start
    
    logger.info("\n" + "=" * 70)
    logger.info("ESTADISTICAS FINALES DEL PROCESAMIENTO POR BATCHES")
    logger.info("=" * 70)
    
    # Estadísticas del procesador
    proc_stats = batch_processor.get_statistics()
    logger.info(f"Total batches procesados: {proc_stats['total_batches']}")
    logger.info(f"Total tiles procesados: {proc_stats['total_tiles']}")
    logger.info(f"Tiempo total inferencia: {proc_stats['total_inference_time']:.2f}s")
    logger.info(f"Tiempo promedio por tile: {proc_stats['avg_inference_time_per_tile']:.3f}s")
    logger.info(f"Batch size promedio: {proc_stats['avg_batch_size']:.1f}")
    
    # Velocidad comparativa
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        logger.info(f"Tiempo promedio por batch: {avg_batch_time:.2f}s")
    
    if read_times:
        avg_read_time = np.mean(read_times)
        logger.info(f"Tiempo promedio lectura: {avg_read_time:.2f}s")
    
    # Velocidad estimada vs procesamiento individual
    estimated_single_time = proc_stats['total_tiles'] * 0.5  # Asumiendo 0.5s por tile individual
    if estimated_single_time > 0 and total_time > 0:
        speedup = estimated_single_time / total_time
        logger.info(f"\nVelocidad estimada vs procesamiento 1x1: {speedup:.1f}x mas rapido")
    
    logger.info(f"\nTiempo total procesamiento: {total_time:.2f}s ({total_time/60:.1f} min)")
    logger.info(f"Pixeles procesados: {processed_pixels:,}")
    if total_time > 0:
        logger.info(f"Velocidad: {processed_pixels/total_time:,.0f} px/s")
    
    # Distribución de clases
    unique, counts = np.unique(full_mask_clean, return_counts=True)
    total_pixels = width * height
    
    logger.info("\nDISTRIBUCION FINAL DE CLASES:")
    for u, c in zip(unique, counts):
        percentage = (c / total_pixels) * 100 if total_pixels > 0 else 0
        logger.info(f"  Clase {u}: {c:,} pixeles ({percentage:.2f}%)")
    
    # Estadísticas de memoria
    memory_monitor.log_memory("FINAL", detailed=True)
    
    logger.info(f"\nARCHIVOS GENERADOS EN: {output_dir}")
    logger.info(f"  - Segmentacion TIFF: {out_tif}")
    logger.info(f"  - Imagen Segmentacion (PNG): {balanced_argmax_png_path} <-- PARA EL APLICATIVO")
    logger.info(f"  - Visualizacion: {verification_path}")
    
    if args.save_npz:
        logger.info(f"  - Tiles NPZ: {npz_dir}")
    
    logger.info("=" * 70)
    logger.info("PROCESAMIENTO POR BATCHES COMPLETADO EXITOSAMENTE")
    logger.info("=" * 70)

def read_tile_data_safe(dataset: gdal.Dataset, tile_info: TileInfo) -> Optional[np.ndarray]:
    """
    Lee los datos de un tile individual de forma segura
    """
    try:
        read_start = time.time()
        
        # Leer datos con manejo de errores
        data = dataset.ReadAsArray(
            tile_info.x_start,
            tile_info.y_start,
            tile_info.x_end - tile_info.x_start,
            tile_info.y_end - tile_info.y_start
        )
        
        if data is None:
            # Intentar leer banda por banda
            logger.warning(f"Tile {tile_info.id}: Fallo lectura directa, intentando banda por banda...")
            return read_tile_band_by_band(dataset, tile_info)
        
        # Formatear y normalizar
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]
        
        # Verificar que tenemos al menos 3 bandas
        if data.shape[0] < 3:
            logger.warning(f"Tile {tile_info.id}: Solo {data.shape[0]} bandas, duplicando...")
            if data.shape[0] == 1:
                # Duplicar banda única para RGB
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                # Añadir tercera banda con ceros
                data = np.vstack([data, np.zeros((1, data.shape[1], data.shape[2]), dtype=data.dtype)])
        
        # Convertir a (H, W, C) y mantener 3 bandas
        tile_rgb = np.transpose(data[:3, ...], (1, 2, 0)).astype(np.float32)
        
        # Aplicar normalización si es necesario
        if args.scaling == 'normalize':
            tile_rgb = tile_rgb / 127.5 - 1.0
        
        tile_info.read_time = time.time() - read_start
        return tile_rgb
        
    except Exception as e:
        logger.error(f"Error en read_tile_data_safe para tile {tile_info.id}: {e}")
        return None

def read_tile_band_by_band(dataset: gdal.Dataset, tile_info: TileInfo) -> Optional[np.ndarray]:
    """
    Lee un tile banda por banda (más robusto para TIFFs problemáticos)
    """
    try:
        width = tile_info.x_end - tile_info.x_start
        height = tile_info.y_end - tile_info.y_start
        
        # Leer cada banda por separado
        bands_data = []
        for band_idx in range(1, min(4, dataset.RasterCount + 1)):  # Máximo 3 bandas
            band = dataset.GetRasterBand(band_idx)
            band_data = band.ReadAsArray(
                tile_info.x_start,
                tile_info.y_start,
                width,
                height
            )
            
            if band_data is None:
                logger.warning(f"Tile {tile_info.id}: Banda {band_idx} no se pudo leer")
                # Crear banda de ceros
                band_data = np.zeros((height, width), dtype=np.float32)
            
            bands_data.append(band_data)
        
        # Si no tenemos 3 bandas, duplicar o añadir
        while len(bands_data) < 3:
            bands_data.append(np.zeros((height, width), dtype=np.float32))
        
        # Apilar bandas
        data = np.stack(bands_data[:3], axis=0)  # (3, H, W)
        
        # Convertir a (H, W, C)
        tile_rgb = np.transpose(data, (1, 2, 0)).astype(np.float32)
        
        # Aplicar normalización si es necesario
        if args.scaling == 'normalize':
            tile_rgb = tile_rgb / 127.5 - 1.0
        
        logger.info(f"Tile {tile_info.id}: Leido banda por banda exitosamente")
        return tile_rgb
        
    except Exception as e:
        logger.error(f"Error en read_tile_band_by_band para tile {tile_info.id}: {e}")
        return None

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    try:
        logger.info("Iniciando segmentacion semantica con procesamiento por batches")
        logger.info(f"Parametros: tile_size={args.tile_size}, overlap={args.overlap}")
        logger.info(f"Batch size: {args.min_batch_size}-{args.max_batch_size}")
        logger.info(f"Margen seguridad memoria: {args.memory_safety_margin*100:.0f}%")
        
        process_image_with_batch()
        
    except KeyboardInterrupt:
        logger.warning("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error durante la ejecucion: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Programa finalizado correctamente")
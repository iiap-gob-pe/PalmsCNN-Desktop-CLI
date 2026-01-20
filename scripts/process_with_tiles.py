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
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Habilitar excepciones de GDAL
gdal.UseExceptions()

# ============================================================================
# CONFIGURACIÓN DE LOGGING Y ARGUMENTOS (SIN UNICODE PARA WINDOWS)
# ============================================================================

# Configurar logging para Windows (SIN caracteres Unicode)
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
    description='Segmentacion por Tiles con Procesamiento por Batches para Maxima Velocidad'
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
parser.add_argument('--max_batch_size', type=int, default=96,  # <--- CAMBIADO DE 1024 A 96
                    help='Tamaño maximo del batch para inferencia')
parser.add_argument('--min_batch_size', type=int, default=64,  # <--- CAMBIADO DE 512 A 64
                    help='Tamaño minimo del batch para inferencia')
parser.add_argument('--min_confidence', type=float, default=0.5,
                    help='Umbral minimo de confianza para predicciones')
parser.add_argument('--memory_safety_margin', type=float, default=0.005,
                    help='Margen de seguridad para memoria (0.005 = 0.5%%)')
parser.add_argument('--prefetch_tiles', type=int, default=10000,
                    help='Numero de tiles a precargar en memoria')
parser.add_argument('--save_npz', action='store_true',
                    help='Guardar tiles como NPZ para inspeccion')
parser.add_argument('--debug', action='store_true',
                    help='Modo debug con mas informacion')

args = parser.parse_args()

# ============================================================================
# ESTRUCTURAS DE DATOS Y CONFIGURACIÓN
# ============================================================================

@dataclass
class TileInfo:
    """Informacion de un tile para procesamiento"""
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
    """Informacion de un batch de tiles"""
    tiles: List[TileInfo]
    batch_input: np.ndarray
    start_time: float

# ============================================================================
# CLASE PARA MANEJO DE MEMORIA Y MONITOREO EXTREMO
# ============================================================================

class ExtremeMemoryMonitor:
    """Monitoriza y gestiona el uso de memoria de forma EXTREMA"""
    
    def __init__(self, safety_margin: float = 0.005):
        self.process = psutil.Process()
        self.system_memory = psutil.virtual_memory()
        self.start_memory = self.get_memory_mb()
        self.safety_margin = safety_margin
        self.history = []
        self.peak_memory = 0
        
        logger.info("MODO EXTREMO - USO 100% DE RAM ACTIVADO")
        logger.info(f"Memoria total del sistema: {self.system_memory.total / (1024**3):.1f} GB")
        logger.info(f"Memoria disponible: {self.system_memory.available / (1024**3):.1f} GB")
        logger.info(f"Margen de seguridad: {self.safety_margin*100:.1f}%")
    
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
        """Actualiza estadisticas de memoria"""
        current = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
        self.history.append(current)
        
        # Actualizar memoria del sistema
        self.system_memory = psutil.virtual_memory()
    
    def log_memory(self, label: str = "", detailed: bool = False):
        """Registra uso de memoria"""
        current = self.get_memory_mb()
        available = self.get_available_memory_mb()
        
        delta = current - self.start_memory
        delta_sign = "+" if delta >= 0 else ""
        
        logger.info(f"Memoria {label}: {current:.1f} MB (delta: {delta_sign}{delta:+.1f} MB)")
        
        if detailed:
            usage_percent = (self.system_memory.total - available) / self.system_memory.total * 100
            logger.info(f"  Disponible: {available:.1f} MB, Uso sistema: {usage_percent:.1f}%")
            logger.info(f"  Pico: {self.peak_memory:.1f} MB")
        
        self.update_stats()
        return current, available
    
    def can_allocate(self, size_mb: float) -> bool:
        """Verifica si se puede asignar cierta cantidad de memoria"""
        available = self.get_available_memory_mb()
        current = self.get_memory_mb()
        
        # Considerar margen de seguridad MUY pequeño
        safe_available = available * (1.0 - self.safety_margin)
        
        can_alloc = size_mb < safe_available
        if not can_alloc:
            logger.warning(f"No se puede asignar {size_mb:.1f} MB. Disponible (seguro): {safe_available:.1f} MB")
        
        return can_alloc
    
    def estimate_batch_size(self, tile_size: int, channels: int = 3) -> int:
        """
        Estima el tamaño de batch optimo - CORREGIDO PARA ESTABILIDAD
        """
        # Memoria aproximada por tile (float32)
        tile_memory_mb = (tile_size * tile_size * channels * 4) / (1024 * 1024)
        
        # Factor de seguridad mayor para evitar el crash de ONNX
        batch_memory_per_tile = tile_memory_mb * 6  # Aumentado factor de seguridad
        
        # Memoria disponible segura
        safe_available = self.get_available_memory_mb() * 0.8 # Usar 80% real, no 99%
        
        # Calcular batch size máximo teórico
        max_batch_by_memory = int(safe_available / batch_memory_per_tile)
        
        # LIMITAR ESTRICTAMENTE A 96 (El punto dulce de tu CPU)
        # No importa cuanta RAM tengas, la CPU no procesa mas rapido con batches de 1000
        LIMIT_BATCH = 96
        
        batch_size = min(max_batch_by_memory, args.max_batch_size, LIMIT_BATCH)
        
        # Asegurar mínimo
        batch_size = max(batch_size, 1)
        
        logger.info(f"Estimacion batch size (CORREGIDO):")
        logger.info(f"  - Memoria por tile: {tile_memory_mb:.1f} MB")
        logger.info(f"  - Memoria disponible segura: {safe_available:.1f} MB")
        logger.info(f"  - Batch seleccionado: {batch_size} (Limitado para estabilidad)")
        
        return batch_size

# ============================================================================
# CLASE PARA PROCESAMIENTO EXTREMO POR BATCHES
# ============================================================================

class ExtremeBatchTileProcessor:
    """Procesa multiples tiles simultaneamente en batches EXTREMOS"""
    
    def __init__(self, session: rt.InferenceSession, memory_monitor: ExtremeMemoryMonitor):
        self.session = session
        self.memory_monitor = memory_monitor
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        
        # Estadisticas
        self.total_inference_time = 0.0
        self.total_tiles_processed = 0
        self.batch_count = 0
        
        # Configuracion dinamica EXTREMA
        self.batch_size = 512
        self.current_batch_size = 512
        
        # Para procesamiento paralelo
        self.num_workers = min(8, mp.cpu_count())
    
    def process_extreme_batch(self, batch_tiles: List[TileInfo], batch_input: np.ndarray) -> List[TileInfo]:
        """
        Procesa un batch EXTREMO de tiles simultaneamente
        """
        if not batch_tiles:
            return []
        
        self.batch_count += 1
        batch_start = time.time()
        
        try:
            # Ejecutar inferencia en batch (MUCHO MÁS GRANDE)
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
                        
                        # Aplicar mascara de confianza
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
                    
                    # Aplicar post-procesamiento basico
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
            logger.error(f"Error procesando batch extremo: {e}")
            # Fallback: procesar en batches más pequeños
            return self.process_batch_fallback(batch_tiles, batch_input)
    
    def process_batch_fallback(self, batch_tiles: List[TileInfo], batch_input: np.ndarray) -> List[TileInfo]:
        """Fallback para procesamiento en batches más pequeños"""
        logger.warning("Usando fallback de procesamiento en batches pequeños")
        
        processed_tiles = []
        batch_size = 64  # Batch más pequeño para fallback
        
        for i in range(0, len(batch_tiles), batch_size):
            sub_batch = batch_tiles[i:i + batch_size]
            sub_input = batch_input[i:i + batch_size]
            
            try:
                sub_predictions = self.session.run(
                    [self.output_name], 
                    {self.input_name: sub_input}
                )[0]
                
                for j, tile_info in enumerate(sub_batch):
                    if j < len(sub_predictions):
                        pred = sub_predictions[j]
                        
                        # Procesar tile individualmente
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
                        processed_tiles.append(tile_info)
                        self.total_tiles_processed += 1
                        
            except Exception as e:
                logger.error(f"Error en fallback batch {i//batch_size}: {e}")
        
        return processed_tiles
    
    def apply_basic_postprocessing(self, mask: np.ndarray) -> np.ndarray:
        """Post-procesamiento basico para un tile"""
        # Filtro de mediana rapido
        if mask.size > 0:
            mask = cv2.medianBlur(mask.astype(np.uint8), 3)
        return mask
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadisticas del procesador"""
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
            'avg_batch_size': avg_batch,
            'num_workers': self.num_workers
        }

# ============================================================================
# FUNCIONES PARA GENERAR BALANCED_ARGMAX.PNG
# ============================================================================

def save_balanced_argmax_image(mask: np.ndarray, output_path: Path, base_name: str):
    """
    Guarda la máscara como imagen PNG con colores específicos para balanced_argmax
    Esto es lo que el aplicativo espera para la vista de Segmentación
    """
    try:
        logger.info(f"\n[IMG] GENERANDO IMAGEN DE SEGMENTACION (balanced_argmax)...")
        
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
            logger.info(f"[OK] Imagen de segmentacion guardada: {output_path} ({img_size:.2f} MB)")
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
# FUNCIONES PARA REPORTE FINAL
# ============================================================================

def generar_reporte_final(full_mask_clean: np.ndarray, output_dir: Path, 
                         base_name: str, width: int, height: int,
                         total_time: float, total_tiles: int,
                         processed_pixels: int, proc_stats: Dict[str, Any],
                         batch_times: List[float], read_times: List[float]) -> None:
    """
    Genera un reporte final detallado del procesamiento
    """
    report_path = output_dir / f'{base_name}_reporte_final.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("REPORTE FINAL DE SEGMENTACION SEMANTICA\n")
        f.write("=" * 80 + "\n\n")
        
        # Informacion basica
        f.write("1. INFORMACION DE ENTRADA:\n")
        f.write(f"   - Imagen: {args.image_path}\n")
        f.write(f"   - Modelo: {args.model}\n")
        f.write(f"   - Dimensiones: {width} x {height} px\n")
        f.write(f"   - Tiles totales: {total_tiles}\n\n")
        
        # Configuracion
        f.write("2. CONFIGURACION DE PROCESAMIENTO:\n")
        f.write(f"   - Tamaño tile: {args.tile_size} px\n")
        f.write(f"   - Solapamiento: {args.overlap} px\n")
        f.write(f"   - Batch size: {args.min_batch_size}-{args.max_batch_size}\n")
        f.write(f"   - Umbral confianza: {args.min_confidence}\n\n")
        
        # Estadisticas de tiempo
        f.write("3. ESTADISTICAS DE TIEMPO:\n")
        f.write(f"   - Tiempo total: {total_time:.2f} segundos ({total_time/60:.1f} minutos)\n")
        f.write(f"   - Tiempo inferencia: {proc_stats['total_inference_time']:.2f} segundos\n")
        
        if batch_times:
            f.write(f"   - Tiempo promedio por batch: {np.mean(batch_times):.2f} segundos\n")
        if read_times:
            f.write(f"   - Tiempo promedio lectura: {np.mean(read_times):.2f} segundos\n")
        
        if total_time > 0:
            f.write(f"   - Velocidad procesamiento: {processed_pixels/total_time:,.0f} px/segundo\n\n")
        
        # Estadisticas de batches
        f.write("4. ESTADISTICAS DE BATCHES:\n")
        f.write(f"   - Batches procesados: {proc_stats['total_batches']}\n")
        f.write(f"   - Tiles procesados: {proc_stats['total_tiles']}\n")
        f.write(f"   - Tiempo promedio por tile: {proc_stats['avg_inference_time_per_tile']:.3f} segundos\n")
        f.write(f"   - Batch size promedio: {proc_stats['avg_batch_size']:.1f}\n\n")
        
        # Distribucion de clases
        f.write("5. DISTRIBUCION DE CLASES:\n")
        unique, counts = np.unique(full_mask_clean, return_counts=True)
        total_pixels = width * height
        
        for u, c in zip(unique, counts):
            percentage = (c / total_pixels) * 100 if total_pixels > 0 else 0
            nombre_clase = obtener_nombre_clase(u)
            f.write(f"   - {nombre_clase}: {c:,} px ({percentage:.2f}%)\n")
        
        # Archivos generados
        f.write("\n6. ARCHIVOS GENERADOS:\n")
        archivos = list(output_dir.glob(f"{base_name}*"))
        for archivo in sorted(archivos):
            if archivo.is_file():
                size_mb = archivo.stat().st_size / (1024 * 1024)
                f.write(f"   - {archivo.name} ({size_mb:.2f} MB)\n")
        
        # Verificacion de integridad
        f.write("\n7. VERIFICACION DE INTEGRIDAD:\n")
        if (output_dir / f'{base_name}_balanced_argmax.tif').exists():
            f.write("   ✓ Archivo TIFF generado correctamente\n")
        if (output_dir / f'{base_name}_balanced_argmax.png').exists():
            f.write("   ✓ Imagen PNG generada correctamente\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("PROCESAMIENTO COMPLETADO EXITOSAMENTE\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Reporte final generado: {report_path}")

def obtener_nombre_clase(id_clase: int) -> str:
    """Obtiene el nombre de la clase basado en el ID"""
    nombres = {
        0: "Fondo",
        1: "Mauritia flexuosa",
        2: "Euterpe precatoria",
        3: "Oenocarpus bataua"
    }
    return nombres.get(id_clase, f"Clase {id_clase}")

# ============================================================================
# VERIFICACION Y VALIDACION FINAL
# ============================================================================

def verificar_archivos_salida(output_dir: Path, base_name: str) -> bool:
    """
    Verifica que todos los archivos de salida se han generado correctamente
    """
    archivos_requeridos = [
        f'{base_name}_balanced_argmax.tif',
        f'{base_name}_balanced_argmax.png',
        f'{base_name}_verificacion.png'
    ]
    
    todos_ok = True
    logger.info("\nVERIFICANDO ARCHIVOS DE SALIDA:")
    
    for archivo in archivos_requeridos:
        ruta = output_dir / archivo
        if ruta.exists():
            size_mb = ruta.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {archivo} ({size_mb:.2f} MB)")
            
            # Verificar que no este vacio
            if size_mb < 0.001:
                logger.warning(f"    ADVERTENCIA: Archivo muy pequeno, podria estar vacio")
                todos_ok = False
        else:
            logger.error(f"  ✗ {archivo} NO ENCONTRADO")
            todos_ok = False
    
    return todos_ok

# ============================================================================
# LIMPIEZA FINAL Y OPTIMIZACION
# ============================================================================

def realizar_limpieza_final(dataset: gdal.Dataset, out_ds: gdal.Dataset,
                           full_mask_clean: np.ndarray, memory_monitor: ExtremeMemoryMonitor):
    """
    Realiza limpieza final de recursos y memoria
    """
    logger.info("\nREALIZANDO LIMPIEZA FINAL...")
    
    try:
        # Liberar recursos de GDAL
        if out_ds:
            out_ds.FlushCache()
            out_ds = None
        
        if dataset:
            dataset = None
        
        # Liberar arrays grandes
        if 'full_mask_clean' in locals():
            del full_mask_clean
        gc.collect()
        
        # Monitorear memoria final
        current_mem = memory_monitor.get_memory_mb()
        peak_mem = memory_monitor.peak_memory
        delta = current_mem - memory_monitor.start_memory
        
        logger.info(f"Memoria liberada. Uso actual: {current_mem:.1f} MB")
        logger.info(f"Pico de memoria: {peak_mem:.1f} MB")
        logger.info(f"Cambio total: {delta:+.1f} MB")
        
        # Forzar limpieza de garbage collector
        collected = gc.collect()
        if collected > 0:
            logger.info(f"Garbage collector libero {collected} objetos")
        
    except Exception as e:
        logger.warning(f"Advertencia en limpieza final: {e}")

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

def prepare_mega_batch_input(tiles: List[TileInfo], tile_size: int, max_batch_size: int) -> List[np.ndarray]:
    """
    Prepara MULTIPLES batches mega para inferencia
    """
    # Filtrar tiles validos
    valid_tiles = [t for t in tiles if t.tile_data is not None and t.read_success]
    
    if not valid_tiles:
        logger.warning("No hay tiles validos para procesar")
        return []
    
    batches_input = []
    current_batch = []
    
    for tile_info in valid_tiles:
        tile_data = tile_info.tile_data
        
        # Aplicar padding si es necesario
        if tile_data.shape[0] < tile_size or tile_data.shape[1] < tile_size:
            padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
            padded_tile[:tile_data.shape[0], :tile_data.shape[1], :] = tile_data
            current_batch.append(padded_tile)
        else:
            current_batch.append(tile_data)
        
        # Liberar memoria del tile individual
        tile_info.tile_data = None
        
        # Cuando llegamos al tamaño maximo del batch, crear array
        if len(current_batch) >= max_batch_size:
            batch_array = np.stack(current_batch, axis=0)
            batches_input.append(batch_array)
            current_batch = []
    
    # Añadir el ultimo batch si no esta vacio
    if current_batch:
        batch_array = np.stack(current_batch, axis=0)
        batches_input.append(batch_array)
    
    return batches_input

# ============================================================================
# FUNCIÓN PARA CARGA MASIVA DE TILES
# ============================================================================

def load_all_tiles_massively(dataset: gdal.Dataset, rows: int, cols: int, 
                            tile_size: int, effective_size: int, overlap: int,
                            width: int, height: int) -> List[TileInfo]:
    """
    Carga TODOS los tiles de la imagen en memoria de una vez
    """
    logger.info(f"CARGANDO TODOS LOS TILES EN MEMORIA...")
    
    all_tiles = []
    total_tiles = rows * cols
    loaded_count = 0
    
    start_time = time.time()
    
    for row in range(rows):
        for col in range(cols):
            loaded_count += 1
            
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
            
            # Crear objeto tile
            tile_info = TileInfo(
                id=loaded_count,
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
            )
            
            # Leer datos del tile
            try:
                data = dataset.ReadAsArray(x_start, y_start, tile_width, tile_height)
                
                if data is None:
                    tile_info.read_success = False
                    all_tiles.append(tile_info)
                    continue
                
                # Formatear y normalizar
                if len(data.shape) == 2:
                    data = data[np.newaxis, ...]
                
                # Verificar que tenemos al menos 3 bandas
                if data.shape[0] < 3:
                    if data.shape[0] == 1:
                        data = np.repeat(data, 3, axis=0)
                    elif data.shape[0] == 2:
                        data = np.vstack([data, np.zeros((1, data.shape[1], data.shape[2]), dtype=data.dtype)])
                
                # Convertir a (H, W, C)
                tile_rgb = np.transpose(data[:3, ...], (1, 2, 0)).astype(np.float32)
                
                # Aplicar normalizacion si es necesario
                if args.scaling == 'normalize':
                    tile_rgb = tile_rgb / 127.5 - 1.0
                
                tile_info.tile_data = tile_rgb
                tile_info.read_success = True
                
            except Exception as e:
                logger.debug(f"Error cargando tile {loaded_count}: {e}")
                tile_info.read_success = False
            
            all_tiles.append(tile_info)
            
            # Mostrar progreso cada 500 tiles
            if loaded_count % 500 == 0:
                elapsed = time.time() - start_time
                rate = loaded_count / elapsed
                logger.info(f"  - Cargados: {loaded_count}/{total_tiles} tiles ({rate:.1f} tiles/s)")
    
    load_time = time.time() - start_time
    successful_tiles = sum(1 for t in all_tiles if t.read_success)
    
    logger.info(f"CARGA MASIVA COMPLETADA: {successful_tiles}/{total_tiles} tiles en {load_time:.2f}s")
    logger.info(f"  - Velocidad: {loaded_count/load_time:.1f} tiles/s")
    
    return all_tiles

# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO EXTREMO
# ============================================================================

def process_image_extreme():
    """
    Procesa la imagen por tiles con procesamiento EXTREMO para usar 100% RAM
    """
    # Crear directorios de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar monitor de memoria EXTREMO
    memory_monitor = ExtremeMemoryMonitor(args.memory_safety_margin)
    
    logger.info("=" * 80)
    logger.info("INICIANDO SEGMENTACION POR TILES - MODO 100% RAM")
    logger.info("=" * 80)
    
    # 1. Cargar Modelo ONNX optimizado AL MÁXIMO
    logger.info(f"Cargando modelo: {args.model}")
    try:
        # Configuracion OPTIMIZADA AL MAXIMO
        sess_options = rt.SessionOptions()
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Usar TODOS los threads disponibles
        sess_options.intra_op_num_threads = mp.cpu_count()
        sess_options.inter_op_num_threads = 1
        
        # Cargar modelo
        session = rt.InferenceSession(
            args.model, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        logger.info(f"Modelo cargado. Forma de entrada: {input_shape}")
        logger.info(f"   - Threads de inferencia: {mp.cpu_count()}")
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise
    
    # 2. Abrir Imagen con GDAL optimizado
    logger.info(f"Abriendo imagen: {args.image_path}")
    
    # Configurar GDAL para maxima velocidad
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetConfigOption('GDAL_CACHEMAX', '512')  # 512 MB de cache
    
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
    
    # 3. Calcular configuracion de tiles
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
    
    logger.info(f"Configuracion de tiles:")
    logger.info(f"   - Tamaño tile: {tile_size}x{tile_size}")
    logger.info(f"   - Solapamiento: {overlap} pixels")
    logger.info(f"   - Tamaño efectivo: {effective_size}x{effective_size}")
    logger.info(f"   - Tiles totales: {rows} filas x {cols} columnas = {total_tiles}")
    
    # 4. Determinar batch size dinamico EXTREMO
    dynamic_batch_size = memory_monitor.estimate_batch_size(tile_size)
    
    # --- MODIFICACIÓN: QUITAR EL DOBLADO DE BATCH ---
    # ELIMINAR O COMENTAR ESTAS LINEAS SI EXISTEN:
    # if available_gb > 50:
    #     dynamic_batch_size = min(dynamic_batch_size * 2, args.max_batch_size)
    
    # FORZAR PREFETCH ALTO PARA USAR LA RAM COMO ALMACÉN
    # Esto llena la RAM con datos pendientes, no con datos procesando
    args.prefetch_tiles = 10000 
    
    logger.info(f"   - Batch size dinamico: {dynamic_batch_size}")
    logger.info(f"   - Tiles a precargar: {args.prefetch_tiles} (MAXIMO ABSOLUTO)")
    
    # 5. Crear archivos de salida
    driver = gdal.GetDriverByName('GTiff')
    out_tif = output_dir / f'{base_name}_balanced_argmax.tif'
    
    tif_options = [
        'COMPRESS=LZW',
        'PREDICTOR=2',
        'BIGTIFF=IF_SAFER',
        'TILED=YES',
        'BLOCKXSIZE=512',  # Bloques más grandes
        'BLOCKYSIZE=512',
        'NUM_THREADS=ALL_CPUS'
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
    
    # 6. CARGAR TODOS LOS TILES EN MEMORIA
    all_tiles = load_all_tiles_massively(dataset, rows, cols, tile_size, 
                                        effective_size, overlap, width, height)
    
    # Filtrar tiles exitosos
    successful_tiles = [t for t in all_tiles if t.read_success]
    logger.info(f"Tiles validos para procesamiento: {len(successful_tiles)}/{total_tiles}")
    
    # 7. Inicializar procesador EXTREMO
    batch_processor = ExtremeBatchTileProcessor(session, memory_monitor)
    batch_processor.batch_size = dynamic_batch_size
    
    # 8. PROCESAMIENTO MEGA-BATCH
    logger.info("\n" + "=" * 80)
    logger.info("PROCESANDO TODOS LOS TILES EN MEGA-BATCHES")
    logger.info("=" * 80)
    
    total_start = time.time()
    processed_count = 0
    processed_pixels = 0
    
    # Listas para estadisticas
    batch_times = []
    read_times = []
    
    # Preparar batches MEGA
    logger.info(f"Preparando batches mega de {dynamic_batch_size} tiles cada uno...")
    batches_input = prepare_mega_batch_input(successful_tiles, tile_size, dynamic_batch_size)
    
    logger.info(f"Total de batches a procesar: {len(batches_input)}")
    logger.info(f"Tiles por batch: ~{dynamic_batch_size}")
    
    # Procesar cada batch MEGA
    for batch_idx, batch_input in enumerate(batches_input, 1):
        batch_tiles_start = (batch_idx - 1) * dynamic_batch_size
        batch_tiles_end = min(batch_tiles_start + batch_input.shape[0], len(successful_tiles))
        batch_tiles = successful_tiles[batch_tiles_start:batch_tiles_end]
        
        logger.info(f"\nProcesando MEGA-BATCH {batch_idx}/{len(batches_input)}")
        logger.info(f"   - Tiles en batch: {len(batch_tiles)}")
        
        # Verificar memoria
        current_mem, available_mem = memory_monitor.log_memory(f"antes de MEGA-BATCH {batch_idx}", detailed=True)
        
        # Procesar batch EXTREMO
        batch_start = time.time()
        batch_results = batch_processor.process_extreme_batch(batch_tiles, batch_input)
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Actualizar progreso
        processed_count += len(batch_results)
        progress_percent = (processed_count / len(successful_tiles)) * 100
        
        # Imprimir progreso para la GUI
        print(f"[PROGRESS] {progress_percent:.2f}", flush=True)
        
        logger.info(f"   - Tiempo batch: {batch_time:.2f}s ({batch_time/len(batch_tiles):.3f}s/tile)")
        logger.info(f"   - Progreso total: {progress_percent:.1f}%")
        
        # Escribir resultados inmediatamente
        write_start = time.time()
        tiles_written = 0
        
        for tile_info in batch_results:
            if tile_info.processed_mask is not None:
                try:
                    # Extraer region central
                    offset_x = tile_info.output_x - tile_info.x_start
                    offset_y = tile_info.output_y - tile_info.y_start
                    
                    # Verificar offsets
                    if (0 <= offset_y < tile_info.processed_mask.shape[0] and
                        0 <= offset_x < tile_info.processed_mask.shape[1] and
                        offset_y + tile_info.output_height <= tile_info.processed_mask.shape[0] and
                        offset_x + tile_info.output_width <= tile_info.processed_mask.shape[1]):
                        
                        central_region = tile_info.processed_mask[
                            offset_y:offset_y + tile_info.output_height,
                            offset_x:offset_x + tile_info.output_width
                        ]
                        
                        # Escribir al TIFF
                        out_band.WriteArray(
                            central_region, 
                            tile_info.output_x, 
                            tile_info.output_y
                        )
                        
                        tiles_written += 1
                        processed_pixels += tile_info.output_width * tile_info.output_height
                        
                except Exception as e:
                    logger.error(f"   - Error escribiendo tile {tile_info.id}: {e}")
        
        write_time = time.time() - write_start
        logger.info(f"   - Tiles escritos: {tiles_written}/{len(batch_results)} en {write_time:.2f}s")
        
        # Liberar memoria del batch
        del batch_input
        for tile_info in batch_tiles:
            tile_info.tile_data = None
            tile_info.processed_mask = None
        
        gc.collect()
    
    # 9. Leer la mascara completa del TIFF para procesamiento final
    logger.info("\n" + "=" * 80)
    logger.info("GENERANDO ARCHIVOS FINALES")
    logger.info("=" * 80)
    
    try:
        # Asegurar que todo se haya escrito al disco
        out_band.FlushCache()
        
        # Reabrir el TIFF para leerlo completo
        out_ds_temp = None
        full_mask_clean = None
        
        try:
            # Cerrar y reabrir para leer
            out_ds = None
            out_ds_temp = gdal.Open(str(out_tif))
            if out_ds_temp:
                full_mask_clean = out_ds_temp.ReadAsArray()
                logger.info(f"Mascara leida del TIFF: {full_mask_clean.shape}")
            else:
                logger.error("No se pudo reabrir el TIFF para lectura")
                full_mask_clean = np.zeros((height, width), dtype=np.uint8)
        finally:
            if out_ds_temp:
                out_ds_temp = None
        
        # 10. Generar imagen balanced_argmax.png (CRÍTICO PARA EL APLICATIVO)
        balanced_argmax_png_path = output_dir / f'{base_name}_balanced_argmax.png'
        if full_mask_clean is not None:
            save_balanced_argmax_image(full_mask_clean, balanced_argmax_png_path, base_name)
        
        # 11. Guardar tambien la visualizacion original
        verification_path = output_dir / f'{base_name}_verificacion.png'
        if full_mask_clean is not None:
            save_colored_mask(full_mask_clean, verification_path)
        
        # Obtener estadisticas finales
        total_time = time.time() - total_start
        proc_stats = batch_processor.get_statistics()
        
        # 12. Generar reporte final
        if full_mask_clean is not None:
            generar_reporte_final(
                full_mask_clean, output_dir, base_name, width, height,
                total_time, total_tiles, processed_pixels, proc_stats,
                batch_times, read_times
            )
        
        # 13. Verificar archivos generados
        archivos_ok = verificar_archivos_salida(output_dir, base_name)
        
        # 14. Realizar limpieza final
        realizar_limpieza_final(dataset, out_ds, full_mask_clean, memory_monitor)
        
        # 15. Mensaje final de confirmacion
        logger.info("\n" + "=" * 80)
        logger.info("PROCESO COMPLETADO SATISFACTORIAMENTE")
        logger.info("=" * 80)
        
        if archivos_ok:
            logger.info("✓ Todos los archivos de salida se generaron correctamente")
            logger.info(f"✓ Directorio de salida: {output_dir}")
            logger.info("✓ Puede proceder con el siguiente paso en el flujo de trabajo")
        else:
            logger.warning("⚠ Algunos archivos no se generaron correctamente")
            logger.warning("  Revise los mensajes de error anteriores")
            
    except Exception as e:
        logger.error(f"Error en el procesamiento final: {e}")
        raise

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    exit_code = 0
    try:
        logger.info("Iniciando segmentacion semantica - MODO 100% RAM")
        logger.info(f"Parametros: tile_size={args.tile_size}, overlap={args.overlap}")
        logger.info(f"Batch size: {args.min_batch_size}-{args.max_batch_size}")
        logger.info(f"Margen seguridad memoria: {args.memory_safety_margin*100:.1f}%")
        logger.info(f"Tiles a precargar: {args.prefetch_tiles}")
        
        process_image_extreme()
        
    except KeyboardInterrupt:
        logger.warning("\n" + "!" * 70)
        logger.warning("PROCESO INTERRUMPIDO POR EL USUARIO")
        logger.warning("!" * 70)
        exit_code = 130
    except Exception as e:
        logger.error("\n" + "X" * 70)
        logger.error(f"ERROR CRITICO DURANTE LA EJECUCION: {e}", exc_info=True)
        logger.error("X" * 70)
        exit_code = 1
    finally:
        # Asegurar limpieza final incluso si hay errores
        try:
            # Limpiar cualquier recurso restante
            gc.collect()
        except:
            pass
    
    if exit_code == 0:
        logger.info("\n" + "✓" * 70)
        logger.info("PROGRAMA FINALIZADO CORRECTAMENTE")
        logger.info("✓" * 70)
    
    sys.exit(exit_code)
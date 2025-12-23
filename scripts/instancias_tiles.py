import os
import psutil
import tracemalloc
from osgeo import gdal, ogr, osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from PIL import Image
import onnxruntime as ort
from scipy.ndimage import binary_erosion
import skimage.morphology
import geopandas as gpd
import pandas as pd

# ======================================================================
# NOTA: Las rutas hardcoded han sido eliminadas para evitar conflictos
# Ahora se definen dinámicamente basadas en los argumentos
# ======================================================================

# ======================================================================
# FUNCIONES DE MONITOREO DE MEMORIA Y TILES
# ======================================================================
def get_memory_usage():
    """Obtiene el uso de memoria actual del proceso"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        'available_mb': psutil.virtual_memory().available / 1024 / 1024,
        'percent': psutil.virtual_memory().percent,
        'total_mb': psutil.virtual_memory().total / 1024 / 1024
    }

def print_memory_usage(label=""):
    """Imprime el uso de memoria actual"""
    mem = get_memory_usage()
    print(f"{label} MEMORIA - RSS: {mem['rss_mb']:.1f} MB, "
          f"Disponible: {mem['available_mb']:.1f} MB, "
          f"Uso: {mem['percent']:.1f}%")

def calculate_array_memory(array):
    """Calcula el tamaño en MB de un array numpy"""
    if array is None:
        return 0
    return array.nbytes / 1024 / 1024

def calculate_tile_size_info(tile_height, tile_width, bandas=3):
    """
    Calcula información detallada del tamaño de un tile
    """
    # Tamaño en píxeles
    total_pixels = tile_height * tile_width
    
    # Tamaño en MB para diferentes tipos de datos
    size_float32 = total_pixels * bandas * 4 / 1024 / 1024  # float32
    size_float64 = total_pixels * bandas * 8 / 1024 / 1024  # float64
    size_uint8 = total_pixels * bandas * 1 / 1024 / 1024    # uint8
    
    # Tamaño de máscara (1 banda)
    mask_float32 = total_pixels * 4 / 1024 / 1024
    
    return {
        'tile_pixels': total_pixels,
        'tile_dimensions': f"{tile_height}x{tile_width}",
        'size_float32_mb': size_float32,
        'size_float64_mb': size_float64,
        'size_uint8_mb': size_uint8,
        'mask_float32_mb': mask_float32,
        'total_3bands_float32_mb': size_float32 + mask_float32
    }

def print_tile_info(tile_height, tile_width, bandas=3, label=""):
    """Imprime información detallada sobre el tamaño del tile"""
    info = calculate_tile_size_info(tile_height, tile_width, bandas)
    
    print(f"\n{label} INFORMACIÓN DEL TILE:")
    print(f"  Dimensiones: {info['tile_dimensions']} píxeles")
    print(f"  Total píxeles: {info['tile_pixels']:,}")
    print(f"  Tamaño estimado:")
    print(f"    - float32 (3 bands): {info['size_float32_mb']:.2f} MB")
    print(f"    - float64 (3 bands): {info['size_float64_mb']:.2f} MB")
    print(f"    - uint8 (3 bands): {info['size_uint8_mb']:.2f} MB")
    print(f"    - Máscara float32: {info['mask_float32_mb']:.2f} MB")
    print(f"    - TOTAL (img + mask): {info['total_3bands_float32_mb']:.2f} MB")

def optimize_tile_size(max_memory_mb=4096, win_size=700, bandas=3, 
                       min_tile_size=500, max_tile_size=4000):
    """
    Optimiza automáticamente el tamaño del tile basado en la memoria disponible
    """
    mem = get_memory_usage()
    available_mb = mem['available_mb'] * 0.7  # Usar 70% de la memoria disponible
    
    # Limitar a max_memory_mb si se especifica
    if max_memory_mb:
        available_mb = min(available_mb, max_memory_mb)
    
    print(f"\nMemoria disponible para tiles: {available_mb:.1f} MB")
    
    # Calcular tamaño de tile óptimo
    # Consideramos: tile + máscara + ventana de procesamiento
    bytes_per_pixel = (bandas + 1) * 4  # +1 para máscara, float32=4 bytes
    tile_size_pixels = int(np.sqrt((available_mb * 1024 * 1024) / bytes_per_pixel))
    
    # Ajustar a límites razonables
    tile_size_pixels = max(min_tile_size, min(tile_size_pixels, max_tile_size))
    
    # Redondear a múltiplo de 256 para eficiencia
    tile_size_pixels = (tile_size_pixels // 256) * 256
    if tile_size_pixels < min_tile_size:
        tile_size_pixels = min_tile_size
    
    return tile_size_pixels

# Constantes para PIXELADO/JAGGED (como en .mat original)
CLASS_TO_SS = {"mauritia":-128, "euterpe":-96, "oenocarpus":-64}
CLASS_TO_CITYSCAPES = {"mauritia":15, "euterpe":25, "oenocarpus":35}
THRESHOLD = {"mauritia":3, "euterpe":1, "oenocarpus":2} 
MIN_SIZE = {"mauritia":500, "euterpe":400, "oenocarpus":200}
SELEM = {3: np.ones((3,3), dtype=bool), 1: np.ones((1,1), dtype=bool), 2: np.ones((1,1), dtype=bool)}
SELEN = {3: np.ones((36,3), dtype=bool), 1: np.ones((7,7), dtype=bool), 2: np.ones((3,3), dtype=bool)} 

nodata_value = -9999

def scale_image(image, flag=None, nodata_value=nodata_value):
    if flag is None:
        return image

def watershed_cut(depthImage, ssMask):
    resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)
    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ssCode = CLASS_TO_SS[semClass]
        ssMaskClass = (ssMask == ssCode)
        ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass
        ccLabels = skimage.morphology.label(ccImage)
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000)
        ccIDs = np.unique(ccLabels)[1:]
        for ccID in ccIDs:          
            ccIDMask = (ccLabels == ccID)
            ccIDMask = skimage.morphology.binary_erosion(ccIDMask, SELEM[THRESHOLD[semClass]])
            ccIDMask = binary_erosion(ccIDMask, SELEN[THRESHOLD[semClass]])
            resultImage[ccIDMask] = csCode
    return resultImage.astype(np.float32)

def process_instances_raster(raster):
    resultImage = np.zeros(shape=raster.shape, dtype=np.float32)
    ninstances={"mauritia":0, "euterpe":0, "oenocarpus":0}
    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ccImage = (raster == csCode)
        ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000)
        ccLabels = skimage.morphology.label(ccImage)
        ccIDs = np.unique(ccLabels)[1:]
        ninstances[semClass]=len(ccIDs)
        for ccID in ccIDs:          
            ccIDMask = (ccLabels == ccID)
            resultImage[ccIDMask] = csCode
    return resultImage.astype(np.float32), ninstances

# ======================================================================
# NUEVA FUNCIÓN: apply_instance_onnx_tiled_optimized
# Con monitoreo detallado del peso de cada tile
# ======================================================================
def apply_instance_onnx_tiled_optimized(feature_file_list, mask, roi, output_folder, session, 
                                        window_radius, internal_window_radius, 
                                        target_memory_mb=2048, make_tif=True, make_png=False):
    """
    Versión optimizada con monitoreo detallado del peso de cada tile
    """
    print("=" * 70)
    print("INICIANDO PROCESAMIENTO CON MONITOREO DETALLADO DE TILES")
    print("=" * 70)
    
    # Monitoreo inicial
    print_memory_usage("INICIO")
    
    image_path = feature_file_list[0]
    mask_path = mask[0]

    dataset = gdal.Open(image_path)
    datasetresponse = gdal.Open(mask_path)

    bandas = min(dataset.RasterCount, 3)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    
    print(f"\nDIMENSIONES IMAGEN: {width}x{height} píxeles, {bandas} bandas")
    print(f"RADIO VENTANA: {window_radius} px (tamaño: {window_radius*2}x{window_radius*2})")

    # Optimizar tamaño de tile automáticamente
    tile_size = optimize_tile_size(
        max_memory_mb=target_memory_mb,
        win_size=window_radius*2,
        bandas=bandas,
        min_tile_size=1024,
        max_tile_size=4000
    )
    
    # Calcular overlap (mínimo window_radius)
    overlap = max(window_radius, 100)
    
    print(f"\n" + "="*50)
    print("CONFIGURACIÓN OPTIMIZADA:")
    print(f"  - Tamaño de tile: {tile_size}x{tile_size} píxeles")
    print(f"  - Overlap: {overlap} píxeles")
    print(f"  - Memoria objetivo: {target_memory_mb} MB")
    
    # Mostrar información detallada del tamaño del tile
    print_tile_info(tile_size, tile_size, bandas, "TILE BASE")
    
    # Crear archivo de salida
    output = np.zeros((height, width), dtype=np.float32) + nodata_value
    output_size_mb = calculate_array_memory(output)
    print(f"\nArray de salida: {output.shape[1]}x{output.shape[0]} = {output_size_mb:.2f} MB")

    win_size = window_radius * 2

    # Calcular listas de ventanas GLOBALES (igual que tu código original)
    cr = [0, width]
    rr = [0, height]

    collist = [x for x in range(cr[0] + window_radius, cr[1] - window_radius, internal_window_radius * 2)]
    collist.append(cr[1] - window_radius)
    rowlist = [x for x in range(rr[0] + window_radius, rr[1] - window_radius, internal_window_radius * 2)]
    rowlist.append(rr[1] - window_radius)
    
    print(f"\nVENTANAS DE PROCESAMIENTO:")
    print(f"  - Columnas: {len(collist)} ventanas")
    print(f"  - Filas: {len(rowlist)} ventanas")
    print(f"  - Total ventanas: {len(collist) * len(rowlist)}")
    
    # Información de la ventana de procesamiento
    window_pixels = win_size * win_size
    window_size_mb = window_pixels * (bandas + 1) * 4 / 1024 / 1024  # +1 para máscara, float32
    print(f"\nVENTANA DE PROCESAMIENTO:")
    print(f"  - Dimensiones: {win_size}x{win_size} píxeles")
    print(f"  - Total píxeles: {window_pixels:,}")
    print(f"  - Tamaño estimado (img + mask): {window_size_mb:.2f} MB")

    # Estadísticas acumulativas
    tile_stats = {
        'total_tiles_processed': 0,
        'total_tile_memory_mb': 0,
        'max_tile_memory_mb': 0,
        'min_tile_memory_mb': float('inf'),
        'tile_sizes': []
    }
    
    # Crear un diccionario para cache de tiles
    tile_cache = {}
    cache_stats = {'hits': 0, 'misses': 0}
    
    # Iniciar monitoreo detallado
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Procesar en batches de columnas
    total_windows = len(collist) * len(rowlist)
    processed_windows = 0
    
    print(f"\n" + "="*50)
    print("INICIANDO PROCESAMIENTO POR TILES")
    print(f"{'='*50}")
    
    for col_idx, col in enumerate(collist):
        print(f"\n> COLUMNA {col_idx+1}/{len(collist)}: X={col}")
        print_memory_usage("  Antes de procesar columna")
        
        # Determinar rango de tiles para esta columna
        tile_start_col = max(0, col - window_radius - overlap)
        tile_end_col = min(width, col + window_radius + overlap)
        tile_width = tile_end_col - tile_start_col
        
        # Procesar cada fila en esta columna
        for row_idx, n in enumerate(rowlist):
            processed_windows += 1
            
            # Determinar rango de tiles para esta fila
            tile_start_row = max(0, n - window_radius - overlap)
            tile_end_row = min(height, n + window_radius + overlap)
            tile_height = tile_end_row - tile_start_row
            
            # Crear clave única para este tile
            tile_key = (tile_start_col, tile_start_row, tile_width, tile_height)
            
            # Monitorear tamaño de este tile específico
            tile_info = calculate_tile_size_info(tile_height, tile_width, bandas)
            current_tile_memory = tile_info['total_3bands_float32_mb']
            
            # Actualizar estadísticas
            tile_stats['total_tiles_processed'] += 1
            tile_stats['total_tile_memory_mb'] += current_tile_memory
            tile_stats['max_tile_memory_mb'] = max(tile_stats['max_tile_memory_mb'], current_tile_memory)
            tile_stats['min_tile_memory_mb'] = min(tile_stats['min_tile_memory_mb'], current_tile_memory)
            tile_stats['tile_sizes'].append({
                'dimensions': f"{tile_height}x{tile_width}",
                'memory_mb': current_tile_memory,
                'position': (col, n)
            })
            
            # Mostrar información del tile cada 5 ventanas o si es grande
            if (row_idx == 0) or (current_tile_memory > 100) or (processed_windows % 5 == 0):
                print(f"\n  Ventana {processed_windows}/{total_windows}:")
                print(f"    Posición: col={col}, row={n}")
                print(f"    Tile: {tile_height}x{tile_width} = {current_tile_memory:.2f} MB")
                print(f"    Memoria actual:")
                print_memory_usage("      ")
            
            # Verificar si el tile ya está en caché
            if tile_key in tile_cache:
                cache_stats['hits'] += 1
                d_tile, r_tile = tile_cache[tile_key]
            else:
                cache_stats['misses'] += 1
                
                # Leer tile de imagen
                d_tile = np.zeros((tile_height, tile_width, bandas), dtype=np.float32)
                for b in range(bandas):
                    band_data = dataset.GetRasterBand(b+1).ReadAsArray(
                        tile_start_col, tile_start_row, tile_width, tile_height)
                    d_tile[:,:,b] = band_data
                
                # Leer máscara del tile
                r_tile = datasetresponse.GetRasterBand(1).ReadAsArray(
                    tile_start_col, tile_start_row, tile_width, tile_height).astype(np.float32)
                
                # Preprocesamiento
                d_tile[np.isnan(d_tile)] = nodata_value
                d_tile[np.isinf(d_tile)] = nodata_value
                d_tile[d_tile == -9999] = nodata_value
                
                # Guardar en caché (limitado a 3 tiles para no usar mucha memoria)
                if len(tile_cache) < 3:
                    tile_cache[tile_key] = (d_tile.copy(), r_tile.copy())
                    cache_size_mb = calculate_array_memory(d_tile) + calculate_array_memory(r_tile)
                    print(f"    > Tile guardado en cache: {cache_size_mb:.2f} MB")
            
            # Verificar si la ventana completa está dentro del tile
            col_in_tile = col - tile_start_col
            row_in_tile = n - tile_start_row
            
            col_start = col_in_tile - window_radius
            col_end = col_start + win_size
            row_start = row_in_tile - window_radius
            row_end = row_start + win_size
            
            if (col_start >= 0 and col_end <= tile_width and 
                row_start >= 0 and row_end <= tile_height):
                
                # Extraer ventana específica
                d = d_tile[row_start:row_end, col_start:col_end, :].copy()
                r = r_tile[row_start:row_end, col_start:col_end].copy()
                
                window_memory = calculate_array_memory(d) + calculate_array_memory(r)
                
                if d.shape[0] == win_size and d.shape[1] == win_size:
                    # Procesamiento IDÉNTICO a tu código original
                    ssBatch = (r > 0).astype(np.float32)
                    ssMaskBatch = np.zeros_like(r, dtype=np.float32)
                    
                    ssMaskBatch[r == 1] = CLASS_TO_SS["mauritia"]
                    ssMaskBatch[r == 2] = CLASS_TO_SS["euterpe"]
                    ssMaskBatch[r == 3] = CLASS_TO_SS["oenocarpus"]
                    
                    d = scale_image(d)
                    img_j = d * ssBatch[..., np.newaxis]
                    input_j = np.concatenate([img_j, ssMaskBatch[..., np.newaxis]], axis=-1).astype(np.float32)
                    ss_batch_input = ssBatch.astype(np.float32)
                    
                    # Monitorear memoria antes de inferencia
                    if window_memory > 50:  # Solo mostrar para ventanas grandes
                        print(f"    > Procesando ventana: {window_memory:.2f} MB")
                    
                    # Inferencia ONNX
                    outputs = session.run(None, {
                        input_names[0]: input_j[np.newaxis, ...],
                        input_names[1]: ss_batch_input[np.newaxis, ...]
                    })
                    tmp_output = outputs[0][0].astype(np.uint8)
                    
                    # Watershed cut
                    outputImage = watershed_cut(tmp_output, ssMaskBatch)
                    
                    if internal_window_radius < window_radius:
                        mm = int(np.rint(window_radius - internal_window_radius))
                        outputImage = outputImage[mm:-mm, mm:-mm]
                    
                    # Escribir resultado
                    out_row_start = n - internal_window_radius
                    out_row_end = out_row_start + outputImage.shape[0]
                    out_col_start = col - internal_window_radius
                    out_col_end = out_col_start + outputImage.shape[1]
                    
                    output[out_row_start:out_row_end, 
                           out_col_start:out_col_end] = outputImage
            
            # Mostrar progreso cada 10 ventanas
            if (processed_windows % 10 == 0) and (processed_windows > 0):
                progress = (processed_windows / total_windows) * 100
                print(f"\n   Progreso: {progress:.1f}% ({processed_windows}/{total_windows} ventanas)")
                print_memory_usage("    Estado actual")
        
        # Limpiar caché periódicamente
        if col_idx % 2 == 0 and len(tile_cache) > 0:
            tile_cache.clear()
            import gc
            gc.collect()
            print(f"\n  [CACHE] Cache limpiado, memoria liberada")
            print_memory_usage("    Después de limpiar cache")
    
    # Finalizar monitoreo
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print(f"\n" + "="*70)
    print("ESTADÍSTICAS DETALLADAS DE TILES")
    print(f"{'='*70}")
    
    # Calcular estadísticas finales
    if tile_stats['total_tiles_processed'] > 0:
        avg_tile_memory = tile_stats['total_tile_memory_mb'] / tile_stats['total_tiles_processed']
        
        print(f"\n RESUMEN DE TILES PROCESADOS:")
        print(f"  - Total tiles procesados: {tile_stats['total_tiles_processed']}")
        print(f"  - Memoria total tiles: {tile_stats['total_tile_memory_mb']:.1f} MB")
        print(f"  - Memoria promedio por tile: {avg_tile_memory:.2f} MB")
        print(f"  - Tile más grande: {tile_stats['max_tile_memory_mb']:.2f} MB")
        print(f"  - Tile más pequeño: {tile_stats['min_tile_memory_mb']:.2f} MB")
        
        # Encontrar los 3 tiles más grandes
        tile_sizes_sorted = sorted(tile_stats['tile_sizes'], key=lambda x: x['memory_mb'], reverse=True)
        print(f"\n TOP 3 TILES MÁS GRANDES:")
        for i, tile in enumerate(tile_sizes_sorted[:3]):
            print(f"  {i+1}. {tile['dimensions']} = {tile['memory_mb']:.2f} MB (pos: {tile['position']})")
    
    print(f"\n ESTADÍSTICAS DE CACHE:")
    print(f"  - Cache hits: {cache_stats['hits']}")
    print(f"  - Cache misses: {cache_stats['misses']}")
    total_access = cache_stats['hits'] + cache_stats['misses']
    if total_access > 0:
        cache_efficiency = (cache_stats['hits'] / total_access) * 100
        print(f"  - Eficiencia del cache: {cache_efficiency:.1f}%")
    
    print(f"\n ESTADÍSTICAS DE MEMORIA (tracemalloc):")
    for stat in top_stats[:5]:
        print(f"  {stat}")
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\n USO DE MEMORIA FINAL:")
    print(f"  - Uso máximo (peak): {peak / 10**6:.1f} MB")
    print(f"  - Uso final (current): {current / 10**6:.1f} MB")
    print_memory_usage("FINAL")

    # Aplicar process_instances_raster al resultado COMPLETO
    print(f"\n" + "="*50)
    print("APLICANDO POST-PROCESAMIENTO DE INSTANCIAS...")
    print(f"{'='*50}")
    
    output, quantification = process_instances_raster(output)
    output_size_final = calculate_array_memory(output)
    print(f"[OK] Post-procesamiento completado")
    print(f"[OK] Tamaño del array final: {output_size_final:.2f} MB")

    # Guardar TIFF
    name_saved_final = os.path.basename(image_path).split('.tif')[0] + '__predicted.tif'
    out_path = os.path.join(output_folder, name_saved_final)
    
    if make_tif:
        print(f"\n[DISCO] GUARDANDO RESULTADO...")
        print(f"  Ruta: {out_path}")
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(out_path, output.shape[1], output.shape[0], 1, gdal.GDT_Float32)
        out_ds.GetRasterBand(1).WriteArray(output)
        out_ds.SetGeoTransform(dataset.GetGeoTransform())
        out_ds.SetProjection(dataset.GetProjection())
        out_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
        out_ds.FlushCache()
        out_ds = None
        print("[OK] Archivo guardado exitosamente")

    mau = quantification['mauritia']
    eut = quantification['euterpe']
    oeno = quantification['oenocarpus']

    return name_saved_final, mau, eut, oeno, output

# ======================================================================
# EJECUCIÓN PRINCIPAL (CON ARGPARSE)
# ======================================================================
if __name__ == "__main__":
    import argparse
    
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Conteo de instancias por tiles')
    parser.add_argument('image_path', help='Ruta a la imagen original')
    parser.add_argument('mask_path', help='Ruta a la máscara de segmentación')
    parser.add_argument('--target_memory_mb', type=int, default=4096)
    parser.add_argument('--window_radius', type=int, default=350)
    parser.add_argument('--output_dir', help='Directorio de salida', default='output')
    parser.add_argument('--model_path', help='Ruta al modelo', default='models/model_converted.onnx')

    args = parser.parse_args()

    # Asignar variables basadas en argumentos (NO HARDCODED)
    INPUT_RASTER = args.image_path
    MASK_CLAS = args.mask_path
    output_folder = args.output_dir
    window_radius = args.window_radius
    internal_window_radius = int(round(window_radius * 0.75))
    
    # Construir rutas de salida dinámicamente
    base_name = os.path.basename(INPUT_RASTER).replace('.tif', '')
    OUTPUT_RASTER = os.path.join(output_folder, f"{base_name}_predicted.tif")
    OUTPUT_IMAGE = os.path.join(output_folder, f"{base_name}_predicted.png")
    BALANCED_ARGMAX_IMAGE = os.path.join(output_folder, f"{base_name}_balanced_argmax.png")
    
    # Configurar parámetros para la segmentación de instancias
    feature_file_list = [INPUT_RASTER]
    mask = [MASK_CLAS]
    roi = []
    
    # Verificar que los archivos de entrada existan
    def verificar_archivos():
        archivos = {
            "Imagen RGB": INPUT_RASTER,
            "Máscara": MASK_CLAS
        }
        
        for nombre, ruta in archivos.items():
            if not os.path.exists(ruta):
                print(f"ERROR: {nombre} no encontrado en: {ruta}")
                directorio = os.path.dirname(ruta)
                if os.path.exists(directorio):
                    archivos_similares = [f for f in os.listdir(directorio) if base_name in f]
                    if archivos_similares:
                        print(f"Archivos similares encontrados en {directorio}:")
                        for archivo in archivos_similares:
                            print(f"  - {archivo}")
                return False
            else:
                print(f"[OK] {nombre} encontrado: {ruta}")
        
        salidas = {
            "Salida ráster": os.path.dirname(OUTPUT_RASTER),
            "Salida imagen": os.path.dirname(OUTPUT_IMAGE),
            "Salida balanced_argmax": os.path.dirname(BALANCED_ARGMAX_IMAGE)
        }
        
        for nombre, directorio in salidas.items():
            if not os.path.exists(directorio):
                print(f"Creando directorio: {directorio}")
                os.makedirs(directorio, exist_ok=True)
        
        return True
    
    # Ejecutar verificación
    if not verificar_archivos():
        print("Error: No se pueden encontrar todos los archivos necesarios.")
        exit(1)
    
    # Cargar el modelo ONNX
    model_path = args.model_path
    session = ort.InferenceSession(model_path)
    input_names = [inp.name for inp in session.get_inputs()]
    print("Inputs del modelo ONNX:", input_names)
    
    # Monitoreo inicial del sistema
    print("\n" + "="*70)
    print("ANÁLISIS INICIAL DEL SISTEMA")
    print("="*70)

    mem_info = psutil.virtual_memory()
    print(f"[INFO] INFORMACIÓN DEL SISTEMA:")
    print(f"  - Memoria total: {mem_info.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"  - Memoria disponible: {mem_info.available / 1024 / 1024 / 1024:.1f} GB")
    print(f"  - Porcentaje de uso: {mem_info.percent}%")
    print(f"  - Procesadores lógicos: {psutil.cpu_count()}")
    print(f"  - Procesadores físicos: {psutil.cpu_count(logical=False)}")

    # Aplicar segmentación de instancias CON MONITOREO DETALLADO
    print("\n" + "="*70)
    print("INICIANDO PROCESAMIENTO DE IMAGEN")
    print("="*70)

    name_saved_final, mau, eut, oeno, output_array = apply_instance_onnx_tiled_optimized(
        feature_file_list,
        mask,
        roi,
        output_folder,
        session,
        window_radius,
        internal_window_radius,
        target_memory_mb=args.target_memory_mb,
        make_tif=True,
        make_png=False
    )

    # Renombrar el archivo generado con manejo de errores
    out_imag = os.path.join(output_folder, name_saved_final)
    
    # CORRECCIÓN: Verificar si existe antes de renombrar
    if os.path.exists(OUTPUT_RASTER):
        try:
            os.remove(OUTPUT_RASTER)
            print(f"[INFO] Archivo anterior eliminado: {OUTPUT_RASTER}")
        except PermissionError:
            print(f"[ERROR] No se pudo eliminar el archivo existente: {OUTPUT_RASTER}. Está en uso.")
            # Intentar con un nombre alternativo si falla
            base, ext = os.path.splitext(OUTPUT_RASTER)
            OUTPUT_RASTER = f"{base}_new{ext}"

    # Ahora sí renombrar
    try:
        os.rename(out_imag, OUTPUT_RASTER)
        print(f"[EXITO] Archivo renombrado a: {OUTPUT_RASTER}")
    except OSError as e:
        print(f"[ERROR] Fallo al renombrar: {e}")
        # Si falla, mantener el nombre temporal
        print(f"[INFO] Se mantiene el nombre temporal: {out_imag}")
        OUTPUT_RASTER = out_imag

    # Imprimir informe de conteo
    print("\n" + "="*70)
    print("INFORME FINAL DE SEGMENTACIÓN")
    print("="*70)
    print(f"[ARCHIVOS] ARCHIVOS PROCESADOS:")
    print(f"  - Imagen Original: {INPUT_RASTER}")
    print(f"  - Máscara Clasificada: {MASK_CLAS}")
    print(f"  - Ráster de Instancias: {OUTPUT_RASTER}")

    print(f"\n[RESULTADOS] RESULTADOS DE DETECCIÓN:")
    print(f"  Mauritia flexuosa:     {mau:4d} palmeras {'[SI]' if mau > 0 else '[NO]'}")
    print(f"  Euterpe precatoria:    {eut:4d} palmeras {'[SI]' if eut > 0 else '[NO]'}")
    print(f"  Oenocarpus bataua:     {oeno:4d} palmeras {'[SI]' if oeno > 0 else '[NO]'}")
    print(f"  TOTAL DETECTADAS:     {mau + eut + oeno:4d} palmeras")

    # Verificar el archivo de salida
    if os.path.exists(OUTPUT_RASTER):
        file_size = os.path.getsize(OUTPUT_RASTER) / 1024 / 1024
        print(f"\n[DISCO] ARCHIVO DE SALIDA:")
        print(f"  - Ruta: {OUTPUT_RASTER}")
        print(f"  - Tamaño en disco: {file_size:.2f} MB")
    else:
        print(f"\n[ERROR] ERROR: No se pudo crear el archivo de salida")

    print("="*70)

    # MODIFICACIÓN CRÍTICA: Solo generar la imagen de conteo, NO sobrescribir balanced_argmax
    print(f"\n[IMG] GENERANDO IMAGEN DE VISUALIZACIÓN DE CONTEO...")
    
    # Usar el array en memoria en lugar de leer del archivo
    imagen_array = output_array

    unique_values = np.unique(imagen_array)
    print(f"VALORES ÚNICOS EN EL RÁSTER: {unique_values}")

    rescaled_image = np.interp(imagen_array, (0, 35), (0, 255)).astype(np.uint8)
    rescaled_values = np.unique(rescaled_image)
    print(f"VALORES ÚNICOS REESCALADOS: {rescaled_values}")

    colors = [
        "#000000",  # Negro: Fondo
        "#ff7f00",  # Naranja: Mauritia flexuosa
        "#08F6EB",  # Cian: Euterpe precatoria
        "#E008F6"   # Púrpura: Oenocarpus bataua
    ]
    custom_cmap = ListedColormap(colors)

    bounds = [0, 15, 25, 35, 40]
    norm = BoundaryNorm(bounds, custom_cmap.N)

    # CAMBIO CRÍTICO: Usar float32 en lugar de float64 para ahorrar memoria
    plt.figure(figsize=(12, 8), dpi=100)  # Reducir DPI para menos memoria
    
    # Convertir a float32 antes de mostrar
    imagen_array_float32 = imagen_array.astype(np.float32)
    
    try:
        # Intentar generar la imagen con configuración de memoria baja
        plt.imshow(imagen_array_float32, cmap=custom_cmap, norm=norm, interpolation='nearest')
        plt.axis('off')
        
        # Liberar memoria antes de guardar
        import gc
        gc.collect()
        
        # Guardar SOLAMENTE como predicted.png (Vista de Conteo)
        plt.savefig(OUTPUT_IMAGE, bbox_inches='tight', pad_inches=0, dpi=100)  # Reducir DPI
        plt.close('all')  # Cerrar todas las figuras
        
        # Verificar archivo
        if os.path.exists(OUTPUT_IMAGE):
            img_size = os.path.getsize(OUTPUT_IMAGE) / 1024 / 1024
            print(f"[OK] Imagen guardada (predicted): {OUTPUT_IMAGE} ({img_size:.2f} MB)")
        
    except MemoryError as e:
        print(f"[ERROR] Memoria insuficiente para generar imagen PNG: {e}")
        print("[INFO] Generando imagen PNG con calidad reducida...")
        
        # Alternativa: usar OpenCV para generar la imagen (usa menos memoria)
        try:
            import cv2
            
            # Mapear valores a colores BGR
            color_map = {
                0: [0, 0, 0],        # Negro
                15: [0, 127, 255],   # Naranja (BGR)
                25: [235, 246, 8],   # Cian (BGR)
                35: [246, 8, 224]    # Púrpura (BGR)
            }
            
            # Crear imagen RGB
            h, w = imagen_array.shape
            rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
            
            for value, color in color_map.items():
                mask = (imagen_array == value)
                rgb_image[mask] = color
            
            # Guardar con OpenCV
            cv2.imwrite(OUTPUT_IMAGE, rgb_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"[BACKUP] Imagen guardada con OpenCV: {OUTPUT_IMAGE}")
            
            if os.path.exists(OUTPUT_IMAGE):
                img_size = os.path.getsize(OUTPUT_IMAGE) / 1024 / 1024
                print(f"[OK] Imagen guardada (predicted): {OUTPUT_IMAGE} ({img_size:.2f} MB)")
                
        except Exception as e2:
            print(f"[ERROR] Fallo en alternativa OpenCV: {e2}")
            print("[WARNING] No se pudo generar imagen PNG")

    # Vectorización del ráster de instancias
    print("\n" + "="*70)
    print("VECTORIZACIÓN DE RESULTADOS")
    print("="*70)

    list_in_path = [OUTPUT_RASTER]

    for in_path in list_in_path:
        if not os.path.exists(in_path):
            print("ERROR: Archivo no encontrado")
            print(f"   Ruta: {in_path}")
            continue
        
        print(f"PROCESANDO: {in_path}")

        out_POL_path = os.path.join(in_path.split('.tif')[0] + '_poly.gpkg')
        out_CEN_path = os.path.join(in_path.split('.tif')[0] + '_centers.gpkg')
        out_CSV_path = os.path.join(in_path.split('.tif')[0] + '_atributos.csv')

        src_ds = gdal.Open(in_path)
        srcband = src_ds.GetRasterBand(1)
        dst_layername = 'palms_instances'
        drv = ogr.GetDriverByName("GPKG")
        dst_ds = drv.CreateDataSource(out_POL_path)
        prj = src_ds.GetProjection()
        sp_ref = osr.SpatialReference(wkt=prj)
        dst_layer = dst_ds.CreateLayer(dst_layername, srs=sp_ref)
        fld = ogr.FieldDefn("ID", ogr.OFTInteger)
        dst_layer.CreateField(fld)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex("ID")
        gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)
        del src_ds
        del dst_ds

        gdf = gpd.read_file(out_POL_path)
        gdf = gdf[gdf["ID"] > 0]

        c1, c2, c3 = 0, 0, 0
        ca1, ca2, ca3 = 0, 0, 0

        def generateCentroidColumns(item):
            global c1, c2, c3, ca1, ca2, ca3
            ESPECIE = 'Mauritia flexuosa'
            geom = item.geometry
            if item['ID'] == 15:
                ESPECIE = 'Mauritia flexuosa'
                c1 += 1
                ca1 += geom.area
            elif item['ID'] == 25:
                ESPECIE = 'Euterpe precatoria'
                c2 += 1
                ca2 += geom.area
            elif item['ID'] == 35:
                ESPECIE = 'Oenocarpus bataua'
                c3 += 1
                ca3 += geom.area
            areacopa = geom.area
            cx = geom.centroid.x
            cy = geom.centroid.y
            clase = item['ID']
            return pd.Series([clase, ESPECIE, areacopa, cx, cy])

        gdf[['ID', 'ESPECIE', 'ÁREA(m2)', 'UTM(ESTE)', 'UTM(NORTE)']] = gdf.apply(generateCentroidColumns, axis=1)
        
        print(f"\n RESUMEN DE VECTORIZACIÓN:")
        print(f"  Mauritia flexuosa (ID 15): {c1:4d} polígonos")
        print(f"  Euterpe precatoria (ID 25): {c2:4d} polígonos")
        print(f"  Oenocarpus bataua (ID 35): {c3:4d} polígonos")
        print(f"  TOTAL POLÍGONOS:           {c1 + c2 + c3:4d}")

        gdf[['ID', 'ESPECIE', 'ÁREA(m2)', 'UTM(ESTE)', 'UTM(NORTE)']].to_csv(out_CSV_path, index=False)
        csv_size = os.path.getsize(out_CSV_path) / 1024 if os.path.exists(out_CSV_path) else 0
        print(f"[OK] Atributos CSV: {out_CSV_path} ({csv_size:.1f} KB)")

        def generarcentroid(item):
            return pd.Series([item.ESPECIE, item.geometry.centroid])
        dfcentroid = gdf.apply(generarcentroid, axis=1).set_axis(['ESPECIE', 'geometry'], axis=1)
        dfcentroid = gpd.GeoDataFrame(dfcentroid, geometry='geometry')

        gdf.to_file(out_POL_path)
        poly_size = os.path.getsize(out_POL_path) / 1024 / 1024 if os.path.exists(out_POL_path) else 0
        print(f"[OK] Polígonos GPKG: {out_POL_path} ({poly_size:.2f} MB)")

        dfcentroid.to_file(out_CEN_path)
        center_size = os.path.getsize(out_CEN_path) / 1024 / 1024 if os.path.exists(out_CEN_path) else 0
        print(f"[OK] Centroides GPKG: {out_CEN_path} ({center_size:.2f} MB)")

    print("\n" + "="*70)
    print(" PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
    print("="*70)
    print(f" Hora de finalización: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duración aproximada: Verificar en el registro de tiempo del sistema")
    print(f"  Archivos generados:")
    print(f"    - {OUTPUT_RASTER} (Ráster de instancias)")
    print(f"    - {OUTPUT_IMAGE} (Imagen de conteo)")
    # Nota: balanced_argmax.png ya fue generado por process_with_tiles.py y NO fue sobrescrito
    print(f"    - {out_POL_path} (Polígonos vectoriales)")
    print(f"    - {out_CEN_path} (Centroides vectoriales)")
    print(f"    - {out_CSV_path} (Atributos en CSV)")
    print("="*70)
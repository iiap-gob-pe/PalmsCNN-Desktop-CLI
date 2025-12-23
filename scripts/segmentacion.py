import sys
import os
import argparse
import numpy as np
from osgeo import gdal
import onnxruntime as rt
import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tempfile
import psutil  # <-- NUEVO IMPORT para monitorear RAM
import time    # <-- NUEVO IMPORT para timestamp

# Habilitar excepciones de GDAL
gdal.UseExceptions()

# Configuración de argumentos (TU CÓDIGO ORIGINAL)
parser = argparse.ArgumentParser(description='Segmentación de palmeras')
parser.add_argument('image_path', help='Ruta a la imagen a procesar')
parser.add_argument('--model', help='Ruta al modelo ONNX', default='models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx')
parser.add_argument('--output', help='Directorio de salida', default='output')
parser.add_argument('--window_radius', type=int, default=256)
parser.add_argument('--internal_window_radius', type=int)
parser.add_argument('--scaling', choices=['none', 'mean_std', 'normalize'], default='normalize')

args = parser.parse_args()

# Parámetros (TU CÓDIGO ORIGINAL)
feature_file_list = [args.image_path]
model_onnx_path = args.model
output_folder = args.output
window_radius = args.window_radius
internal_window_radius = args.internal_window_radius if args.internal_window_radius else int(round(window_radius * 0.75))
application_name = 'balanced'
nodata_value = 0 

print("=== PARAMETROS DE EJECUCION (OPTIMIZADO MEMMAP) ===", flush=True)
print(f"Imagen: {args.image_path}", flush=True)
print(f"Modelo: {model_onnx_path}", flush=True)
print(f"Salida: {output_folder}", flush=True)
print(f"Ventana: {window_radius*2}x{window_radius*2}", flush=True)
print(f"Ventana interna: {internal_window_radius*2}x{internal_window_radius*2}", flush=True)
print(f"Escalado: {args.scaling}", flush=True)
print("================================================", flush=True)

def rint(num): return int(round(num))

def get_memory_usage():
    """Obtiene el uso de memoria actual del proceso en MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convertir a MB

def print_memory_status(label=""):
    """Imprime el estado de memoria con timestamp"""
    current_mem = get_memory_usage()
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {label} RAM: {current_mem:.1f} MB", flush=True)
    return current_mem

def diagnostic_image_analysis(img_path):
    dataset = gdal.Open(img_path)
    if not dataset: return None, None
    info = {'size': (dataset.RasterXSize, dataset.RasterYSize), 'bands': dataset.RasterCount}
    band_stats = []
    for i in range(info['bands']):
        band = dataset.GetRasterBand(i+1)
        stats = band.GetStatistics(True, True)
        band_stats.append(stats)
    return info, band_stats

# MODIFICACIÓN CRÍTICA AQUÍ: En lugar de crear un array gigante en RAM, usamos disco.
def load_and_preprocess_to_memmap(img_path, output_dir, scaling='normalize'):
    dataset = gdal.Open(img_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = min(3, dataset.RasterCount)
    
    # Monitorear RAM al inicio
    initial_mem = print_memory_status("Inicio carga imagen")
    
    # Archivo temporal para la imagen procesada
    temp_img_file = os.path.join(output_dir, 'temp_input_float.dat')
    # Creamos array en DISCO
    img_memmap = np.memmap(temp_img_file, dtype=np.float32, mode='w+', shape=(height, width, bands))
    
    print(f"Pre-procesando imagen a disco ({width}x{height})...", flush=True)
    
    # Leemos y procesamos por bloques para no saturar RAM
    block_size = 2048
    block_count = 0
    
    for y in range(0, height, block_size):
        rows = min(block_size, height - y)
        # Leer bloque original
        data_block = dataset.ReadAsArray(0, y, width, rows) # (Bands, Rows, Cols)
        
        if len(data_block.shape) == 2: data_block = data_block[np.newaxis, ...]
        data_block = np.transpose(data_block[:bands, ...], (1, 2, 0)).astype(np.float32)
        
        # Aplicar TU lógica de escalado al bloque
        if scaling == 'normalize':
            data_block = data_block / 127.5 - 1.0 # Aproximación segura y rápida
        
        # Escribir al memmap (disco)
        img_memmap[y:y+rows, :, :] = data_block
        
        block_count += 1
        if block_count % 5 == 0:  # Mostrar RAM cada 5 bloques
            print_memory_status(f"Bloque {block_count}")
    
    final_mem = print_memory_status("Fin carga imagen")
    print(f"Memoria usada durante carga: {final_mem - initial_mem:.1f} MB", flush=True)
    
    return img_memmap, dataset, temp_img_file

def postprocess_mask(mask, min_region_size=30):
    try:
        from skimage import morphology
        cleaned_mask = mask.copy()
        for class_id in [1, 2, 3]:
            class_mask = mask == class_id
            if np.sum(class_mask) > 0:
                class_mask_cleaned = morphology.remove_small_objects(class_mask, min_size=min_region_size)
                class_mask_cleaned = morphology.remove_small_holes(class_mask_cleaned, area_threshold=min_region_size)
                cleaned_mask[class_mask & ~class_mask_cleaned] = 0
                cleaned_mask[class_mask_cleaned] = class_id
        return cleaned_mask
    except:
        print("Advertencia: No se pudo postprocesar máscara (skimage no disponible)", flush=True)
        return mask

def save_tiff_mask(mask, output_path, reference_dataset):
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
    out_dataset.SetGeoTransform(reference_dataset.GetGeoTransform())
    out_dataset.SetProjection(reference_dataset.GetProjection())
    out_dataset.GetRasterBand(1).WriteArray(mask)
    out_dataset.FlushCache()
    out_dataset = None
    print(f"TIFF guardado: {output_path}", flush=True)

def save_png_mask(mask, output_path):
    colors = [[0, 0, 0], [255, 127, 0], [8, 246, 235], [224, 8, 246]]
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors): 
        rgb_mask[mask == i] = color
    
    # Downsample para guardar PNG si es gigante (evita crash matplotlib)
    h, w = mask.shape
    scale = max(1, max(h, w) // 3000)
    if scale > 1:
        rgb_mask = rgb_mask[::scale, ::scale, :]
        print(f"Reduciendo PNG a {rgb_mask.shape[1]}x{rgb_mask.shape[0]} para visualización", flush=True)
        
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb_mask)
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor='white')
    plt.close()
    print(f"PNG guardado: {output_path}", flush=True)

# FUNCIÓN PRINCIPAL ADAPTADA A MEMMAP
def apply_semantic_segmentation_onnx(input_file_list, output_folder, application_name, model_onnx_path,
                                     window_radius, internal_window_radius, make_tif=True, make_png=True, scaling='none'):
    os.makedirs(output_folder, exist_ok=True)
    
    # Monitorear RAM total del sistema
    total_system_mem = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    available_system_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
    print(f"\nSistema - Memoria total: {total_system_mem:.1f} GB, Disponible: {available_system_mem:.1f} GB", flush=True)
    
    # Cargar Modelo
    sess_opts = rt.SessionOptions()
    sess_opts.enable_cpu_mem_arena = False # IMPORTANTE PARA MEMORIA
    session = rt.InferenceSession(model_onnx_path, sess_opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Modelo cargado: {model_onnx_path}", flush=True)
    
    # Monitorear RAM después de cargar modelo
    model_load_mem = print_memory_status("Después de cargar modelo")

    for img_path in input_file_list:
        print(f"\n{'='*60}", flush=True)
        print(f"Procesando imagen: {img_path}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Monitorear RAM antes de procesar imagen
        start_mem = print_memory_status("Inicio procesamiento imagen")
        
        # 1. PREPROCESAMIENTO A DISCO (MEMMAP)
        img, dataset, temp_img_file = load_and_preprocess_to_memmap(img_path, output_folder, scaling)
        height, width = img.shape[:2]
        print(f"Imagen dimensiones: {width}x{height}", flush=True)
        
        # 2. OUTPUT A DISCO (MEMMAP)
        temp_out_file = os.path.join(output_folder, 'temp_output_mask.dat')
        output_mask = np.memmap(temp_out_file, dtype=np.uint8, mode='w+', shape=(height, width))
        output_mask[:] = nodata_value
        print(f"Memmap de salida creado: {temp_out_file}", flush=True)

        # Sistema de ventanas (Tu lógica original)
        step = internal_window_radius * 2
        collist = list(range(window_radius, width - window_radius, step))
        if not collist: 
            collist = [window_radius]
            print("Advertencia: Imagen pequeña, usando única ventana", flush=True)
        elif collist[-1] < width - window_radius: 
            collist.append(width - window_radius)
        
        rowlist = list(range(window_radius, height - window_radius, step))
        if not rowlist: 
            rowlist = [window_radius]
        elif rowlist[-1] < height - window_radius: 
            rowlist.append(height - window_radius)

        print(f"Procesando {len(rowlist)}x{len(collist)} ventanas...", flush=True)
        print(f"Total ventanas: {len(rowlist) * len(collist)}", flush=True)
        
        # Estadísticas de memoria por columna
        memory_by_column = []

        # Contador de progreso
        total_windows = len(collist)
        processed_windows = 0
        
        for col_idx, col in enumerate(collist):
            # Monitorear RAM antes de procesar columna
            col_start_mem = print_memory_status(f"Antes columna {col_idx+1}")
            
            windows = []
            rows_for_col = []
            
            print(f"\n[Columna {col_idx+1}/{total_windows}] (x={col})...", flush=True)
            
            for row in rowlist:
                # Leer ventana (desde el memmap en disco, muy rápido y poca RAM)
                window = img[row - window_radius:row + window_radius, 
                           col - window_radius:col + window_radius].copy()
                
                if window.shape[0] == window_radius * 2 and window.shape[1] == window_radius * 2:
                    windows.append(window)
                    rows_for_col.append(row)
            
            if windows:
                windows = np.stack(windows) # Esto es lo único que va a RAM (batch pequeño)
                print(f"  Procesando {len(windows)} ventanas en esta columna...", flush=True)
                
                # Monitorear RAM después de crear batch
                batch_mem = print_memory_status("  Después de crear batch")
                
                # Inferencia
                pred = session.run([output_name], {input_name: windows})[0]
                
                # Monitorear RAM después de inferencia
                inference_mem = print_memory_status("  Después de inferencia")
                
                for i, row in enumerate(rows_for_col):
                    pred_mask = np.argmax(pred[i], axis=-1).astype(np.uint8)
                    
                    if internal_window_radius < window_radius:
                        mm = rint(window_radius - internal_window_radius)
                        pred_mask = pred_mask[mm:-mm, mm:-mm]
                    
                    # Escribir resultado a disco (memmap)
                    output_mask[row - internal_window_radius:row + internal_window_radius,
                                col - internal_window_radius:col + internal_window_radius] = pred_mask
                
                processed_windows += len(windows)
                
                # Monitorear RAM después de procesar columna
                col_end_mem = print_memory_status(f"Después columna {col_idx+1}")
                mem_used_col = col_end_mem - col_start_mem
                memory_by_column.append(mem_used_col)
                
                print(f"  Progreso: {processed_windows}/{len(rowlist)*len(collist)} ventanas procesadas", flush=True)
                print(f"  RAM usada en columna: {mem_used_col:+.1f} MB (total: {col_end_mem:.1f} MB)", flush=True)
                
                # Liberar memoria del batch
                del windows, pred
                
                # Forzar garbage collection cada 5 columnas
                if (col_idx + 1) % 5 == 0:
                    import gc
                    gc.collect()
                    gc_mem = print_memory_status("  Después de GC")
                    print(f"  Garbage Collection liberó: {col_end_mem - gc_mem:.1f} MB", flush=True)
        
        # Estadísticas de memoria
        if memory_by_column:
            avg_mem_per_col = sum(memory_by_column) / len(memory_by_column)
            max_mem_col = max(memory_by_column)
            min_mem_col = min(memory_by_column)
            
            print(f"\n{'='*60}", flush=True)
            print(f"ESTADÍSTICAS DE MEMORIA POR COLUMNA:", flush=True)
            print(f"  Máximo uso por columna: {max_mem_col:.1f} MB", flush=True)
            print(f"  Mínimo uso por columna: {min_mem_col:.1f} MB", flush=True)
            print(f"  Promedio uso por columna: {avg_mem_per_col:.1f} MB", flush=True)
            print(f"  Total columnas procesadas: {len(memory_by_column)}", flush=True)
            print(f"{'='*60}", flush=True)
        
        print("\nInferencia completada. Postprocesando máscara...", flush=True)
        
        # Postprocesamiento
        post_start_mem = print_memory_status("Antes postprocesamiento")
        full_mask = np.array(output_mask) 
        output_mask_processed = postprocess_mask(full_mask, min_region_size=20)
        post_end_mem = print_memory_status("Después postprocesamiento")
        print(f"RAM usada en postprocesamiento: {post_end_mem - post_start_mem:+.1f} MB", flush=True)
        
        # Guardar resultados
        base_name = os.path.basename(img_path).split('.')[0]
        if make_tif:
            tif_out = os.path.join(output_folder, f"{base_name}_{application_name}_argmax.tif")
            save_tiff_mask(output_mask_processed, tif_out, dataset)
        if make_png:
            png_out = os.path.join(output_folder, f"{base_name}_{application_name}_argmax.png")
            save_png_mask(output_mask_processed, png_out)

        # Limpiar temporales
        del img, output_mask
        try:
            os.remove(temp_img_file)
            os.remove(temp_out_file)
            print(f"Archivos temporales eliminados", flush=True)
        except Exception as e:
            print(f"Advertencia: No se pudieron eliminar archivos temporales: {e}", flush=True)
        
        # Memoria final
        final_mem = print_memory_status("Fin procesamiento imagen")
        total_mem_used = final_mem - start_mem
        print(f"\nResumen memoria para {os.path.basename(img_path)}:", flush=True)
        print(f"  Memoria inicial: {start_mem:.1f} MB", flush=True)
        print(f"  Memoria final: {final_mem:.1f} MB", flush=True)
        print(f"  Memoria usada total: {total_mem_used:+.1f} MB", flush=True)
        print(f"  Memoria para cargar modelo: {model_load_mem - start_mem:.1f} MB", flush=True)
        
        print(f"\n{'='*60}", flush=True)
        print(f"SEGMENTACIÓN COMPLETADA: {img_path}", flush=True)
        print(f"{'='*60}\n", flush=True)

# Ejecutar
if __name__ == "__main__":
    print("\n" + "="*70, flush=True)
    print("INICIANDO SEGMENTACIÓN DE PALMERAS", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Memoria inicial
    initial_total_mem = get_memory_usage()
    print(f"[INICIO] Memoria proceso: {initial_total_mem:.1f} MB", flush=True)
    
    start_time = time.time()
    
    apply_semantic_segmentation_onnx(
        input_file_list=feature_file_list,
        output_folder=output_folder,
        application_name=application_name,
        model_onnx_path=model_onnx_path,
        window_radius=window_radius,
        internal_window_radius=internal_window_radius,
        make_tif=True,
        make_png=True,
        scaling=args.scaling
    )
    
    # Tiempo y memoria final
    end_time = time.time()
    final_total_mem = get_memory_usage()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*70}", flush=True)
    print("RESUMEN FINAL DE EJECUCIÓN:", flush=True)
    print(f"  Tiempo total: {elapsed_time:.1f} segundos", flush=True)
    print(f"  Memoria inicial: {initial_total_mem:.1f} MB", flush=True)
    print(f"  Memoria final: {final_total_mem:.1f} MB", flush=True)
    print(f"  Memoria usada neta: {final_total_mem - initial_total_mem:+.1f} MB", flush=True)
    print(f"{'='*70}", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("PROCESAMIENTO FINALIZADO EXITOSAMENTE", flush=True)
    print("="*70, flush=True)
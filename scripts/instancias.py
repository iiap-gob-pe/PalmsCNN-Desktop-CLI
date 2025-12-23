import os
import sys
import argparse
import psutil
import time
from osgeo import gdal, ogr, osr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
import onnxruntime as ort
from scipy.ndimage import binary_erosion
import skimage.morphology
import geopandas as gpd
import pandas as pd
import tempfile

# Configuración de argumentos
parser = argparse.ArgumentParser(description='Conteo de instancias de palmeras')
parser.add_argument('image_path', help='Ruta a la imagen original')
parser.add_argument('--window_radius', type=int, default=350, help='Radio de la ventana (350 para 700x700)')
parser.add_argument('--mask_path', help='Ruta a la máscara de segmentación')
parser.add_argument('--model_path', help='Ruta al modelo ONNX', default='models/model_converted.onnx')
parser.add_argument('--output_dir', help='Directorio de salida', default='output')
parser.add_argument('--overlap', type=int, default=128, help='Solapamiento entre ventanas (128px como el modo optimizado)')  # NUEVO PARÁMETRO

args = parser.parse_args()

# Función para obtener uso de RAM
def get_memory_usage():
    """Obtiene el uso de memoria actual del proceso en MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # MB

# Usar parámetros de línea de comandos
INPUT_RASTER = args.image_path
MASK_CLAS = args.mask_path if args.mask_path else os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.tif', '_balanced_argmax.tif'))
OUTPUT_RASTER = os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.tif', '_predicted.tif'))
OUTPUT_RASTER_CLAS = os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.tif', '_predicted_clas.tif'))
OUTPUT_IMAGE = os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.tif', '_predicted.png'))

# Archivos vectoriales de salida
OUTPUT_POLYGONS = os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.tif', '_predicted_poly.gpkg'))
OUTPUT_CENTROIDS = os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.tif', '_predicted_centers.gpkg'))
OUTPUT_ATTRIBUTES = os.path.join(args.output_dir, os.path.basename(args.image_path).replace('.tif', '_predicted_atributos.csv'))

# Tamaño fijo para el modelo de 700x700
window_radius = args.window_radius
internal_window_radius = int(round(window_radius * 0.75))  # 525x525 interna
output_folder = args.output_dir
overlap = args.overlap  # NUEVO: Overlap configurable

print("=== PARAMETROS DE INSTANCIAS (OPTIMIZADO MEMMAP) ===", flush=True)
print(f"Imagen: {INPUT_RASTER}", flush=True)
print(f"Mascara: {MASK_CLAS}", flush=True)
print(f"Modelo: {args.model_path}", flush=True)
print(f"Ventana: {window_radius*2}x{window_radius*2}", flush=True)
print(f"Ventana interna: {internal_window_radius*2}x{internal_window_radius*2}", flush=True)
print(f"Overlap: {overlap}px", flush=True)  # NUEVO: Mostrar overlap
print("================================", flush=True)

# Verificar que los archivos de entrada existan
def verificar_archivos():
    archivos = {
        "Imagen RGB": INPUT_RASTER,
        "Máscara": MASK_CLAS
    }
    
    for nombre, ruta in archivos.items():
        if not os.path.exists(ruta):
            print(f"ERROR: {nombre} no encontrado en: {ruta}", flush=True)
            return False
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    return True

if not verificar_archivos():
    sys.exit(1)

# Cargar el modelo ONNX
if not os.path.exists(args.model_path):
    print(f"ERROR: Modelo no encontrado: {args.model_path}", flush=True)
    sys.exit(1)

try:
    # OPTIMIZACION DE MEMORIA ONNX
    sess_opts = ort.SessionOptions()
    sess_opts.enable_cpu_mem_arena = False # Liberar memoria inmediatamente
    session = ort.InferenceSession(args.model_path, sess_opts, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    print("Modelo ONNX cargado correctamente", flush=True)
except Exception as e:
    print(f"ERROR cargando modelo ONNX: {e}", flush=True)
    sys.exit(1)

# Constantes para PIXELADO/JAGGED
CLASS_TO_SS = {"mauritia": -128, "euterpe": -96, "oenocarpus": -64}
CLASS_TO_CITYSCAPES = {"mauritia": 15, "euterpe": 25, "oenocarpus": 35}
THRESHOLD = {"mauritia": 3, "euterpe": 1, "oenocarpus": 2} 
MIN_SIZE = {"mauritia": 500, "euterpe": 400, "oenocarpus": 200}
SELEM = {3: np.ones((3, 3), dtype=bool), 1: np.ones((1, 1), dtype=bool), 2: np.ones((1, 1), dtype=bool)}
SELEN = {3: np.ones((36, 3), dtype=bool), 1: np.ones((7, 7), dtype=bool), 2: np.ones((3, 3), dtype=bool)} 

nodata_value = -9999

def scale_image(image, flag=None, nodata_value=nodata_value):
    if flag is None: return image
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
    ninstances = {"mauritia": 0, "euterpe": 0, "oenocarpus": 0}
    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ccImage = (raster == csCode)
        ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000)
        ccLabels = skimage.morphology.label(ccImage)
        ccIDs = np.unique(ccLabels)[1:]
        ninstances[semClass] = len(ccIDs)
        for ccID in ccIDs:          
            ccIDMask = (ccLabels == ccID)
            resultImage[ccIDMask] = csCode
    return resultImage.astype(np.float32), ninstances

def vectorize_and_analyze_results(input_raster_path, output_poly_path, output_centroids_path, output_csv_path):
    print("Iniciando vectorización y análisis...", flush=True)
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None: return 0, 0, 0, 0, 0, 0
    
    srcband = src_ds.GetRasterBand(1)
    dst_layername = 'palms_Area_ha'
    drv = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_poly_path): drv.DeleteDataSource(output_poly_path)
    dst_ds = drv.CreateDataSource(output_poly_path)
    
    sp_ref = osr.SpatialReference()
    if src_ds.GetProjection(): sp_ref.ImportFromWkt(src_ds.GetProjection())

    dst_layer = dst_ds.CreateLayer(dst_layername, srs=sp_ref)
    fld = ogr.FieldDefn("ID", ogr.OFTInteger)
    dst_layer.CreateField(fld)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("ID")

    gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)
    del src_ds
    del dst_ds

    gdf = gpd.read_file(output_poly_path)
    gdf = gdf[gdf["ID"] > 0]

    if len(gdf) == 0: return 0, 0, 0, 0, 0, 0

    c1, c2, c3, ca1, ca2, ca3 = 0, 0, 0, 0, 0, 0

    def generateCentroidColumns(item):
        nonlocal c1, c2, c3, ca1, ca2, ca3
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

        return pd.Series([item['ID'], ESPECIE, geom.area, geom.centroid.x, geom.centroid.y])

    gdf[['ID', 'ESPECIE', 'ÁREA(m2)', 'UTM(ESTE)', 'UTM(NORTE)']] = gdf.apply(generateCentroidColumns, axis=1)

    def generarcentroid(item):
        return pd.Series([item.ESPECIE, item.geometry.centroid])
    
    dfcentroid = gdf.apply(generarcentroid, axis=1)
    dfcentroid.columns = ['ESPECIE', 'geometry']
    dfcentroid = gpd.GeoDataFrame(dfcentroid, geometry='geometry')

    gdf.to_file(output_poly_path)
    dfcentroid.to_file(output_centroids_path)
    gdf.to_csv(output_csv_path, index=False)
    
    return c1, c2, c3, ca1, ca2, ca3

# ---------------------------------------------------------
# EJECUCIÓN PRINCIPAL CON GESTIÓN DE MEMMAP
# ---------------------------------------------------------

# Re-abrimos para lógica limpia
dataset = gdal.Open(INPUT_RASTER)
datasetresponse = gdal.Open(MASK_CLAS)
bandas = min(dataset.RasterCount, 3)

# Obtener información de la imagen
width = dataset.RasterXSize
height = dataset.RasterYSize
print(f"Dimensiones de la imagen: {width}x{height}", flush=True)
print(f"Bandas: {bandas}", flush=True)

# 1. CREAR MEMMAP EN LUGAR DE NP.ZEROS
temp_file = os.path.join(output_folder, 'temp_inst_matrix.dat')
if os.path.exists(temp_file): 
    try: 
        os.remove(temp_file)
        print("Archivo temporal anterior eliminado", flush=True)
    except: 
        print("No se pudo eliminar archivo temporal anterior", flush=True)
    
# Esta variable 'output' ahora vive en DISCO, no en RAM
output = np.memmap(temp_file, dtype=np.float32, mode='w+', shape=(height, width))
output[:] = nodata_value
print(f"Memmap creado en disco: {temp_file}", flush=True)

# ============================================================================
# CONFIGURACIÓN DE VENTANAS CON OVERLAP (MODIFICACIÓN CRÍTICA)
# ============================================================================
cr = [0, width]
rr = [0, height]

# Calcular paso con overlap (igual que el modo optimizado)
step_size = (internal_window_radius * 2) - overlap  # Reducir paso para solapar
print(f"Step size con overlap: {step_size}px (internal_window_radius*2={internal_window_radius*2} - overlap={overlap})", flush=True)

# Generar listas de posiciones con overlap
collist = [x for x in range(cr[0] + window_radius, cr[1] - window_radius, step_size)]
if collist and collist[-1] < cr[1] - window_radius: 
    collist.append(cr[1] - window_radius)
rowlist = [x for x in range(rr[0] + window_radius, rr[1] - window_radius, step_size)]
if rowlist and rowlist[-1] < rr[1] - window_radius: 
    rowlist.append(rr[1] - window_radius)

win_size = window_radius * 2

print(f"Procesando {len(rowlist)} filas x {len(collist)} columnas", flush=True)
print(f"Total de ventanas: {len(rowlist) * len(collist)}", flush=True)
print(f"Tamaño de ventana: {win_size}x{win_size}", flush=True)
print(f"Overlap efectivo entre ventanas: {overlap}px", flush=True)

# Memoria inicial
initial_memory = get_memory_usage()
print(f"RAM inicial: {initial_memory:.1f} MB", flush=True)
print("", flush=True)

# Variables para estadísticas
start_time = time.time()
total_columns = len(collist)
processed_columns = 0

# ============================================================================
# BUCLE PRINCIPAL CON OVERLAP
# ============================================================================
for col_idx, col in enumerate(collist):
    processed_columns += 1
    
    # Memoria antes de procesar esta columna
    mem_before = get_memory_usage()
    
    imageBatch = []
    responses = []
    valid_rows = []
    
    # Recolección de batch
    for n in rowlist:
        d = np.zeros((win_size, win_size, bandas))
        for b in range(bandas):
            band_data = dataset.GetRasterBand(b + 1).ReadAsArray(col - window_radius, n - window_radius, win_size, win_size)
            if band_data is not None: 
                d[:, :, b] = band_data
            else: 
                d[:, :, b] = nodata_value
        
        d[np.isnan(d)] = nodata_value
        d[d == -9999] = nodata_value
        
        r = datasetresponse.GetRasterBand(1).ReadAsArray(col - window_radius, n - window_radius, win_size, win_size)
        if r is None: 
            r = np.zeros((win_size, win_size)) + nodata_value
        else: 
            r = r.astype(float)
        
        if d.shape[0] == win_size and d.shape[1] == win_size:
            d = scale_image(d)
            imageBatch.append(d)
            responses.append(r)
            valid_rows.append(n)

    if not imageBatch:
        print(f"Columna {col_idx+1}/{total_columns}: Sin ventanas válidas", flush=True)
        continue

    # Memoria después de cargar datos
    mem_after_load = get_memory_usage()
    
    imageBatch = np.stack(imageBatch)
    responses = np.stack(responses)

    ssBatch = (responses > 0).astype(np.float32)
    ssMaskBatch = np.zeros_like(responses, dtype=np.float32)
    
    for i in range(responses.shape[0]):
        r_i = responses[i]
        ssMaskBatch[i][r_i == 1] = CLASS_TO_SS["mauritia"]
        ssMaskBatch[i][r_i == 2] = CLASS_TO_SS["euterpe"]
        ssMaskBatch[i][r_i == 3] = CLASS_TO_SS["oenocarpus"]

    imageBatch = imageBatch.reshape((imageBatch.shape[0], win_size, win_size, bandas))
    outputBatch = np.zeros((len(valid_rows), win_size, win_size), dtype=np.uint8)

    # Inferencia
    successful_inferences = 0
    for j in range(len(valid_rows)):
        try:
            img_j = imageBatch[j] * ssBatch[j][..., np.newaxis]
            input_j = np.concatenate([img_j, ssMaskBatch[j][..., np.newaxis]], axis=-1).astype(np.float32)
            ss_batch_input = ssBatch[j].astype(np.float32)
            
            outputs = session.run(None, {
                input_names[0]: input_j[np.newaxis, ...],
                input_names[1]: ss_batch_input[np.newaxis, ...]
            })
            outputBatch[j] = outputs[0][0].astype(np.uint8)
            successful_inferences += 1
            
        except Exception as e:
            continue

    # Memoria después de inferencia
    mem_after_inference = get_memory_usage()

    # Watershed y Reconstrucción
    outputdwt = []
    for j in range(len(outputBatch)):
        try:
            outputImage = watershed_cut(outputBatch[j], ssMaskBatch[j])
            outputdwt.append(outputImage)
        except:
            outputdwt.append(np.zeros((win_size, win_size), dtype=np.float32))

    # ============================================================================
    # ENSAMBLAR RESULTADOS CON OVERLAP (MODIFICACIÓN CRÍTICA)
    # ============================================================================
    if outputdwt:
        outputdwt = np.stack(outputdwt)
        for j, n in enumerate(valid_rows):
            p = outputdwt[j]
            if internal_window_radius < window_radius:
                mm = int(np.rint(window_radius - internal_window_radius))
                p = p[mm:-mm, mm:-mm]
            
            # Calcular región de escritura con overlap
            start_row = n - internal_window_radius
            end_row = n + internal_window_radius
            start_col = col - internal_window_radius
            end_col = col + internal_window_radius
            
            # Aplicar overlap: para ventanas que no están en los bordes extremos
            # reducir la región de escritura para evitar duplicados
            if col != collist[0] and col != collist[-1] and n != rowlist[0] and n != rowlist[-1]:
                overlap_margin = overlap // 2
                start_row += overlap_margin
                end_row -= overlap_margin
                start_col += overlap_margin
                end_col -= overlap_margin
                p = p[overlap_margin:-overlap_margin, overlap_margin:-overlap_margin]
            
            # Escribir en el MEMMAP (Disco) en lugar de RAM
            if (start_row >= 0 and end_row <= output.shape[0] and 
                start_col >= 0 and end_col <= output.shape[1]):
                output[start_row:end_row, start_col:end_col] = p
                
    # Liberar memoria del batch
    output.flush()  # Guardar cambios en disco
    del imageBatch, responses, outputBatch, outputdwt
    
    # Memoria después de liberar
    mem_after_free = get_memory_usage()
    
    # Calcular progreso
    elapsed_time = time.time() - start_time
    progress_percent = (processed_columns / total_columns) * 100
    
    # Mostrar información de la columna
    print(f"Columna {col_idx+1}/{total_columns} completada", flush=True)
    print(f"  Ventanas procesadas: {len(valid_rows)}", flush=True)
    print(f"  Inferencias exitosas: {successful_inferences}/{len(valid_rows)}", flush=True)
    print(f"  RAM uso:", flush=True)
    print(f"    Inicio columna: {mem_before:.1f} MB", flush=True)
    print(f"    Después de cargar: {mem_after_load:.1f} MB (+{mem_after_load - mem_before:.1f} MB)", flush=True)
    print(f"    Después de inferencia: {mem_after_inference:.1f} MB", flush=True)
    print(f"    Después de liberar: {mem_after_free:.1f} MB", flush=True)
    print(f"    Uso máximo en columna: {max(mem_before, mem_after_load, mem_after_inference, mem_after_free):.1f} MB", flush=True)
    print(f"  Progreso: {progress_percent:.1f}%", flush=True)
    
    # Calcular ETA
    if progress_percent > 0:
        estimated_total_time = elapsed_time / (progress_percent / 100)
        remaining_time = estimated_total_time - elapsed_time
        
        if remaining_time > 3600:
            eta_str = f"{remaining_time/3600:.1f} horas"
        elif remaining_time > 60:
            eta_str = f"{remaining_time/60:.1f} minutos"
        else:
            eta_str = f"{remaining_time:.0f} segundos"
        
        print(f"  Tiempo transcurrido: {elapsed_time:.1f}s", flush=True)
        print(f"  ETA: {eta_str}", flush=True)
    
    print("", flush=True)

# Memoria final antes de post-procesamiento
mem_before_post = get_memory_usage()
print(f"RAM antes del post-procesamiento: {mem_before_post:.1f} MB", flush=True)

# PROCESAMIENTO FINAL
print("Realizando post-procesamiento...", flush=True)
output, quantification = process_instances_raster(output)

# Guardar TIFF
print("Guardando raster final...", flush=True)
name_saved_final = os.path.basename(INPUT_RASTER).replace('.tif', '_predicted.tif')
out_path = os.path.join(output_folder, name_saved_final)
driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(out_path, output.shape[1], output.shape[0], 1, gdal.GDT_Float32)

# Escribir por bloques para no saturar RAM
block_size = 1024
print(f"Escribiendo en bloques de {block_size}...", flush=True)
for y in range(0, output.shape[0], block_size):
    rows = min(block_size, output.shape[0] - y)
    out_ds.GetRasterBand(1).WriteArray(output[y:y+rows, :], 0, y)

out_ds.SetGeoTransform(dataset.GetGeoTransform())
out_ds.SetProjection(dataset.GetProjection())
out_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
out_ds.FlushCache()
out_ds = None

print(f"Raster guardado: {out_path}", flush=True)

# Copiar clasificación
if os.path.exists(MASK_CLAS):
    import shutil
    shutil.copy2(MASK_CLAS, OUTPUT_RASTER_CLAS)
    print(f"Clasificacion copiada: {OUTPUT_RASTER_CLAS}", flush=True)

# Resultados texto
mau = quantification['mauritia']
eut = quantification['euterpe']
oeno = quantification['oenocarpus']
total = mau + eut + oeno

print("", flush=True)
print("RESULTADOS DEL CONTEO:", flush=True)
print(f"Mauritia flexuosa: {mau} palmeras", flush=True)
print(f"Euterpe precatoria: {eut} palmeras", flush=True)
print(f"Oenocarpus bataua: {oeno} palmeras", flush=True)
print(f"TOTAL: {total} palmeras", flush=True)

# Generar PNG
if os.path.exists(OUTPUT_RASTER):
    try:
        # Leemos con GDAL para hacer downsample si es muy grande
        ds_png = gdal.Open(OUTPUT_RASTER)
        w_png = ds_png.RasterXSize
        h_png = ds_png.RasterYSize
        scale = max(1, max(w_png, h_png) // 2000)  # Escalar si es gigante
        
        print(f"Generando imagen PNG...", flush=True)
        print(f"  Dimensiones originales: {w_png}x{h_png}", flush=True)
        print(f"  Factor de escala: 1/{scale}", flush=True)
        
        img_arr = ds_png.ReadAsArray(0, 0, w_png, h_png, buf_xsize=w_png//scale, buf_ysize=h_png//scale)
        
        colors = ["#000000", "#ff7f00", "#08F6EB", "#E008F6"]
        custom_cmap = ListedColormap(colors)
        bounds = [0, 15, 25, 35, 40]
        norm = BoundaryNorm(bounds, custom_cmap.N)
        
        plt.figure(figsize=(12, 10))
        plt.imshow(img_arr, cmap=custom_cmap, norm=norm, interpolation='nearest')
        plt.axis('off')
        plt.savefig(OUTPUT_IMAGE, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        print(f"PNG generado: {OUTPUT_IMAGE}", flush=True)
        
    except Exception as e:
        print(f"Advertencia: No se pudo generar PNG - {e}", flush=True)

# Vectorizar
print("Vectorizando resultados...", flush=True)
vectorize_and_analyze_results(OUTPUT_RASTER, OUTPUT_POLYGONS, OUTPUT_CENTROIDS, OUTPUT_ATTRIBUTES)

print(f"Archivos vectoriales generados:", flush=True)
print(f"  {OUTPUT_POLYGONS}", flush=True)
print(f"  {OUTPUT_CENTROIDS}", flush=True)
print(f"  {OUTPUT_ATTRIBUTES}", flush=True)

# Limpiar archivo temporal
print("Limpiando archivo temporal...", flush=True)
del output
try:
    if os.path.exists(temp_file): 
        os.remove(temp_file)
        print("Archivo temporal eliminado", flush=True)
except Exception as e:
    print(f"Advertencia: No se pudo eliminar archivo temporal - {e}", flush=True)

# Memoria final
final_memory = get_memory_usage()
total_time = time.time() - start_time

print("", flush=True)
print("=" * 60, flush=True)
print("RESUMEN FINAL:", flush=True)
print("=" * 60, flush=True)
print(f"Tiempo total de ejecucion: {total_time:.2f} segundos", flush=True)
print(f"RAM inicial: {initial_memory:.1f} MB", flush=True)
print(f"RAM final: {final_memory:.1f} MB", flush=True)
print(f"Uso neto de RAM: {final_memory - initial_memory:+.1f} MB", flush=True)
print(f"Columnas procesadas: {processed_columns}/{total_columns}", flush=True)
print(f"Total palmeras detectadas: {total}", flush=True)
print("=" * 60, flush=True)
print("Finalizado.", flush=True)
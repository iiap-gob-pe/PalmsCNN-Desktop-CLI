import os
import psutil
import numpy as np
from osgeo import gdal, ogr, osr
import onnxruntime as ort
from scipy.ndimage import binary_erosion
import skimage.morphology
from skimage.measure import label # NUEVO IMPORT NECESARIO
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.wkt import loads  # NUEVO IMPORT PARA CONVERTIR GEOMETRÍAS OGR A SHAPELY
import warnings
warnings.filterwarnings('ignore')

# Constantes (iguales)
CLASS_TO_SS = {"mauritia":-128, "euterpe":-96, "oenocarpus":-64}
CLASS_TO_CITYSCAPES = {"mauritia":15, "euterpe":25, "oenocarpus":35}
THRESHOLD = {"mauritia":3, "euterpe":1, "oenocarpus":2}
MIN_SIZE = {"mauritia":500, "euterpe":400, "oenocarpus":200}
SELEM = {3: np.ones((3,3), dtype=bool), 1: np.ones((1,1), dtype=bool), 2: np.ones((1,1), dtype=bool)}
SELEN = {3: np.ones((36,3), dtype=bool), 1: np.ones((7,7), dtype=bool), 2: np.ones((3,3), dtype=bool)}
nodata_value = -9999

# ======================================================================
# FUNCIÓN DE VECTORIZACIÓN OPTIMIZADA (REEMPLAZADA)
# ======================================================================
def raster_to_vector_gpkg(raster_path, output_poly_path, output_centers_path, output_csv_path, geotransform=None, projection=None):
    """
    Versión optimizada usando GDAL.Polygonize (mucho más rápida, menos memoria y polígonos exactos)
    """
    print(f"\n[VECTOR] Iniciando vectorización optimizada con GDAL.Polygonize: {raster_path}")

    try:
        src_ds = gdal.Open(raster_path)
        if src_ds is None:
            print(f"[ERROR] No se pudo abrir el raster: {raster_path}")
            return None, None, None

        srcband = src_ds.GetRasterBand(1)

        if geotransform is None:
            geotransform = src_ds.GetGeoTransform()
        if projection is None:
            projection = src_ds.GetProjection()

        # Área aproximada de un píxel (para estimar N_PIXELES)
        pixel_area = abs(geotransform[1] * geotransform[5])

        # Preparar capa en memoria para Polygonize
        mem_driver = ogr.GetDriverByName('Memory')
        poly_ds = mem_driver.CreateDataSource('mem')
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)
        poly_layer = poly_ds.CreateLayer('polys', srs=srs, geom_type=ogr.wkbPolygon)

        # Campo para el valor de clase
        class_field = ogr.FieldDefn('CLASS_ID', ogr.OFTInteger)
        poly_layer.CreateField(class_field)
        class_idx = poly_layer.GetLayerDefn().GetFieldIndex('CLASS_ID')

        # Polygonize (usamos 8-connected para coincidir con connectivity=2 del label original)
        print("[VECTOR] Ejecutando GDAL.Polygonize (esto será muy rápido)...")
        gdal.Polygonize(srcband, None, poly_layer, class_idx, ['8CONNECTED=8'])

        print(f"[VECTOR] Polygonize completado. Procesando {poly_layer.GetFeatureCount()} features...")

        # Listas para GeoDataFrames (compatibilidad con código original)
        polygons_list = []
        centers_list = []
        attributes = []

        # Contadores y mapeos
        counters = {"mauritia": 0, "euterpe": 0, "oenocarpus": 0}
        class_to_especie = {15: "Mauritia flexuosa", 25: "Euterpe precatoria", 35: "Oenocarpus bataua"}
        class_to_prefix = {15: "MFLX", 25: "EPRE", 35: "OBAT"}

        poly_layer.ResetReading()
        for feat in poly_layer:
            class_id = feat.GetField('CLASS_ID')
            if class_id not in [15, 25, 35]:
                continue

            geom = feat.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                continue

            # Convertir a Shapely
            shapely_geom = loads(geom.ExportToWkt())

            area_m2 = geom.Area()
            n_pixels = int(round(area_m2 / pixel_area)) if pixel_area > 0 else 0

            centroid = geom.Centroid()
            cx, cy = centroid.GetX(), centroid.GetY()

            # Determinar especie y contador
            especie = class_to_especie[class_id]
            key = {15: "mauritia", 25: "euterpe", 35: "oenocarpus"}[class_id]
            counters[key] += 1
            instancia_id = f"{class_to_prefix[class_id]}_{counters[key]:04d}"

            # Para polígonos
            polygons_list.append({
                'geometry': shapely_geom,
                'ID': instancia_id,
                'ESPECIE': especie,
                'ID_RASTER': class_id,
                'AREA_M2': area_m2,
                'CX': cx,
                'CY': cy,
                'N_PIXELES': n_pixels
            })

            # Para centroides
            centers_list.append({
                'geometry': Point(cx, cy),
                'ID': instancia_id,
                'ESPECIE': especie,
                'ID_RASTER': class_id,
                'AREA_M2': area_m2
            })

            # Para CSV
            attributes.append({
                'ID': instancia_id,
                'ESPECIE': especie,
                'ID_RASTER': class_id,
                'AREA_M2': area_m2,
                'UTM_ESTE': cx,
                'UTM_NORTE': cy,
                'N_PIXELES': n_pixels
            })

        # Limpiar datasource en memoria
        poly_ds = None

        # Resumen
        total = sum(counters.values())
        print(f"[VECTOR] Total polígonos creados: {total}")
        print(f" - Mauritia flexuosa: {counters['mauritia']}")
        print(f" - Euterpe precatoria: {counters['euterpe']}")
        print(f" - Oenocarpus bataua: {counters['oenocarpus']}")

        if polygons_list:
            # Crear GeoDataFrames
            gdf_polygons = gpd.GeoDataFrame(polygons_list, geometry='geometry')
            gdf_polygons.crs = projection

            gdf_centers = gpd.GeoDataFrame(centers_list, geometry='geometry')
            gdf_centers.crs = projection

            # Guardar archivos
            print(f"[VECTOR] Guardando polígonos GPKG: {output_poly_path}")
            gdf_polygons.to_file(output_poly_path, driver='GPKG', layer='palmeras_poligonos')

            print(f"[VECTOR] Guardando centroides GPKG: {output_centers_path}")
            gdf_centers.to_file(output_centers_path, driver='GPKG', layer='palmeras_centroides')

            print(f"[VECTOR] Guardando CSV: {output_csv_path}")
            df_csv = pd.DataFrame(attributes)
            df_csv.to_csv(output_csv_path, index=False, encoding='utf-8')

            # Verificar tamaños
            for path, name in [(output_poly_path, "GPKG polígonos"), (output_centers_path, "GPKG centroides"), (output_csv_path, "CSV atributos")]:
                if os.path.exists(path):
                    size = os.path.getsize(path) / (1024 * 1024) if 'GPKG' in name else os.path.getsize(path) / 1024
                    unit = "MB" if 'GPKG' in name else "KB"
                    print(f"[VECTOR] {name}: {size:.2f} {unit}")

            return gdf_polygons, gdf_centers, df_csv
        else:
            print("[VECTOR] No se encontraron polígonos para vectorizar")
            return None, None, None

    except Exception as e:
        print(f"[ERROR] Error en vectorización optimizada: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        src_ds = None
        gc.collect()

# ======================================================================
# FUNCIONES EXISTENTES (sin cambios)
# ======================================================================
def watershed_cut(depthImage, ssMask):
    resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)
    for semClass in CLASS_TO_CITYSCAPES.keys():
        csCode = CLASS_TO_CITYSCAPES[semClass]
        ssCode = CLASS_TO_SS[semClass]
        ssMaskClass = (ssMask == ssCode)
        ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass
        ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000)
        ccLabels = skimage.morphology.label(ccImage)
        ccIDs = np.unique(ccLabels)[1:]
        for ccID in ccIDs:
            ccIDMask = (ccLabels == ccID)
            ccIDMask = skimage.morphology.binary_erosion(ccIDMask, SELEM[THRESHOLD[semClass]])
            ccIDMask = binary_erosion(ccIDMask, SELEN[THRESHOLD[semClass]])
            resultImage[ccIDMask] = csCode
    return resultImage.astype(np.float32)
def process_instances_raster_tiled(raster, chunk_height=8000, overlap=800):
    print("[MEM] Iniciando post-procesamiento optimizado en tiles (por franjas)...")
   
    height, width = raster.shape
    ninstances = {"mauritia": 0, "euterpe": 0, "oenocarpus": 0}
   
    step = chunk_height - overlap
    for start_row in range(0, height, step):
        end_row = min(start_row + chunk_height, height)
       
        # Franja con overlap
        o_start = max(0, start_row - overlap)
        o_end = min(height, end_row + overlap)
       
        chunk = raster[o_start:o_end, :].copy()
       
        chunk_mask = np.zeros(chunk.shape, dtype=bool)
       
        for semClass in CLASS_TO_CITYSCAPES.keys():
            csCode = CLASS_TO_CITYSCAPES[semClass]
           
            ccImage = (chunk == csCode)
           
            ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])
            ccImage = skimage.morphology.remove_small_holes(ccImage, area_threshold=1000)
           
            ccLabels = label(ccImage)
           
            border_labels = set()
            if o_start > 0:
                border_labels.update(np.unique(ccLabels[0, :]))
            if o_end < height:
                border_labels.update(np.unique(ccLabels[-1, :]))
            border_labels.discard(0)
           
            unique_labels = np.unique(ccLabels)
            unique_labels = unique_labels[unique_labels > 0]
            internal_labels = [lbl for lbl in unique_labels if lbl not in border_labels]
            ninstances[semClass] += len(internal_labels)
           
            chunk_mask |= ccImage
           
            del ccImage, ccLabels
            gc.collect()
       
        # Aplicar máscara solo a la parte central (sin overlap)
        inner_start = max(0, start_row - o_start)
        inner_end = inner_start + (end_row - start_row)
        inner_mask = chunk_mask[inner_start:inner_end, :]
       
        raster[start_row:end_row, :][~inner_mask] = nodata_value
       
        del chunk, chunk_mask
        gc.collect()
       
        print(f"[MEM] Chunk procesado: filas {start_row}-{end_row}/{height}")
   
    return raster, ninstances
def prepare_window(args):
    """Función para preparar una ventana (usada en paralelo)"""
    (row, col, full_image, full_mask, window_radius) = args
    col_start = col - window_radius
    col_end = col + window_radius
    row_start = row - window_radius
    row_end = row + window_radius
    d = full_image[row_start:row_end, col_start:col_end, :].copy()
    r = full_mask[row_start:row_end, col_start:col_end].copy()
    ssBatch = (r > 0).astype(np.float32)
    ssMaskBatch = np.zeros_like(r, dtype=np.float32)
    ssMaskBatch[r == 1] = CLASS_TO_SS["mauritia"]
    ssMaskBatch[r == 2] = CLASS_TO_SS["euterpe"]
    ssMaskBatch[r == 3] = CLASS_TO_SS["oenocarpus"]
    img_j = d * ssBatch[..., np.newaxis]
    input_j = np.concatenate([img_j, ssMaskBatch[..., np.newaxis]], axis=-1).astype(np.float32)
    ss_batch_input = ssBatch.astype(np.float32)
    return input_j, ss_batch_input, ssMaskBatch, (row, col, row_start, row_end, col_start, col_end)
def generate_png_visualization(output_array, output_path):
    """Genera el archivo PNG con los puntos de colores"""
    print(f"\n[IMG] GENERANDO IMAGEN DE VISUALIZACIÓN PNG...", flush=True)
   
    try:
        # Colores: Negro, Naranja (Mauritia), Cian (Euterpe), Púrpura (Oenocarpus)
        colors = ["#000000", "#ff7f00", "#08F6EB", "#E008F6"]
        custom_cmap = ListedColormap(colors)
        bounds = [0, 15, 25, 35, 40]
        norm = BoundaryNorm(bounds, custom_cmap.N)
        # Configurar figura (DPI bajo para ahorrar RAM con imágenes gigantes)
        plt.figure(figsize=(12, 8), dpi=100)
       
        # Convertir y plotear
        plt.imshow(output_array, cmap=custom_cmap, norm=norm, interpolation='nearest')
        plt.axis('off')
       
        # Guardar
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close('all')
       
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] Imagen PNG guardada: {output_path} ({size_mb:.2f} MB)", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] Falló Matplotlib ({e}). Intentando con OpenCV...", flush=True)
        try:
            import cv2
            # Mapeo manual de colores BGR para OpenCV
            h, w = output_array.shape
            rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
           
            # Mauritia (15) -> Naranja (BGR: 0, 127, 255)
            rgb_image[output_array == 15] = [0, 127, 255]
            # Euterpe (25) -> Cian (BGR: 235, 246, 8)
            rgb_image[output_array == 25] = [235, 246, 8]
            # Oenocarpus (35) -> Púrpura (BGR: 246, 8, 224)
            rgb_image[output_array == 35] = [246, 8, 224]
           
            cv2.imwrite(output_path, rgb_image)
            print(f"[OK] Imagen PNG guardada con OpenCV: {output_path}", flush=True)
            return True
        except Exception as e2:
            print(f"[FATAL] No se pudo generar PNG: {e2}", flush=True)
            return False
def apply_instance_onnx_tiled_optimized(feature_file_list, mask, roi, output_folder, session,
                                        window_radius, internal_window_radius):
    print("=" * 70)
    print("INICIANDO PROCESAMIENTO FULL RAM + BATCHES + 24 NUCLEOS")
    print("=" * 70)
    image_path = feature_file_list[0]
    mask_path = mask[0]
    # === CARGA ULTRA-RAPIDA ===
    print("\nCARGANDO IMAGEN Y MASCARA COMPLETAS EN RAM...")
    dataset_img = gdal.Open(image_path)
    dataset_mask = gdal.Open(mask_path)
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
   
    # Obtener geotransform y projection para usar después en vectorización
    geotransform = dataset_img.GetGeoTransform()
    projection = dataset_img.GetProjection()
    # Lectura multibanda unica
    full_image = dataset_img.ReadAsArray()
    if full_image.ndim == 3:
        full_image = np.transpose(full_image, (1, 2, 0))[:, :, :3].astype(np.float32)
    print(f"[OK] Imagen completa en RAM: {full_image.nbytes / 1024**2:.1f} MB")
    full_mask = dataset_mask.ReadAsArray().astype(np.float32)
    print(f"[OK] Mascara completa en RAM: {full_mask.nbytes / 1024**2:.1f} MB")
    dataset_img = None
    dataset_mask = None
    gc.collect()
    # Output en RAM
    output = np.zeros((height, width), dtype=np.float32) + nodata_value
    input_names = [inp.name for inp in session.get_inputs()]
   
    # === VENTANAS ===
    step = internal_window_radius * 2
    collist = list(range(window_radius, width - window_radius, step))
    rowlist = list(range(window_radius, height - window_radius, step))
    if width - window_radius not in collist:
        collist.append(width - window_radius)
    if height - window_radius not in rowlist:
        rowlist.append(height - window_radius)
    windows_coords = [(row, col) for col in collist for row in rowlist]
    total_windows = len(windows_coords)
    print(f"VENTANAS TOTALES: {total_windows}")
    # === VERIFICAR ESPECIFICACIONES DEL MODELO ===
    input_details = session.get_inputs()
    expected_batch_size = input_details[0].shape[0] if len(input_details[0].shape) > 0 else 1
    print(f"MODELO REQUIERE BATCH SIZE {expected_batch_size}")
   
    # Crear funcion de procesamiento para ventana individual
    def process_single_window(coords):
        row, col = coords
        col_start = col - window_radius
        col_end = col + window_radius
        row_start = row - window_radius
        row_end = row + window_radius
        d = full_image[row_start:row_end, col_start:col_end, :].copy()
        r = full_mask[row_start:row_end, col_start:col_end].copy()
        ssBatch = (r > 0).astype(np.float32)
        ssMaskBatch = np.zeros_like(r, dtype=np.float32)
        ssMaskBatch[r == 1] = CLASS_TO_SS["mauritia"]
        ssMaskBatch[r == 2] = CLASS_TO_SS["euterpe"]
        ssMaskBatch[r == 3] = CLASS_TO_SS["oenocarpus"]
        img_j = d * ssBatch[..., np.newaxis]
        input_j = np.concatenate([img_j, ssMaskBatch[..., np.newaxis]], axis=-1).astype(np.float32)
        ss_batch_input = ssBatch.astype(np.float32)
       
        input_j = np.expand_dims(input_j, axis=0)
        ss_batch_input = np.expand_dims(ss_batch_input, axis=0)
       
        outputs = session.run(None, {
            input_names[0]: input_j,
            input_names[1]: ss_batch_input
        })
       
        depthImage = outputs[0].astype(np.uint8)[0]
        outputImage = watershed_cut(depthImage, ssMaskBatch)
       
        if internal_window_radius < window_radius:
            mm = int(np.rint(window_radius - internal_window_radius))
            if outputImage.shape[0] > mm*2 and outputImage.shape[1] > mm*2:
                outputImage = outputImage[mm:-mm, mm:-mm]
       
        return {
            'row': row, 'col': col, 'output': outputImage,
            'internal_radius': internal_window_radius
        }
    # === PROCESAMIENTO PARALELO ===
    print(f"\nPROCESANDO {total_windows} VENTANAS EN PARALELO (24 NUCLEOS)...")
   
    results = []
    processed = 0
    lock = threading.Lock()
   
    def progress_callback(future):
        nonlocal processed
        with lock:
            processed += 1
            if processed % 50 == 0 or processed == total_windows:
                percentage = (processed / total_windows) * 100
                print(f"Progreso: {processed}/{total_windows} ({percentage:.1f}%)", flush=True)
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(process_single_window, coords): coords for coords in windows_coords}
        for future in futures:
            future.add_done_callback(lambda f: progress_callback(f))
       
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"ERROR procesando ventana: {e}")
    # === ENSAMBLAJE ===
    print("\nENSAMBLANDO RESULTADOS...")
    for result in results:
        row = result['row']
        col = result['col']
        outputImage = result['output']
        internal_radius = result['internal_radius']
       
        out_row_start = row - internal_radius
        out_row_end = out_row_start + outputImage.shape[0]
        out_col_start = col - internal_radius
        out_col_end = out_col_start + outputImage.shape[1]
       
        out_row_start = max(0, out_row_start)
        out_col_start = max(0, out_col_start)
        out_row_end = min(height, out_row_end)
        out_col_end = min(width, out_col_end)
       
        img_slice_rows = out_row_end - out_row_start
        img_slice_cols = out_col_end - out_col_start
       
        if img_slice_rows > 0 and img_slice_cols > 0:
            outputImage_slice = outputImage[:img_slice_rows, :img_slice_cols]
            output[out_row_start:out_row_end, out_col_start:out_col_end] = outputImage_slice
   
    print(f"ENSAMBLAJE COMPLETADO")
    # === LIBERACIÓN DE MEMORIA ===
    print("\n[MEM] Liberando imagen y máscara de entrada...")
    del full_image
    del full_mask
    gc.collect()
    # === POST-PROCESAMIENTO TILEADO ===
    print("\nPOST-PROCESAMIENTO Y GUARDADO...")
    output, quantification = process_instances_raster_tiled(output, chunk_height=8000, overlap=800)
    base_name = Path(image_path).stem
   
    # 1. Guardar TIFF
    tif_path = os.path.join(output_folder, f"{base_name}_predicted.tif")
    print(f"Guardando TIFF: {tif_path}")
    driver_img = gdal.Open(image_path)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(tif_path, width, height, 1, gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    out_ds.GetRasterBand(1).WriteArray(output)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_ds.GetRasterBand(1).SetNoDataValue(nodata_value)
    out_ds.FlushCache()
    out_ds = None
    driver_img = None
    # 2. Guardar CSV de resumen
    mau = quantification['mauritia']
    eut = quantification['euterpe']
    oeno = quantification['oenocarpus']
    csv_summary_path = os.path.join(output_folder, f"{base_name}_predicted_summary.csv")
    with open(csv_summary_path, 'w') as f:
        f.write("ESPECIE,CONTEO\n")
        f.write(f"Mauritia flexuosa,{mau}\n")
        f.write(f"Euterpe precatoria,{eut}\n")
        f.write(f"Oenocarpus bataua,{oeno}\n")
        f.write(f"TOTAL,{mau + eut + oeno}\n")
    print(f"Guardado CSV resumen: {csv_summary_path}")
    # 3. Guardar PNG
    png_path = os.path.join(output_folder, f"{base_name}_predicted.png")
    generate_png_visualization(output, png_path)
    # 4. VECTORIZAR A GPKG
    print("\n" + "="*60)
    print("GENERANDO ARCHIVOS VECTORIALES GPKG")
    print("="*60)
   
    # Definir rutas de salida para archivos vectoriales
    poly_gpkg_path = os.path.join(output_folder, f"{base_name}_polygons.gpkg")
    centers_gpkg_path = os.path.join(output_folder, f"{base_name}_centroids.gpkg")
    atributos_csv_path = os.path.join(output_folder, f"{base_name}_atributos.csv")
   
    # Llamar a la función de vectorización
    gdf_poly, gdf_cent, df_atrib = raster_to_vector_gpkg(
        tif_path,
        poly_gpkg_path,
        centers_gpkg_path,
        atributos_csv_path,
        geotransform,
        projection
    )
   
    if gdf_poly is not None:
        print(f"\n[EXITO] Vectorización completada exitosamente")
        print(f" - Polígonos: {poly_gpkg_path}")
        print(f" - Centroides: {centers_gpkg_path}")
        print(f" - Atributos: {atributos_csv_path}")
        print(f" - Total instancias vectorizadas: {len(gdf_poly)}")
    else:
        print(f"\n[ADVERTENCIA] No se pudieron generar archivos vectoriales")
    return f"{base_name}_predicted.tif", mau, eut, oeno, output
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('mask_path')
    parser.add_argument('--window_radius', type=int, default=350)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--model_path', default='models/model_converted.onnx')
    parser.add_argument('--internal_window_radius', type=int, default=None)
    args = parser.parse_args()
    if args.internal_window_radius is None:
        args.internal_window_radius = int(round(args.window_radius * 0.75))
    # Asegurar que existe el directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    # Sesion ONNX
    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 24
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session = ort.InferenceSession(args.model_path, sess_options=sess_options, providers=providers)
    result = apply_instance_onnx_tiled_optimized(
        [args.image_path],
        [args.mask_path],
        [],
        args.output_dir,
        session,
        args.window_radius,
        args.internal_window_radius
    )
    print("\n" + "="*60)
    print("RESULTADOS FINALES:")
    print("="*60)
    print(f"Mauritia flexuosa: {result[1]}")
    print(f"Euterpe precatoria: {result[2]}")
    print(f"Oenocarpus bataua: {result[3]}")
    print(f"TOTAL: {result[1] + result[2] + result[3]}")
   
    # Mostrar archivos generados
    base_name = Path(args.image_path).stem
    print(f"\nARCHIVOS GENERADOS EN {args.output_dir}:")
    print(f"1. {base_name}_predicted.tif (Ráster de instancias)")
    print(f"2. {base_name}_predicted.png (Imagen de visualización)")
    print(f"3. {base_name}_predicted_summary.csv (Resumen de conteo)")
    print(f"4. {base_name}_polygons.gpkg (Polígonos vectoriales)")
    print(f"5. {base_name}_centroids.gpkg (Centroides vectoriales)")
    print(f"6. {base_name}_atributos.csv (Atributos detallados)")
    print("="*60)
    print("PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
    print("="*60)
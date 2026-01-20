import os
import subprocess
import json
import sys
import logging
import numpy as np
from PIL import Image
import time
import rasterio
from rasterio.windows import Window
import re
import threading
import queue
import psutil
import glob
import pandas as pd

# ============================================================================
# CLASE SPINNER (ANIMACIÓN DE CARGA)
# ============================================================================
class ConsoleSpinner:
    """Clase para mostrar una animación en la terminal mientras se espera"""
    def __init__(self, message="Procesando... no cierre la ventana"):
        self.message = message
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.idx = 0
        
    def next_frame(self):
        """Retorna el siguiente cuadro de la animación"""
        frame = self.frames[self.idx % len(self.frames)]
        self.idx += 1
        return f"\r{frame} {self.message} "

# ============================================================================
# FUNCIÓN DE EJECUCIÓN CON ANIMACIÓN
# ============================================================================

def run_command_with_terminal_output(cmd, cwd=None, progress_callback=None, progress_range=(0, 100)):
    """Ejecuta comando y muestra animación mientras espera output."""
    print(f"\n[TERMINAL] Ejecutando: {' '.join(cmd)}")
    sys.stdout.flush()
    
    out_queue = queue.Queue()
    
    def read_output(process, q):
        """Hilo secundario que lee la salida del proceso"""
        for line in iter(process.stdout.readline, ''):
            q.put(line)
        process.stdout.close()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=cwd,
            shell=False,
            encoding='utf-8',
            errors='replace'
        )
        
        t = threading.Thread(target=read_output, args=(process, out_queue))
        t.daemon = True
        t.start()
        
        start_pct, end_pct = progress_range
        range_span = end_pct - start_pct
        
        spinner = ConsoleSpinner("Procesando... por favor espere")
        last_activity = time.time()
        
        while True:
            try:
                line = out_queue.get(timeout=0.1)
                sys.stdout.write("\r" + " " * 60 + "\r")
                
                line_str = line.rstrip()
                if line_str:
                    print(f"[SCRIPT] {line_str}")
                    sys.stdout.flush()
                    
                    if progress_callback and "[PROGRESS]" in line_str:
                        try:
                            parts = line_str.split("[PROGRESS]")
                            if len(parts) > 1:
                                number_part = parts[1].strip().split()[0]
                                script_percent = float(number_part)
                                global_percent = start_pct + (script_percent / 100.0 * range_span)
                                global_percent = max(start_pct, min(end_pct, global_percent))
                                progress_callback(int(global_percent))
                        except ValueError:
                            pass
                
                last_activity = time.time()
                
            except queue.Empty:
                if process.poll() is not None and out_queue.empty():
                    break
                
                sys.stdout.write(spinner.next_frame())
                sys.stdout.flush()
        
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        
        t.join(timeout=1)
        return process.returncode
        
    except Exception as e:
        print(f"\n[ERROR] Error ejecutando comando: {e}")
        return 1

# ============================================================================
# IMPORTACIONES Y CLASE PRINCIPAL
# ============================================================================

try:
    from .tile_processor import EfficientTileProcessor
    from .memory_optimizer import analyze_image_processing, calculate_optimal_tile_size, estimate_tile_memory_usage_mb, get_available_memory_mb
except ImportError:
    try:
        from tile_processor import EfficientTileProcessor
        from memory_optimizer import analyze_image_processing, calculate_optimal_tile_size, estimate_tile_memory_usage_mb, get_available_memory_mb
    except ImportError:
        print("Advertencia: No se pudieron importar algunos módulos de optimización")
        
        def analyze_image_processing(image_path):
            return {'error': 'Módulo de optimización no disponible'}
        
        def calculate_optimal_tile_size(*args, **kwargs):
            return (512, 512)
        
        def estimate_tile_memory_usage_mb(*args, **kwargs):
            return 200.0
        
        def get_available_memory_mb():
            return 4096.0

logger = logging.getLogger(__name__)

class PalmProcessor:
    def __init__(self, config_path=None):
        self.logger = self._setup_logger()
        
        if config_path is None:
            config_path = self.find_config_file()
        self.config = self.load_config(config_path)
        self.setup_paths()
        
        self.log("DIAGNOSTICO - Configuración de modelos:")
        models_config = self.config.get("models", {})
        self.log(f"    Models config: {models_config}")
        
        if "segmentacion" in models_config:
            seg_path = models_config["segmentacion"]
            exists = os.path.exists(seg_path)
            self.log(f"    Modelo segmentación: {seg_path} - {'EXISTE' if exists else 'NO EXISTE'}")
        
        if "instancias" in models_config:
            inst_path = models_config["instancias"]
            exists = os.path.exists(inst_path)
            self.log(f"    Modelo instancias: {inst_path} - {'EXISTE' if exists else 'NO EXISTE'}")
        
        self.tile_processor = EfficientTileProcessor(self.config)
        
        self.model_overhead_mb = 400
        self.ram_safety_threshold = 0.7
        self.min_tile_size = (128, 128)
        self.max_tile_size = (1024, 1024)

    def _get_dynamic_resources(self):
        """MODO 128GB: Configuración EXTREMA forzada manualmente."""
        self.log(f"⚡ MODO ULTRA RAM (HARDCODED) - FORZANDO MAXIMA POTENCIA")
        target_mb = 90000
        batch_size = 128
        self.log(f"   ► RAM Forzada: {target_mb} MB")
        self.log(f"   ► Batch Size Forzado: {batch_size}")
        return target_mb, batch_size

    def _setup_logger(self):
        """Configurar logger para PalmProcessor"""
        import logging
        logger = logging.getLogger('PalmProcessor')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('[PalmProcessor] %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            logger.propagate = False
        return logger
    
    def log(self, message):
        """Método de logging unificado"""
        self.logger.info(message)

    def get_image_dimensions(self, image_path):
        """Obtener dimensiones de una imagen usando GDAL o rasterio"""
        try:
            from osgeo import gdal
            dataset = gdal.Open(image_path)
            if dataset:
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                dataset = None
                return f"{width}x{height}"
        except:
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    width = src.width
                    height = src.height
                    return f"{width}x{height}"
            except:
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        width, height = img.size
                        return f"{width}x{height}"
                except:
                    pass
        return "Desconocido"
        
        

    def find_config_file(self):
        """Buscar config.json en ubicaciones posibles"""
        possible_paths = [
            "config.json",
            "../config.json",
            "./config.json",
            os.path.join(os.path.dirname(__file__), "..", "config.json"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.log(f"Configuracion encontrada en: {path}")
                return path
                
        default_config = self.get_default_config()
        with open("config.json", "w") as f:
            json.dump(default_config, f, indent=4)
        self.log("Configuracion por defecto creada en: config.json")
        return "config.json"
        
    def load_config(self, config_path):
        """Cargar configuración desde JSON"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.log(f"Configuracion cargada exitosamente desde: {config_path}")
                return config
        except Exception as e:
            self.log(f"Error cargando configuracion: {e}")
            return self.get_default_config()
            
    def get_default_config(self):
        """Configuración por defecto"""
        return {
            "scripts": {
                "segmentacion": "scripts/segmentacion.py",
                "instancias": "scripts/instancias.py",
                "segmentacion_tiles": "scripts/process_with_tiles.py",
                "instancias_tiles": "scripts/instancias_tiles.py"
            },
            "parameters": {
                "segmentacion": {
                    "window_radius": 256,
                    "internal_window_radius": 192,
                    "scaling": "normalize"
                },
                "instancias": {
                    "window_radius": 350,
                    "internal_window_radius": 262,
                    "scaling": "none"
                },
                "segmentacion_tiles": {
                    "tile_size": 512,
                    "overlap": 64,
                    "scaling": "normalize"
                },
                "instancias_tiles": {
                    "window_radius": 350,
                    "target_memory_mb": 4096
                }
            },
            "output": {
                "directory": "output",
                "save_images": True,
                "save_logs": True
            },
            "optimization": {
                "memory_management": {
                    "low_ram_mode": False,
                    "max_preview_size": 2000,
                    "tile_size": 512
                },
                "performance": {
                    "enable_downsampling": True,
                    "downsampling_factor": 4
                }
            }
        }
        
    def setup_paths(self):
        """Configurar rutas y crear directorios necesarios"""
        base_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
        
        for script_type in ["segmentacion", "instancias", "segmentacion_tiles", "instancias_tiles"]:
            if script_type in self.config["scripts"]:
                script_path = self.config["scripts"][script_type]
                if not os.path.isabs(script_path):
                    self.config["scripts"][script_type] = os.path.join(base_dir, script_path)
        
        output_dir = self.config["output"]["directory"]
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(base_dir, output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        self.config["output"]["directory"] = output_dir
        
        self.log(f"Directorio de salida: {output_dir}")

    # ===================================================================
    # NUEVO: FUNCIONES PARA DIAGNÓSTICO Y LECTURA ROBUSTA DE CSV
    # ===================================================================

    def _analyze_dataframe(self, df, csv_path):
        """Analizar un DataFrame para extraer estadísticas de especies"""
        try:
            total_rows = len(df)
            
            # ESTRATEGIA 1: Buscar columna de especie por nombre común
            especie_col = None
            col_names_lower = [str(col).lower() for col in df.columns]
            
            patterns = ['especie', 'species', 'class', 'clase', 'tipo', 'type', 'categoria']
            
            for pattern in patterns:
                for i, col_name in enumerate(col_names_lower):
                    if pattern in col_name:
                        especie_col = df.columns[i]
                        self.log(f"   ✓ Columna de especie encontrada: '{especie_col}' (patrón: '{pattern}')")
                        break
                if especie_col:
                    break
            
            # ESTRATEGIA 2: Si no encontramos, buscar columna con valores de especies conocidas
            if not especie_col:
                species_values = ['mauritia', 'euterpe', 'oenocarpus', 'flexuosa', 'precatoria', 'bataua']
                for col in df.columns:
                    if df[col].dtype == 'object':
                        sample_values = df[col].dropna().astype(str).str.lower()
                        for species in species_values:
                            if sample_values.str.contains(species).any():
                                especie_col = col
                                self.log(f"   ✓ Columna de especie inferida: '{col}' (contiene '{species}')")
                                break
                    if especie_col:
                        break
            
            # ESTRATEGIA 3: Contar basado en valores encontrados
            if especie_col:
                df['especie_norm'] = df[especie_col].astype(str).str.lower().str.strip()
                
                mau_count = len(df[df['especie_norm'].str.contains('mauritia|flexuosa')])
                eut_count = len(df[df['especie_norm'].str.contains('euterpe|precatoria')])
                oeno_count = len(df[df['especie_norm'].str.contains('oenocarpus|bataua')])
                
                other_count = total_rows - (mau_count + eut_count + oeno_count)
                
                if other_count > 0:
                    self.log(f"   ⚠️ {other_count} objetos no clasificados como especies conocidas")
                
                return f"Total: {total_rows} palmeras (Mauritia: {mau_count}, Euterpe: {eut_count}, Oenocarpus: {oeno_count})"
            
            # ESTRATEGIA 4: Si no hay columna de especie, contar filas totales
            else:
                self.log(f"   ⚠️ No se identificó columna de especie. Columnas disponibles: {list(df.columns)}")
                return f"Total: {total_rows} objetos detectados (especies no especificadas)"
                
        except Exception as e:
            self.log(f"   Error analizando DataFrame: {e}")
            return None

    def _extract_stats_from_script_logs(self, output_dir, base_name):
        """Extraer estadísticas de archivos de log generados por el script"""
        try:
            import re
            
            log_files = []
            patterns = [
                f"{base_name}*.txt",
                f"{base_name}*.log",
                "*.log",
                "*reporte*.txt",
                "*resultados*.txt"
            ]
            
            for pattern in patterns:
                log_files.extend(glob.glob(os.path.join(output_dir, pattern)))
            
            self.log(f"   Buscando en {len(log_files)} archivos de log...")
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    patterns = [
                        r"Mauritia flexuosa:\s*(\d+)",
                        r"Euterpe precatoria:\s*(\d+)",
                        r"Oenocarpus bataua:\s*(\d+)",
                        r"TOTAL:\s*(\d+)",
                        r"Total:\s*(\d+)",
                        r"Total.*?(\d+).*?palmeras"
                    ]
                    
                    results = {}
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            if pattern.startswith("Mauritia"):
                                results['mauritia'] = int(match)
                            elif pattern.startswith("Euterpe"):
                                results['euterpe'] = int(match)
                            elif pattern.startswith("Oenocarpus"):
                                results['oenocarpus'] = int(match)
                            elif 'TOTAL' in pattern or 'Total:' in pattern:
                                results['total'] = int(match)
                    
                    if results:
                        mau = results.get('mauritia', 0)
                        eut = results.get('euterpe', 0)
                        oeno = results.get('oenocarpus', 0)
                        total = results.get('total', mau + eut + oeno)
                        
                        return f"Total: {total} palmeras (Mauritia: {mau}, Euterpe: {eut}, Oenocarpus: {oeno})"
                        
                except Exception as e:
                    continue
            
            return None
        except Exception as e:
            self.log(f"   Error extrayendo estadísticas de logs: {e}")
            return None

    def _read_final_statistics(self, base_name, output_dir):
        """
        FUNCIÓN CORREGIDA: Lee las estadísticas priorizando el CSV de resumen,
        luego el CSV de atributos, y solo como último recurso intenta logs.
        """
        try:
            self.log("=" * 70)
            self.log("🔍 INICIANDO DIAGNÓSTICO DE ESTADÍSTICAS (LÓGICA HÍBRIDA CORREGIDA)")
            self.log("=" * 70)
            
            output_dir = os.path.normpath(output_dir)
            
            # 1. PRIORIDAD MÁXIMA: CSV de resumen (el que tiene ESPECIE,CONTEO)
            summary_csv = os.path.join(output_dir, f"{base_name}_predicted_summary.csv")
            
            if os.path.exists(summary_csv):
                self.log(f"✅ Encontrado CSV de resumen: {summary_csv}")
                df = pd.read_csv(summary_csv)
                self.log(f"   Columnas: {list(df.columns)}")
                
                # Normalizar nombres de columnas
                df.columns = [c.strip().upper() for c in df.columns]
                
                mau = df.loc[df['ESPECIE'].str.contains('MAURITIA', case=False, na=False), 'CONTEO'].iloc[0] if not df[df['ESPECIE'].str.contains('MAURITIA', case=False, na=False)].empty else 0
                eut = df.loc[df['ESPECIE'].str.contains('EUTERPE', case=False, na=False), 'CONTEO'].iloc[0] if not df[df['ESPECIE'].str.contains('EUTERPE', case=False, na=False)].empty else 0
                oeno = df.loc[df['ESPECIE'].str.contains('OENOCARPUS', case=False, na=False), 'CONTEO'].iloc[0] if not df[df['ESPECIE'].str.contains('OENOCARPUS', case=False, na=False)].empty else 0
                
                total_row = df[df['ESPECIE'].str.contains('TOTAL', case=False, na=False)]
                total = int(total_row['CONTEO'].iloc[0]) if not total_row.empty else (mau + eut + oeno)
                
                result_str = f"Total: {total} palmeras (Mauritia: {mau}, Euterpe: {eut}, Oenocarpus: {oeno})"
                self.log(f"📊 {result_str}")
                return result_str

            # 2. FALLBACK: CSV de atributos detallados (una fila por palmera)
            atributos_csv = os.path.join(output_dir, f"{base_name}_atributos.csv")
            if os.path.exists(atributos_csv):
                self.log(f"⚠️ CSV resumen no encontrado, usando atributos detallados: {atributos_csv}")
                df = pd.read_csv(atributos_csv)
                self.log(f"   Filas detectadas: {len(df)}")
                
                if 'ESPECIE' in df.columns:
                    df['ESPECIE_NORM'] = df['ESPECIE'].astype(str).str.lower()
                    mau = len(df[df['ESPECIE_NORM'].str.contains('mauritia|flexuosa')])
                    eut = len(df[df['ESPECIE_NORM'].str.contains('euterpe|precatoria')])
                    oeno = len(df[df['ESPECIE_NORM'].str.contains('oenocarpus|bataua')])
                    total = len(df)
                    result_str = f"Total: {total} palmeras (Mauritia: {mau}, Euterpe: {eut}, Oenocarpus: {oeno})"
                    self.log(f"📊 {result_str}")
                    return result_str
                else:
                    self.log("   No se encontró columna ESPECIE")
                    return f"Total detectado: {len(df)} objetos (sin clasificación por especie)"

            # 3. ÚLTIMO RECURSO: intentar leer logs (mejorado, pero solo si no hay CSV)
            self.log("⚠️ Ningún CSV encontrado, intentando leer logs...")
            stats_from_logs = self._extract_stats_from_script_logs(output_dir, base_name)
            if stats_from_logs:
                self.log(f"📊 {stats_from_logs}")
                return stats_from_logs
                
            return "No se pudieron leer las estadísticas finales"

        except Exception as e:
            self.log(f"❌ Error leyendo estadísticas: {e}")
            import traceback
            self.log(traceback.format_exc())
            return "Error en lectura de estadísticas"

    # ===================================================================
    # MÉTODO DE PROCESAMIENTO CON TILES AVANZADO (MODO OPTIMIZADO)
    # ===================================================================

    def _process_with_advanced_tiles(self, image_path: str, progress_callback=None) -> str:
        """MODO OPTIMIZADO: Usa los scripts de tiles avanzados"""
        try:
            self.log("=" * 70)
            self.log("🚀 MODO OPTIMIZADO - USANDO SCRIPTS DE TILES AVANZADOS")
            self.log("=" * 70)
            
            memoria_turbo, batch_turbo = self._get_dynamic_resources()
            
            # CORRECCIÓN CRÍTICA: Usar splitext para nombre base
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Normalizar ruta de salida
            output_dir = os.path.normpath(self.config["output"]["directory"])
            
            if progress_callback:
                progress_callback(10)
            
            # 1. SEGMENTACIÓN POR TILES
            self.log("🧩 Ejecutando segmentación por tiles optimizada...")
            
            seg_tiles_script = self.config["scripts"]["segmentacion_tiles"]
            
            if not os.path.exists(seg_tiles_script):
                raise FileNotFoundError(f"Script de segmentación por tiles no encontrado: {seg_tiles_script}")
            
            seg_tiles_params = self.config["parameters"]["segmentacion_tiles"]
            
            cmd_seg = [
                sys.executable,
                seg_tiles_script,
                image_path,
                "--model", "models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx",
                "--output", output_dir,
                "--tile_size", str(seg_tiles_params["tile_size"]),
                "--overlap", str(seg_tiles_params["overlap"]),
                "--scaling", seg_tiles_params["scaling"],
                "--max_batch_size", str(batch_turbo),
                "--min_batch_size", str(batch_turbo),
                "--memory_safety_margin", "0.1",
            ]
            
            if batch_turbo > 16:
                cmd_seg.extend(["--max_batch_size", "1024"])
                cmd_seg.extend(["--min_batch_size", "512"])
                cmd_seg.extend(["--memory_safety_margin", "0.005"])
                cmd_seg.extend(["--prefetch_tiles", "10000"])
            
            project_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
            
            self.log(f"📊 Parámetros segmentación por tiles (MODO 128):")
            self.log(f"    • Tile size: {seg_tiles_params['tile_size']}")
            self.log(f"    • Overlap: {seg_tiles_params['overlap']}")
            self.log(f"    • Escalado: {seg_tiles_params['scaling']}")
            self.log(f"    • Batch Size: {batch_turbo} tiles simultáneos")
            
            if progress_callback:
                progress_callback(30)
            
            returncode_seg = run_command_with_terminal_output(
                cmd_seg,
                cwd=project_dir,
                progress_callback=progress_callback,
                progress_range=(5, 60)
            )
            
            if returncode_seg != 0:
                raise Exception(f"Error en segmentación por tiles (código {returncode_seg})")
            
            self.log("✅ Segmentación por tiles completada")
            
            if progress_callback:
                progress_callback(60)
            
            # 2. BUSCAR MÁSCARA GENERADA
            mask_path = os.path.join(output_dir, "segmentacion_batch.tif")
            
            if not os.path.exists(mask_path):
                mask_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.tif")
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Máscara no encontrada: {mask_path}")
            
            self.log(f"✓ Máscara encontrada: {mask_path}")
            
            # 3. INSTANCIAS POR TILES
            self.log("🔍 Ejecutando conteo de instancias por tiles...")
            
            inst_tiles_script = self.config["scripts"]["instancias_tiles"]
            
            if not os.path.exists(inst_tiles_script):
                raise FileNotFoundError(f"Script de instancias por tiles no encontrado: {inst_tiles_script}")
            
            inst_tiles_params = self.config["parameters"]["instancias_tiles"]
            
            cmd_inst = [
                sys.executable,
                inst_tiles_script,
                image_path,
                mask_path,
                "--window_radius", str(inst_tiles_params.get("window_radius", 350)),
                "--output_dir", output_dir,
                "--model_path", "models/model_converted.onnx"
            ]
            
            self.log(f"📊 Parámetros instancias por tiles:")
            self.log(f"    • Window radius: {inst_tiles_params.get('window_radius', 350)}")
            
            if progress_callback:
                progress_callback(80)
            
            returncode_inst = run_command_with_terminal_output(
                cmd_inst,
                cwd=project_dir,
                progress_callback=progress_callback,
                progress_range=(65, 95)
            )
            
            if returncode_inst != 0:
                raise Exception(f"Error en conteo por tiles (código {returncode_inst})")
            
            self.log("✅ Conteo por tiles completado")
            
            # 4. LEER ESTADÍSTICAS CON LA NUEVA FUNCIÓN ROBUSTA
            if progress_callback:
                progress_callback(95)
            
            stats = self._read_final_statistics(base_name, output_dir)
            
            if progress_callback:
                progress_callback(100)
            
            self.log("=" * 70)
            self.log("🎉 MODO 128 COMPLETADO EXITOSAMENTE")
            self.log(f"📊 {stats}")
            self.log("=" * 70)
            
            return f"Modo 128 completado. {stats}"
            
        except Exception as e:
            self.log(f"❌ Error en modo 128 con tiles avanzados: {e}")
            import traceback
            self.log(f"📋 Traceback: {traceback.format_exc()}")
            
            try:
                self.log("🔄 Intentando fallback a modo normal de tiles...")
                return self._process_with_scripts_forced_tiles(image_path, progress_callback)
            except Exception as e2:
                self.log(f"❌ Fallback también falló: {e2}")
                raise

    # ===================================================================
    # MÉTODO PRINCIPAL
    # ===================================================================

    def process_image(self, image_path: str, force_tiling: bool = False,
                      use_optimized_tiles: bool = False, progress_callback=None) -> str:
        """Ejecutar el pipeline completo - VERSIÓN MEJORADA"""
        try:
            self.log(f"INICIANDO PROCESAMIENTO: {os.path.basename(image_path)}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"La imagen no existe: {image_path}")

            if progress_callback:
                progress_callback(1)
            
            if use_optimized_tiles:
                self.log("🎯 MODO 128 ACTIVADO - Usando scripts de tiles avanzados")
                return self._process_with_advanced_tiles(image_path, progress_callback)
            
            low_ram_mode = self.config.get("optimization", {}).get("memory_management", {}).get("low_ram_mode", False)
            
            if progress_callback:
                progress_callback(3)
            
            if low_ram_mode or force_tiling:
                self.log("🔋 MODO BAJA MEMORIA DETECTADO - Usando scripts originales")
                return self._process_with_scripts_forced_tiles(image_path, progress_callback)

            if progress_callback:
                progress_callback(5)
            
            memory_analysis = self._analyze_memory_requirements_precise(image_path)
            
            if progress_callback:
                progress_callback(8)
            
            if memory_analysis['requires_tiling']:
                self.log("🧱 ACTIVANDO PROCESAMIENTO POR TILES OPTIMIZADO")
                return self._process_with_optimized_tiles(image_path, memory_analysis, progress_callback)
            else:
                self.log("📏 USANDO PROCESAMIENTO DIRECTO (imagen pequeña)")
                return self._process_directly(image_path, progress_callback)
                
        except Exception as e:
            self.log(f"❌ Error en procesamiento: {e}")
            raise

    # ===================================================================
    # MÉTODOS EXISTENTES (mantenidos para compatibilidad)
    # ===================================================================

    def analyze_memory_requirements(self, image_path: str) -> dict:
        """Solo analiza los requisitos de memoria sin procesar la imagen"""
        try:
            self.log("🔍 Realizando análisis de requisitos de memoria...")
            
            analysis = self._analyze_memory_requirements_precise(image_path)
            
            if 'error' in analysis:
                return {
                    'strategy': {'reason': analysis['reason']},
                    'recommendation': analysis['reason'],
                    'should_use_tiles': True,
                    'optimal_tile_size': (512, 512),
                    'error': analysis['reason']
                }
            
            return {
                'strategy': analysis,
                'recommendation': analysis.get('reason', 'Análisis completado'),
                'should_use_tiles': analysis.get('requires_tiling', True),
                'optimal_tile_size': analysis.get('optimal_tile_size', (512, 512)),
                'memory_required_mb': analysis.get('image_memory_mb', 0),
                'memory_available_mb': analysis.get('available_memory_mb', 0)
            }
            
        except Exception as e:
            error_msg = f"Error en análisis de memoria: {str(e)}"
            self.log(f"❌ {error_msg}")
            return {
                'strategy': {'reason': error_msg},
                'recommendation': error_msg,
                'should_use_tiles': True,
                'optimal_tile_size': (512, 512),
                'error': error_msg
            }

    def _analyze_memory_requirements_precise(self, image_path: str) -> dict:
        """Analisis PRECISO de memoria usando el memory_optimizer"""
        try:
            if self.config.get("optimization", {}).get("memory_management", {}).get("low_ram_mode", False):
                self.log("MODO BAJA MEMORIA DETECTADO - Forzando uso de tiles")
                return {
                    'requires_tiling': True,
                    'optimal_tile_size': (512, 512),
                    'tile_memory_mb': 200.0,
                    'image_memory_mb': 0,
                    'available_memory_mb': 0,
                    'reason': 'Modo baja memoria activado - Forzando tiles'
                }
            
            analysis = analyze_image_processing(image_path)
            
            if 'error' in analysis:
                self.log(f"Error en análisis de memoria: {analysis['error']}")
                return {'requires_tiling': True, 'reason': 'Error en análisis'}
            
            image_info = analysis['image_info']
            system_info = analysis['system_info']
            
            requires_tiling = analysis['strategy'] == "procesamiento_por_tiles"
            optimal_tile_size = analysis['optimal_tile_size'] or (512, 512)
            
            self.log(f"ANALISIS PRECISO COMPLETADO:")
            self.log(f"    • Imagen: {image_info['memory_mb']:.1f} MB")
            self.log(f"    • Memoria disponible: {system_info['available_memory_mb']:.1f} MB")
            self.log(f"    • Tile óptimo: {optimal_tile_size}")
            self.log(f"    • Usar tiles: {requires_tiling}")
            self.log(f"    • Razón: {analysis['recommendation']}")
            
            return {
                'requires_tiling': requires_tiling,
                'optimal_tile_size': optimal_tile_size,
                'tile_memory_mb': analysis['tile_memory_estimate_mb'],
                'image_memory_mb': image_info['memory_mb'],
                'available_memory_mb': system_info['available_memory_mb'],
                'reason': analysis['recommendation']
            }
            
        except Exception as e:
            self.log(f"Error en análisis preciso: {e}")
            return {'requires_tiling': True, 'reason': f'Error: {str(e)}'}

    def _process_with_optimized_tiles(self, image_path: str, memory_analysis: dict, progress_callback=None) -> str:
        """Procesamiento usando el sistema de tiles eficiente"""
        try:
            base_name = os.path.basename(image_path).split('.')[0]
            output_dir = self.config["output"]["directory"]
            
            self.log(f"INICIANDO PROCESAMIENTO POR TILES OPTIMIZADO")
            self.log(f"    • Tile size: {memory_analysis['optimal_tile_size']}")
            self.log(f"    • Memoria/tile: {memory_analysis['tile_memory_mb']:.1f} MB")
            
            if progress_callback:
                progress_callback(10)
            
            seg_output_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.tif")
            
            if progress_callback:
                progress_callback(15)
            
            seg_success = self._run_segmentation_with_tiles(
                image_path,
                seg_output_path,
                memory_analysis['optimal_tile_size'],
                progress_callback
            )
            
            if not seg_success:
                return "Error en segmentación por tiles"
            
            if progress_callback:
                progress_callback(60)
            
            if progress_callback:
                progress_callback(65)
            
            png_output_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.png")
            self._convert_tiff_to_png(seg_output_path, png_output_path)
            
            if progress_callback:
                progress_callback(70)
            
            inst_result = self.run_instances(image_path)
            
            if progress_callback:
                progress_callback(95)
            
            stats = self._read_final_statistics(base_name, output_dir)
            
            self.log("PROCESAMIENTO POR TILES COMPLETADO EXITOSAMENTE")
            return f"Procesamiento por tiles exitoso. {stats}"
            
        except Exception as e:
            self.log(f"Error en procesamiento por tiles: {e}")
            raise

    def _process_with_scripts_forced_tiles(self, image_path: str, progress_callback=None) -> str:
        """Procesamiento por tiles usando scripts originales - MODO BAJA MEMORIA"""
        try:
            self.log("MODO BAJA MEMORIA: Usando scripts originales forzados")
            
            if progress_callback:
                progress_callback(5)
            
            base_name = os.path.basename(image_path).split('.')[0]
            output_dir = self.config["output"]["directory"]
            
            if progress_callback:
                progress_callback(15)
            
            self.log("Ejecutando segmentación con scripts originales...")
            seg_result = self.run_segmentation(image_path)
            self.log("Segmentación completada")
            
            if progress_callback:
                progress_callback(50)
            
            tiff_seg_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.tif")
            png_seg_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.png")
            
            if os.path.exists(tiff_seg_path) and not os.path.exists(png_seg_path):
                self._convert_tiff_to_png(tiff_seg_path, png_seg_path)
            
            if progress_callback:
                progress_callback(70)
            
            self.log("Ejecutando conteo de instancias con scripts originales...")
            inst_result = self.run_instances(image_path)
            self.log("Conteo de instancias completado")
            
            if progress_callback:
                progress_callback(95)
            
            stats = self._read_final_statistics(base_name, output_dir)
            
            self.log("PROCESAMIENTO EN MODO BAJA MEMORIA COMPLETADO EXITOSAMENTE")
            return f"Procesamiento en modo baja memoria exitoso. {stats}"
            
        except Exception as e:
            self.log(f"Error en modo baja memoria: {e}")
            raise

    def _run_segmentation_with_tiles(self, image_path: str, output_path: str,
                                   tile_size: tuple, progress_callback=None) -> bool:
        """Ejecuta la segmentación usando el sistema de tiles"""
        try:
            def segmentation_callback(tile_data, tile_info):
                try:
                    processed_tile = self.tile_processor.process_tile_with_model(tile_data)
                    
                    if processed_tile is not None:
                        self.log(f"Tile {tile_info['id']} procesado")
                    else:
                        self.log(f"Tile {tile_info['id']} retornó None")
                        height, width = tile_data.shape[:2]
                        processed_tile = np.zeros((height, width), dtype=np.float32)
                    
                    return processed_tile
                    
                except Exception as e:
                    self.log(f"Error en tile {tile_info['id']}: {e}")
                    height, width = tile_data.shape[:2]
                    return np.zeros((height, width), dtype=np.float32)
            
            def tile_progress_callback(progress, message):
                if progress_callback:
                    mapped_progress = 20 + (progress * 0.4)
                    progress_callback(int(mapped_progress))
            
            success = self.tile_processor.process_image_by_tiles(
                image_path=image_path,
                output_path=output_path,
                processing_callback=segmentation_callback,
                tile_size=tile_size,
                overlap=32,
                batch_size=2,
                progress_callback=tile_progress_callback
            )
            
            return success
            
        except Exception as e:
            self.log(f"Error en segmentación con tiles: {e}")
            return False

    def _convert_tiff_to_png(self, tiff_path: str, png_path: str):
        """Convierte el TIFF de segmentación a PNG para visualización"""
        try:
            from osgeo import gdal
            import numpy as np
            from PIL import Image
            
            dataset = gdal.Open(tiff_path)
            if dataset:
                seg_data = dataset.ReadAsArray()
                
                if seg_data.dtype == np.float32:
                    seg_normalized = ((seg_data - seg_data.min()) /
                                    (seg_data.max() - seg_data.min() + 1e-8) * 255).astype(np.uint8)
                else:
                    seg_normalized = seg_data.astype(np.uint8)
                
                seg_image = Image.fromarray(seg_normalized)
                seg_image.save(png_path)
                
                self.log(f"PNG generado: {png_path}")
                dataset = None
            else:
                self.log(f"No se pudo abrir el TIFF: {tiff_path}")
                
        except Exception as e:
            self.log(f"Error convirtiendo TIFF a PNG: {e}")

    def _process_directly(self, image_path: str, progress_callback=None) -> str:
        """Método original de procesamiento para imágenes pequeñas"""
        self.log("Usando procesamiento directo para consistencia de resultados...")
        
        if progress_callback:
            progress_callback(10)
        
        self.log("Iniciando segmentacion...")
        seg_result = self.run_segmentation(image_path)
        self.log("Segmentacion completada")
        
        if progress_callback:
            progress_callback(60)
        
        self.log("Iniciando conteo de instancias...")
        inst_result = self.run_instances(image_path)
        self.log("Conteo de instancias completado")
        
        base_name = os.path.basename(image_path).split('.')[0]
        output_dir = self.config["output"]["directory"]
        stats = self._read_final_statistics(base_name, output_dir)
        
        if progress_callback:
            progress_callback(100)
        
        return f"Procesamiento exitoso. {stats}"

    def run_segmentation(self, image_path):
        """Ejecutar script de segmentación mostrando output en terminal"""
        seg_script = self.config["scripts"]["segmentacion"]
        
        if not os.path.exists(seg_script):
            raise FileNotFoundError(f"Script no encontrado: {seg_script}")
            
        seg_params = self.config["parameters"]["segmentacion"]
        window_radius = seg_params["window_radius"]
        internal_window_radius = seg_params.get("internal_window_radius", int(round(window_radius * 0.75)))
        scaling = seg_params.get("scaling", "normalize")
        
        cmd = [
            sys.executable,
            seg_script,
            image_path,
            "--model", "models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx",
            "--output", self.config["output"]["directory"],
            "--window_radius", str(window_radius),
            "--scaling", scaling
        ]
        
        if "internal_window_radius" in seg_params:
            cmd.extend(["--internal_window_radius", str(internal_window_radius)])
        
        project_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
        
        self.log(f"\n{'='*60}")
        self.log(f"INICIANDO SEGMENTACIÓN - VER TERMINAL PARA DETALLES")
        self.log(f"{'='*60}\n")
        
        returncode = run_command_with_terminal_output(cmd, cwd=project_dir)
        
        if returncode != 0:
            raise Exception(f"Error en segmentación (código {returncode})")
            
        return "Segmentación completada - Revisa la terminal para detalles"
        
    def run_instances(self, image_path):
        """Ejecutar script de conteo de instancias mostrando output en terminal"""
        inst_script = self.config["scripts"]["instancias"]
        
        if not os.path.exists(inst_script):
            raise FileNotFoundError(f"Script no encontrado: {inst_script}")
        
        inst_params = self.config["parameters"]["instancias"]
        window_radius = inst_params["window_radius"]
        
        base_name = os.path.basename(image_path).split('.')[0]
        mask_path = os.path.join(self.config["output"]["directory"], f"{base_name}_balanced_argmax.tif")
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Máscara no encontrada: {mask_path}")
        
        cmd = [
            sys.executable,
            inst_script,
            image_path,
            "--window_radius", str(window_radius),
            "--mask_path", mask_path,
            "--model_path", "models/model_converted.onnx",
            "--output_dir", self.config["output"]["directory"]
        ]
        
        self.log(f"\n{'='*60}")
        self.log(f"INICIANDO CONTEO DE INSTANCIAS - VER TERMINAL PARA DETALLES")
        self.log(f"{'='*60}\n")
        
        project_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
        returncode = run_command_with_terminal_output(cmd, cwd=project_dir)
        
        if returncode != 0:
            raise Exception(f"Error en conteo (código {returncode})")
            
        return "Conteo de instancias completado - Revisa la terminal para detalles"
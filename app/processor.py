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


def run_command_with_terminal_output(cmd, cwd=None):
    """Ejecuta comando y muestra output en terminal en tiempo real"""
    print(f"\n[TERMINAL] Ejecutando comando: {' '.join(cmd)}")
    sys.stdout.flush()  # Forzar salida inmediata
    
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
        
        # Capturar y mostrar output en tiempo real
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line:
                print(f"[SCRIPT] {line.rstrip()}")
                sys.stdout.flush()
                output_lines.append(line)
            elif process.poll() is not None:
                break
        
        # Capturar cualquier salida restante
        remaining = process.stdout.read()
        if remaining:
            print(f"[SCRIPT] {remaining.rstrip()}")
            sys.stdout.flush()
            output_lines.append(remaining)
        
        process.wait()
        return process.returncode
        
    except Exception as e:
        print(f"[ERROR] Error ejecutando comando: {e}")
        return 1
    
# Manejo robusto de importaciones
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
        # Inicializar logger primero
        self.logger = self._setup_logger()
        
        if config_path is None:
            config_path = self.find_config_file()
        self.config = self.load_config(config_path)
        self.setup_paths()
        
        # DIAGNOSTICO: Verificar configuración de modelos
        self.log("DIAGNOSTICO - Configuración de modelos:")
        models_config = self.config.get("models", {})
        self.log(f"   Models config: {models_config}")
        
        # Verificar si los archivos de modelos existen
        if "segmentacion" in models_config:
            seg_path = models_config["segmentacion"]
            exists = os.path.exists(seg_path)
            self.log(f"   Modelo segmentación: {seg_path} - {'EXISTE' if exists else 'NO EXISTE'}")
        
        if "instancias" in models_config:
            inst_path = models_config["instancias"]
            exists = os.path.exists(inst_path)
            self.log(f"   Modelo instancias: {inst_path} - {'EXISTE' if exists else 'NO EXISTE'}")
        
        # Inicializar tile processor
        self.tile_processor = EfficientTileProcessor(self.config)
        
        # PARAMETROS DEL DECISOR
        self.model_overhead_mb = 400
        self.ram_safety_threshold = 0.7
        self.min_tile_size = (128, 128)
        self.max_tile_size = (1024, 1024)

    def _setup_logger(self):
        """Configurar logger para PalmProcessor"""
        import logging
        
        # Crear logger específico para PalmProcessor
        logger = logging.getLogger('PalmProcessor')
        logger.setLevel(logging.INFO)
        
        # Solo agregar handlers si no existen
        if not logger.handlers:
            # Handler para consola
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('[PalmProcessor] %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Evitar propagación al logger root
            logger.propagate = False
        
        return logger
    
    def log(self, message):
        """Método de logging unificado"""
        self.logger.info(message)

    def get_image_dimensions(self, image_path):
        """Obtener dimensiones de una imagen usando GDAL o rasterio"""
        try:
            # Intentar con GDAL primero
            from osgeo import gdal
            dataset = gdal.Open(image_path)
            if dataset:
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                dataset = None
                return f"{width}x{height}"
        except:
            try:
                # Intentar con rasterio
                import rasterio
                with rasterio.open(image_path) as src:
                    width = src.width
                    height = src.height
                    return f"{width}x{height}"
            except:
                try:
                    # Intentar con PIL como último recurso
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
        
        # Asegurar que las rutas de scripts sean absolutas
        for script_type in ["segmentacion", "instancias", "segmentacion_tiles", "instancias_tiles"]:
            if script_type in self.config["scripts"]:
                script_path = self.config["scripts"][script_type]
                if not os.path.isabs(script_path):
                    self.config["scripts"][script_type] = os.path.join(base_dir, script_path)
        
        # Crear directorio de salida
        output_dir = self.config["output"]["directory"]
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(base_dir, output_dir)
            
        os.makedirs(output_dir, exist_ok=True)
        self.config["output"]["directory"] = output_dir
        
        self.log(f"Directorio de salida: {output_dir}")

    # ===================================================================
    # NUEVO MÉTODO: PROCESAMIENTO CON TILES AVANZADO (MODO OPTIMIZADO)
    # ===================================================================

    def _process_with_advanced_tiles(self, image_path: str, progress_callback=None) -> str:
        """
        MODO OPTIMIZADO: Usa los scripts de tiles avanzados process_with_tiles.py e instancias_tiles.py
        """
        try:
            self.log("=" * 70)
            self.log("🚀 MODO OPTIMIZADO - USANDO SCRIPTS DE TILES AVANZADOS")
            self.log("=" * 70)
            
            base_name = os.path.basename(image_path).split('.')[0]
            output_dir = self.config["output"]["directory"]
            
            if progress_callback:
                progress_callback(10, "Iniciando segmentación por tiles avanzada...")
            
            # 1. SEGMENTACIÓN POR TILES (process_with_tiles.py)
            self.log("🧩 Ejecutando segmentación por tiles optimizada...")
            
            seg_tiles_script = self.config["scripts"]["segmentacion_tiles"]
            
            if not os.path.exists(seg_tiles_script):
                raise FileNotFoundError(f"Script de segmentación por tiles no encontrado: {seg_tiles_script}")
            
            # Obtener parámetros de segmentación por tiles
            seg_tiles_params = self.config["parameters"]["segmentacion_tiles"]
            
            cmd_seg = [
                sys.executable,
                seg_tiles_script,
                image_path,
                "--model", "models/deeplab_keras_model_palms_iaa_all_0.003_W.onnx",
                "--output", output_dir,
                "--tile_size", str(seg_tiles_params["tile_size"]),
                "--overlap", str(seg_tiles_params["overlap"]),
                "--scaling", seg_tiles_params["scaling"]
            ]
            
            # Agregar parámetros opcionales si existen
            if "max_batch_size" in seg_tiles_params:
                cmd_seg.extend(["--max_batch_size", str(seg_tiles_params["max_batch_size"])])
            
            if "min_batch_size" in seg_tiles_params:
                cmd_seg.extend(["--min_batch_size", str(seg_tiles_params["min_batch_size"])])
            
            if "memory_safety_margin" in seg_tiles_params:
                cmd_seg.extend(["--memory_safety_margin", str(seg_tiles_params["memory_safety_margin"])])
            
            project_dir = os.path.dirname(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
            
            self.log(f"📊 Parámetros segmentación por tiles:")
            self.log(f"   • Tile size: {seg_tiles_params['tile_size']}")
            self.log(f"   • Overlap: {seg_tiles_params['overlap']}")
            self.log(f"   • Escalado: {seg_tiles_params['scaling']}")
            
            if progress_callback:
                progress_callback(30, "Segmentación por tiles en progreso...")
            
            returncode_seg = run_command_with_terminal_output(cmd_seg, cwd=project_dir)
            
            if returncode_seg != 0:
                raise Exception(f"Error en segmentación por tiles (código {returncode_seg})")
            
            self.log("✅ Segmentación por tiles completada")
            
            if progress_callback:
                progress_callback(60, "Segmentación completada, iniciando conteo...")
            
            # 2. BUSCAR MÁSCARA GENERADA
            # El script process_with_tiles.py genera 'segmentacion_batch.tif'
            mask_path = os.path.join(output_dir, "segmentacion_batch.tif")
            
            # Si no existe, buscar la máscara regular
            if not os.path.exists(mask_path):
                mask_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.tif")
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Máscara no encontrada: {mask_path}")
            
            self.log(f"✓ Máscara encontrada: {mask_path}")
            
            # 3. INSTANCIAS POR TILES (instancias_tiles.py)
            self.log("🔍 Ejecutando conteo de instancias por tiles...")
            
            inst_tiles_script = self.config["scripts"]["instancias_tiles"]
            
            if not os.path.exists(inst_tiles_script):
                raise FileNotFoundError(f"Script de instancias por tiles no encontrado: {inst_tiles_script}")
            
            # Obtener parámetros de instancias por tiles
            inst_tiles_params = self.config["parameters"]["instancias_tiles"]
            
            # Construir comando para instancias_tiles.py
            # NOTA: instancias_tiles.py espera argumentos posicionales: imagen máscara [opciones]
            cmd_inst = [
                sys.executable,
                inst_tiles_script,
                image_path,
                mask_path
            ]
            
            # Agregar parámetros adicionales si existen en el script
            # (Basado en tu script, parece que acepta --target_memory_mb)
            if "target_memory_mb" in inst_tiles_params:
                cmd_inst.extend(["--target_memory_mb", str(inst_tiles_params["target_memory_mb"])])
            
            if "window_radius" in inst_tiles_params:
                cmd_inst.extend(["--window_radius", str(inst_tiles_params["window_radius"])])
            
            # Especificar directorio de salida si el script lo soporta
            cmd_inst.extend(["--output_dir", output_dir])
            
            # Agregar ruta del modelo si es necesario
            cmd_inst.extend(["--model_path", "models/model_converted.onnx"])
            
            self.log(f"📊 Parámetros instancias por tiles:")
            self.log(f"   • Window radius: {inst_tiles_params.get('window_radius', 350)}")
            if "target_memory_mb" in inst_tiles_params:
                self.log(f"   • Memoria objetivo: {inst_tiles_params['target_memory_mb']} MB")
            
            if progress_callback:
                progress_callback(80, "Conteo de instancias en progreso...")
            
            returncode_inst = run_command_with_terminal_output(cmd_inst, cwd=project_dir)
            
            if returncode_inst != 0:
                raise Exception(f"Error en conteo por tiles (código {returncode_inst})")
            
            self.log("✅ Conteo por tiles completado")
            
            # 4. LEER ESTADÍSTICAS
            if progress_callback:
                progress_callback(95, "Generando resultados finales...")
            
            # Buscar archivo CSV generado por instancias_tiles.py
            csv_path = os.path.join(output_dir, f"{base_name}_predicted_atributos.csv")
            
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                mau_count = len(df[df['ESPECIE'] == 'Mauritia flexuosa'])
                eut_count = len(df[df['ESPECIE'] == 'Euterpe precatoria'])
                oeno_count = len(df[df['ESPECIE'] == 'Oenocarpus bataua'])
                total_count = mau_count + eut_count + oeno_count
                
                stats = f"Total: {total_count} palmeras (Mauritia: {mau_count}, Euterpe: {eut_count}, Oenocarpus: {oeno_count})"
            else:
                # Intentar leer de otro archivo si existe
                stats = self._read_final_statistics(base_name, output_dir)
            
            if progress_callback:
                progress_callback(100, "✅ Procesamiento optimizado completado")
            
            self.log("=" * 70)
            self.log("🎉 MODO OPTIMIZADO COMPLETADO EXITOSAMENTE")
            self.log(f"📊 {stats}")
            self.log("=" * 70)
            
            return f"Modo optimizado completado. {stats}"
            
        except Exception as e:
            self.log(f"❌ Error en modo optimizado con tiles avanzados: {e}")
            import traceback
            self.log(f"📋 Traceback: {traceback.format_exc()}")
            
            # Fallback a modo normal de tiles
            try:
                self.log("🔄 Intentando fallback a modo normal de tiles...")
                return self._process_with_scripts_forced_tiles(image_path, progress_callback)
            except Exception as e2:
                self.log(f"❌ Fallback también falló: {e2}")
                raise

    # ===================================================================
    # MÉTODO PRINCIPAL MEJORADO - CON MODO OPTIMIZADO CON TILES
    # ===================================================================

    def process_image(self, image_path: str, force_tiling: bool = False, 
                     use_optimized_tiles: bool = False, progress_callback=None) -> str:
        """
        Ejecutar el pipeline completo - VERSIÓN MEJORADA
        """
        try:
            self.log(f"INICIANDO PROCESAMIENTO: {os.path.basename(image_path)}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"La imagen no existe: {image_path}")

            if progress_callback:
                progress_callback(1, "Iniciando procesamiento...")
            
            # NUEVO: Modo optimizado CON TILES AVANZADOS
            if use_optimized_tiles:
                self.log("🎯 MODO OPTIMIZADO ACTIVADO - Usando scripts de tiles avanzados")
                return self._process_with_advanced_tiles(image_path, progress_callback)
            
            # CORRECCIÓN CRÍTICA: Verificar modo baja memoria SOLO si NO es optimizado
            low_ram_mode = self.config.get("optimization", {}).get("memory_management", {}).get("low_ram_mode", False)
            
            if progress_callback:
                progress_callback(3, "Verificando modo de memoria...")
            
            if low_ram_mode or force_tiling:
                self.log("🔋 MODO BAJA MEMORIA DETECTADO - Usando scripts originales")
                return self._process_with_scripts_forced_tiles(image_path, progress_callback)

            # Análisis preciso de memoria para decidir estrategia (solo si no es optimizado ni baja memoria)
            if progress_callback:
                progress_callback(5, "Analizando memoria...")
            
            memory_analysis = self._analyze_memory_requirements_precise(image_path)
            
            if progress_callback:
                progress_callback(8, "Memoria analizada, iniciando procesamiento...")
            
            # Decisión basada en análisis preciso
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
        """
        Solo analiza los requisitos de memoria sin procesar la imagen
        """
        try:
            self.log("🔍 Realizando análisis de requisitos de memoria...")
            
            # Usar el análisis preciso existente
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
        """
        Analisis PRECISO de memoria usando el memory_optimizer
        """
        try:
            # CORRECCIÓN CRÍTICA: Si está en modo baja memoria, FORZAR tiles
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
            
            # Información de la imagen
            image_info = analysis['image_info']
            system_info = analysis['system_info']
            
            requires_tiling = analysis['strategy'] == "procesamiento_por_tiles"
            optimal_tile_size = analysis['optimal_tile_size'] or (512, 512)
            
            self.log(f"ANALISIS PRECISO COMPLETADO:")
            self.log(f"   • Imagen: {image_info['memory_mb']:.1f} MB")
            self.log(f"   • Memoria disponible: {system_info['available_memory_mb']:.1f} MB")
            self.log(f"   • Tile óptimo: {optimal_tile_size}")
            self.log(f"   • Usar tiles: {requires_tiling}")
            self.log(f"   • Razón: {analysis['recommendation']}")
            
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
        """
        Procesamiento usando el sistema de tiles eficiente
        """
        try:
            base_name = os.path.basename(image_path).split('.')[0]
            output_dir = self.config["output"]["directory"]
            
            self.log(f"INICIANDO PROCESAMIENTO POR TILES OPTIMIZADO")
            self.log(f"   • Tile size: {memory_analysis['optimal_tile_size']}")
            self.log(f"   • Memoria/tile: {memory_analysis['tile_memory_mb']:.1f} MB")
            
            if progress_callback:
                progress_callback(10, "Configurando procesamiento por tiles...")
            
            # 1. SEGMENTACIÓN POR TILES
            seg_output_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.tif")
            
            if progress_callback:
                progress_callback(15, "Preparando segmentación por tiles...")
            
            seg_success = self._run_segmentation_with_tiles(
                image_path, 
                seg_output_path, 
                memory_analysis['optimal_tile_size'],
                progress_callback
            )
            
            if not seg_success:
                return "Error en segmentación por tiles"
            
            if progress_callback:
                progress_callback(60, "Segmentación completada, iniciando conteo...")
            
            # 2. CONVERSIÓN A PNG (necesario para el script de instancias)
            if progress_callback:
                progress_callback(65, "Convirtiendo resultados a PNG...")
            
            png_output_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.png")
            self._convert_tiff_to_png(seg_output_path, png_output_path)
            
            # 3. EJECUTAR SCRIPT DE INSTANCIAS (usa la segmentación generada)
            if progress_callback:
                progress_callback(70, "Ejecutando conteo de instancias...")
            
            inst_result = self.run_instances(image_path)
            
            if progress_callback:
                progress_callback(95, "Finalizando procesamiento...")
            
            # 4. GENERAR REPORTE FINAL
            stats = self._read_final_statistics(base_name, output_dir)
            
            self.log("PROCESAMIENTO POR TILES COMPLETADO EXITOSAMENTE")
            return f"Procesamiento por tiles exitoso. {stats}"
            
        except Exception as e:
            self.log(f"Error en procesamiento por tiles: {e}")
            raise

    def _process_with_scripts_forced_tiles(self, image_path: str, progress_callback=None) -> str:
        """
        Procesamiento por tiles PERO usando los scripts originales para máxima compatibilidad
        y mínimo uso de memoria - MODO BAJA MEMORIA
        """
        try:
            self.log("MODO BAJA MEMORIA: Usando scripts originales forzados")
            
            if progress_callback:
                progress_callback(5, "Iniciando modo baja memoria...")
            
            base_name = os.path.basename(image_path).split('.')[0]
            output_dir = self.config["output"]["directory"]
            
            # 1. SEGMENTACIÓN CON SCRIPTS ORIGINALES
            if progress_callback:
                progress_callback(15, "Ejecutando segmentación...")
            
            self.log("Ejecutando segmentación con scripts originales...")
            seg_result = self.run_segmentation(image_path)
            self.log("Segmentación completada")
            
            # 2. CONVERSIÓN A PNG (si es necesario)
            if progress_callback:
                progress_callback(50, "Preparando archivos intermedios...")
            
            tiff_seg_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.tif")
            png_seg_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.png")
            
            # Convertir TIFF a PNG si no existe
            if os.path.exists(tiff_seg_path) and not os.path.exists(png_seg_path):
                self._convert_tiff_to_png(tiff_seg_path, png_seg_path)
            
            # 3. INSTANCIAS CON SCRIPTS ORIGINALES
            if progress_callback:
                progress_callback(70, "Ejecutando conteo de instancias...")
            
            self.log("Ejecutando conteo de instancias con scripts originales...")
            inst_result = self.run_instances(image_path)
            self.log("Conteo de instancias completado")
            
            # 4. GENERAR REPORTE FINAL
            if progress_callback:
                progress_callback(95, "Generando reporte final...")
            
            stats = self._read_final_statistics(base_name, output_dir)
            
            self.log("PROCESAMIENTO EN MODO BAJA MEMORIA COMPLETADO EXITOSAMENTE")
            return f"Procesamiento en modo baja memoria exitoso. {stats}"
            
        except Exception as e:
            self.log(f"Error en modo baja memoria: {e}")
            raise

    def _run_segmentation_with_tiles(self, image_path: str, output_path: str, 
                                   tile_size: tuple, progress_callback=None) -> bool:
        """
        Ejecuta la segmentación usando el sistema de tiles
        """
        try:
            def segmentation_callback(tile_data, tile_info):
                """Callback para procesar cada tile"""
                try:
                    # Procesar el tile con el modelo
                    processed_tile = self.tile_processor.process_tile_with_model(tile_data)
                    
                    if processed_tile is not None:
                        self.log(f"Tile {tile_info['id']} procesado")
                    else:
                        self.log(f"Tile {tile_info['id']} retornó None")
                        # Crear máscara vacía como fallback
                        height, width = tile_data.shape[:2]
                        processed_tile = np.zeros((height, width), dtype=np.float32)
                    
                    return processed_tile
                    
                except Exception as e:
                    self.log(f"Error en tile {tile_info['id']}: {e}")
                    height, width = tile_data.shape[:2]
                    return np.zeros((height, width), dtype=np.float32)
            
            # CONFIGURAR CALLBACK DE PROGRESO PARA TILES - MODIFICADO
            def tile_progress_callback(progress, message):
                if progress_callback:
                    # Mapear progreso de tiles al rango 20-60% con incrementos de 1%
                    mapped_progress = 20 + (progress * 0.4)
                    progress_callback(int(mapped_progress), f"Segmentación: {message}")
            
            # Ejecutar procesamiento por tiles
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
        """
        Convierte el TIFF de segmentación a PNG para visualización
        """
        try:
            from osgeo import gdal
            import numpy as np
            from PIL import Image
            
            dataset = gdal.Open(tiff_path)
            if dataset:
                seg_data = dataset.ReadAsArray()
                
                # Convertir a formato PNG (8-bit)
                if seg_data.dtype == np.float32:
                    # Normalizar y convertir a uint8
                    seg_normalized = ((seg_data - seg_data.min()) / 
                                    (seg_data.max() - seg_data.min() + 1e-8) * 255).astype(np.uint8)
                else:
                    seg_normalized = seg_data.astype(np.uint8)
                
                # Crear imagen PIL y guardar
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
            progress_callback(10, "Iniciando segmentación...")
        
        self.log("Iniciando segmentacion...")
        seg_result = self.run_segmentation(image_path)
        self.log("Segmentacion completada")
        
        if progress_callback:
            progress_callback(60, "Iniciando conteo de instancias...")
        
        self.log("Iniciando conteo de instancias...")
        inst_result = self.run_instances(image_path)
        self.log("Conteo de instancias completado")
        
        # Leer estadísticas reales
        base_name = os.path.basename(image_path).split('.')[0]
        output_dir = self.config["output"]["directory"]
        stats = self._read_final_statistics(base_name, output_dir)
        
        if progress_callback:
            progress_callback(100, "Procesamiento completado")
        
        return f"Procesamiento exitoso. {stats}"

    def _read_final_statistics(self, base_name, output_dir):
        """Leer estadísticas reales del archivo CSV generado"""
        try:
            csv_path = os.path.join(output_dir, f"{base_name}_predicted_atributos.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                mau_count = len(df[df['ESPECIE'] == 'Mauritia flexuosa'])
                eut_count = len(df[df['ESPECIE'] == 'Euterpe precatoria'])
                oeno_count = len(df[df['ESPECIE'] == 'Oenocarpus bataua'])
                total_count = mau_count + eut_count + oeno_count
                
                return f"Total: {total_count} palmeras (Mauritia: {mau_count}, Euterpe: {eut_count}, Oenocarpus: {oeno_count})"
            else:
                return "Archivo de estadísticas no generado"
        except Exception as e:
            self.log(f"Error leyendo estadísticas: {e}")
            return "Error leyendo estadísticas"

    def run_segmentation(self, image_path):
        """Ejecutar script de segmentación mostrando output en terminal"""
        seg_script = self.config["scripts"]["segmentacion"]
        
        if not os.path.exists(seg_script):
            raise FileNotFoundError(f"Script no encontrado: {seg_script}")
            
        # Obtener parámetros específicos de segmentación
        seg_params = self.config["parameters"]["segmentacion"]
        window_radius = seg_params["window_radius"]
        internal_window_radius = seg_params.get("internal_window_radius", int(round(window_radius * 0.75)))
        scaling = seg_params.get("scaling", "normalize")
        
        # Construir comando
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
        
        # Ejecutar con output en terminal
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
        
        # Obtener parámetros
        inst_params = self.config["parameters"]["instancias"]
        window_radius = inst_params["window_radius"]
        
        # Obtener máscara de segmentación
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
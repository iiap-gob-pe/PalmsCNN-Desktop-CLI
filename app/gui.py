import os
import sys
import re
import time
import json
import ctypes
import psutil
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, 
                             QTextEdit, QWidget, QProgressBar,
                             QMessageBox, QGroupBox, QSplitter, QCheckBox, QComboBox,
                             QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
                             QScrollArea, QRadioButton, QButtonGroup, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QImage, QCursor
from PIL import Image
from osgeo import gdal
import numpy as np
from processor import PalmProcessor 
from optimizacion import MemoryManager, ResourceMonitor, performance_monitor

try:
    from .memory_optimizer import memory_optimizer
except ImportError:
    from memory_optimizer import memory_optimizer

class SmartImageViewer(QLabel):
    """Visor inteligente con Zoom y Panning (Arrastrar)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.current_scale = 1.0
        self.max_preview_size = 2000
        self.downsampled_image = None 
        self.original_path = None        
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #202020; border: none; }")
        self.setText("Imagen aparecerá aquí")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setScaledContents(True)
        
        self.drag_start_pos = None
        self.scroll_area = None

    def set_scroll_area(self, scroll_area):
        self.scroll_area = scroll_area

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = None
            self.setCursor(QCursor(Qt.ArrowCursor))

    def mouseMoveEvent(self, event):
        if self.drag_start_pos and self.scroll_area:
            delta = event.pos() - self.drag_start_pos
            h_bar = self.scroll_area.horizontalScrollBar()
            v_bar = self.scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())

    def display_current_scale(self):
        if self.original_pixmap:
            new_size = self.original_pixmap.size() * self.current_scale
            self.resize(new_size)
            self.setPixmap(self.original_pixmap.scaled(
                new_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            self.setText("")

    def save_preview_image(self, output_dir):
        try:
            if self.downsampled_image is not None and self.original_path is not None:
                base_name = os.path.basename(self.original_path)
                base_name_without_ext = os.path.splitext(base_name)[0]
                preview_filename = f"{base_name_without_ext}_preview.png"
                preview_path = os.path.join(output_dir, preview_filename)
                
                img_save = Image.fromarray(self.downsampled_image)
                img_save.save(preview_path)
                
                print(f"Vista previa guardada: {preview_path}")
                return preview_path
            return None
        except Exception as e:
            print(f"Error guardando vista previa: {e}")
            return None
        
    def load_image_smart(self, image_path):
        try:
            self.original_path = image_path
            
            g_image = gdal.Open(image_path)
            if not g_image:
                print(f"No se pudo abrir la imagen con GDAL: {image_path}")
                return False
                
            width = g_image.RasterXSize
            height = g_image.RasterYSize
            bands = g_image.RasterCount
            print(f"Cargando imagen (Reference Logic): {width}x{height}, {bands} bandas")

            if bands >= 3:
                a_image = g_image.ReadAsArray().astype(np.uint8)
                npimg = np.dstack((a_image[0], a_image[1], a_image[2]))
                height, width, channel = npimg.shape
                
                if height > self.max_preview_size or width > self.max_preview_size:
                    print("Aplicando reducción rápida (Slicing ::4)")
                    npimg = np.array(npimg[::4, ::4, :])
                    height, width, channel = npimg.shape
                
                self.downsampled_image = npimg
                bytesPerLine = 3 * width
                qimage = QImage(npimg.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.original_pixmap = QPixmap.fromImage(qimage)
                self.data_ref = npimg
                
            elif bands == 1:
                a_image = g_image.ReadAsArray().astype(np.uint8)
                npimg = a_image
                height, width = npimg.shape
                
                if height > self.max_preview_size or width > self.max_preview_size:
                    npimg = np.array(npimg[::4, ::4])
                    height, width = npimg.shape
                    
                self.downsampled_image = npimg
                bytesPerLine = width
                qimage = QImage(npimg.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
                self.original_pixmap = QPixmap.fromImage(qimage)
            else:
                return False

            self.current_scale = 0.8
            self.display_current_scale()
            return True

        except Exception as e:
            print(f"Error crítico cargando imagen: {e}")
            return False

    def load_full_image(self, image_path):
        try:
            self.original_path = image_path
            if not image_path.lower().endswith(('.tif', '.tiff')):
                qimage = QImage(image_path)
                if not qimage.isNull():
                    self.original_pixmap = QPixmap.fromImage(qimage)
                    self.downsampled_image = self.qimage_to_array(qimage)
                    self.current_scale = 0.8
                    self.display_current_scale()
                    return True
            
            from PIL import Image
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            pil_image.thumbnail((800, 800))
            self.downsampled_image = np.array(pil_image)
            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
            
            if not qimage.isNull():
                self.original_pixmap = QPixmap.fromImage(qimage)
                self.current_scale = 0.8
                self.display_current_scale()
                return True
            return False
        except Exception as e:
            print(f"Error carga normal: {e}")
            return False
    
    def qimage_to_array(self, qimage):
        try:
            qimage = qimage.convertToFormat(QImage.Format_RGB888)
            width = qimage.width()
            height = qimage.height()
            ptr = qimage.bits()
            ptr.setsize(qimage.byteCount())
            arr = np.array(ptr).reshape(height, width, 3)
            return arr
        except Exception as e:
            print(f"Error convirtiendo QImage a array: {e}")
            return None
    
    def wheelEvent(self, event):
        if self.original_pixmap:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)
    
    def zoom_in(self):
        if self.original_pixmap:
            self.current_scale = min(10.0, self.current_scale * 1.2)
            self.display_current_scale()
            print(f"Zoom in: {self.current_scale:.2f}x")
    
    def zoom_out(self):
        if self.original_pixmap:
            self.current_scale = max(0.1, self.current_scale * 0.8)
            self.display_current_scale()
            print(f"Zoom out: {self.current_scale:.2f}x")
    
    def reset_zoom(self):
        self.current_scale = 1.0
        self.display_current_scale()
        print("Zoom reset: 1.0x")

        
class ProcessingThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, str, dict)
    progress_signal = pyqtSignal(int)
    
    def __init__(self, image_path, processor, use_optimized_tiles=False, 
                 force_tiling=False, processing_mode='auto', image_info=None):
        super().__init__()
        self.image_path = image_path
        self.processor = processor
        self.use_optimized_tiles = use_optimized_tiles
        self.force_tiling = force_tiling
        self.processing_mode = processing_mode
        self.image_info = image_info or {}
        
    def run(self):
        try:
            self.update_signal.emit("Iniciando análisis de memoria...")
            self.progress_signal.emit(1)
            
            # --- DEFINIR LA FUNCIÓN CALLBACK ---
            # Esta función será llamada por processor.py cada vez que lea [PROGRESS]
            def progress_callback(value):
                # Validar que sea entero
                try:
                    val = int(value)
                    self.progress_signal.emit(val)
                except:
                    pass
            # -----------------------------------

            if self.processing_mode == 'low_memory':
                self.processor.config["optimization"]["memory_management"]["low_ram_mode"] = True
                self.update_signal.emit("MODO BAJA MEMORIA - Configurando procesador...")
            
            if self.use_optimized_tiles:
                self.update_signal.emit("Usando procesamiento optimizado con cálculo preciso de memoria...")
                # PASAMOS EL CALLBACK AQUÍ
                result = self.processor.process_image(
                    self.image_path, 
                    force_tiling=self.force_tiling,
                    use_optimized_tiles=True,
                    progress_callback=progress_callback  # <--- ¡CONECTADO!
                )
            else:
                self.update_signal.emit("Ejecutando procesamiento estándar...")
                # PASAMOS EL CALLBACK AQUÍ TAMBIÉN
                result = self.processor.process_image(
                    self.image_path, 
                    force_tiling=self.force_tiling,
                    progress_callback=progress_callback  # <--- ¡CONECTADO!
                )
            
            self.progress_signal.emit(100)
            self.finished_signal.emit(True, result, self.processing_mode, self.image_info)
            
        except Exception as e:
            self.finished_signal.emit(False, str(e), self.processing_mode, self.image_info)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.processor = PalmProcessor()
        self.memory_manager = MemoryManager(self.processor.config)
        self.resource_monitor = ResourceMonitor(self.processor.config)
        
        self.tile_optimizer = memory_optimizer
        
        self.setup_performance_monitoring()
        
        self.current_image = None
        self.current_view = "original"
        
        self.use_optimized_tiles = False
        self.force_tiling = False
        
        # Variable para rastrear el pico máximo de memoria
        self.peak_memory_observed = 0.0
        
        self.init_ui()
        self.main_viewer.data_ref = None
        self.setup_drag_drop()
        self.check_permissions()
        
        self.log("Aplicacion iniciada. Interfaz actualizada.")
        
        self.force_windows_memory_clean()

    def log(self, message):
        """Mostrar mensaje en consola y en el log de la interfaz"""
        # Mostrar en terminal
        print(f"[GUI] {message}")
        
        # Mostrar en interfaz
        self.console.append(f"{message}")
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum()
        )
        

    def force_windows_memory_clean(self):
        try:
            import gc
            gc.collect()
            
            if os.name == 'nt':
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024
                
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.c_void_p(-1))
                
                mem_after = process.memory_info().rss / 1024 / 1024
                self.log(f"Limpieza de memoria: {mem_before:.1f}MB -> {mem_after:.1f}MB")
                
        except Exception as e:
            print(f"Nota: No se pudo compactar la memoria (no crítico): {e}")

    def setup_performance_monitoring(self):
        self.performance_monitor = performance_monitor
        self.performance_monitor.performance_update.connect(self.update_performance_display)
        self.performance_monitor.memory_alert.connect(self.handle_memory_alert)
        
        self.current_progress = 0
        self.processing_start_time = None

    def update_performance_display(self, status):
        try:
            # --- CORRECCIÓN: Calcular memoria TOTAL (Padre + Hijos) ---
            # Esto es necesario porque en modo baja memoria el trabajo pesado
            # ocurre en un subproceso que psutil.Process() no cuenta por defecto.
            try:
                current_process = psutil.Process(os.getpid())
                total_mem_bytes = current_process.memory_info().rss
                
                # Sumar memoria de todos los procesos hijos (scripts de segmentación/instancias)
                children = current_process.children(recursive=True)
                for child in children:
                    try:
                        total_mem_bytes += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                real_mem_mb = total_mem_bytes / (1024 * 1024)
            except Exception:
                # Fallback si falla el cálculo complejo
                real_mem_mb = status.get('memory_used_mb', 0)

            # Actualizar el pico máximo observado
            if real_mem_mb > self.peak_memory_observed:
                self.peak_memory_observed = real_mem_mb

            eta_text = "Calculando..."
            if self.current_progress > 0:
                eta_text = self.performance_monitor.calculate_eta(
                    self.current_progress, 
                    status['elapsed_time_seconds']
                )
            
            performance_info = (
                f"RENDIMIENTO EN TIEMPO REAL:\n"
                f"   Tiempo transcurrido: {status['elapsed_time_formatted']}\n"
                f"   ETA: {eta_text}\n"
                f"   RAM actual: {real_mem_mb:.1f} MB\n"
                f"   RAM pico: {self.peak_memory_observed:.1f} MB\n"
                f"   RAM disponible: {status['memory_available_gb']:.1f} GB\n"
                f"   CPU: {status['cpu_percent']:.1f}%\n"
                f"   Progreso: {self.current_progress}%\n"
            )
            
            # --- NUEVO: Actualizar el panel derecho SIEMPRE ---
            # Esto hará que el texto coincida exactamente con la barra verde
            if hasattr(self, 'lbl_performance_details'):
                self.lbl_performance_details.setText(performance_info)
            # --------------------------------------------------

            # El log se mantiene igual (solo hitos importantes para no saturar)
            if (self.current_progress % 10 == 0 or 
                status['elapsed_time_seconds'] % 30 < 2):
                self.log(performance_info)
                
        except Exception as e:
            self.log(f"Error actualizando rendimiento: {e}")

    def handle_memory_alert(self, alert_message):
        self.log(f"ALERTA: {alert_message}")
        self.status_label.setText(f"ALERTA MEMORIA: {alert_message}")
        self.status_label.setStyleSheet("QLabel { padding: 8px; background-color: #fff0f0; border: 1px solid #ff4444; }")

    def start_performance_tracking(self, processing_mode, image_info):
        self.processing_start_time = time.time()
        self.current_progress = 0
        self.peak_memory_observed = 0.0 # Resetear pico al iniciar
        self.performance_monitor.start_monitoring()
        
        # --- NUEVO: Activar panel lateral inmediatamente ---
        self.stats_panel.setVisible(True)
        self.scroll_details.setVisible(True)
        self.lbl_performance_details.setVisible(True)
        self.lbl_performance_details.setText("Iniciando monitor de recursos...")
        # ---------------------------------------------------
        
        mode_info = self.get_processing_mode_info(processing_mode)
        self.log("=" * 60)
        self.log(f"INICIANDO PROCESAMIENTO - MODO: {mode_info['name']}")
        self.log(f"{mode_info['description']}")
        self.log(f"Estrategia: {mode_info['strategy']}")
        self.log(f"Imagen: {os.path.basename(image_info['path'])}")
        self.log(f"Tamaño: {image_info['dimensions']}")
        self.log(f"Tamaño archivo: {image_info['size_mb']:.1f} MB")
        self.log("=" * 60)

    def get_processing_mode_info(self, mode):
        modes = {
            'auto': {
                'name': 'AUTOMÁTICO',
                'description': 'El sistema decide automáticamente la mejor estrategia',
                'strategy': 'Análisis inteligente de memoria para decidir entre procesamiento directo o por tiles'
            },
            'low_memory': {
                'name': 'BAJA MEMORIA',
                'description': 'Máxima optimización de RAM - Siempre usa tiles',
                'strategy': 'Procesamiento por tiles forzado para mínimo consumo de memoria (~900-1000MB)'
            },
            'optimized': {
                'name': 'OPTIMIZADO',
                'description': 'Equilibrio perfecto entre velocidad y uso de memoria',
                'strategy': 'Usa scripts optimizados de procesamiento por tiles (process_with_tiles.py e instancias_tiles.py)'
            }
        }
        return modes.get(mode, modes['auto'])

    def update_progress_with_eta(self, value):
        # 1. Guardar el valor oficial
        self.current_progress = value
        
        # 2. Actualizar la barra visual (LA VERDAD VISUAL)
        self.progress_bar.setValue(value)
        
        # --- NUEVO: ACTUALIZAR CHECKBOXES SEGÚN PROGRESO ---
        if value >= 60:
            self.chk_seg.setChecked(True)
            self.chk_seg.setStyleSheet("QCheckBox { color: green; font-weight: bold; }")
            
        if value >= 95:
            self.chk_inst.setChecked(True)
            self.chk_inst.setStyleSheet("QCheckBox { color: green; font-weight: bold; }")
        # ---------------------------------------------------
        
        # 3. Actualizar el log SOLO si es un hito importante, 
        # PERO USANDO EL MISMO VALOR 'value' QUE LA BARRA
        milestones = [1, 10, 25, 50, 75, 80, 90, 95, 100] # Añadí 80 a la lista
        
        if value in milestones:
            # Obtener datos de tiempo
            current_status = self.performance_monitor.get_current_status()
            eta = self.performance_monitor.calculate_eta(value, current_status['elapsed_time_seconds'])
            
            # IMPRIMIR EN EL LOG EL MISMO VALOR EXACTO
            self.log(f"PROGRESO: {value}% completado") # <--- AQUÍ ESTÁ LA CLAVE
            self.log(f"   Tiempo transcurrido: {current_status['elapsed_time_formatted']}")
            self.log(f"   ETA: {eta}")
            self.log(f"   RAM usada: {current_status['memory_used_mb']:.1f} MB")

    def show_final_performance_report(self, success, processing_mode, image_info):
        try:
            final_report = self.performance_monitor.generate_final_report(
                processing_mode, image_info, success
            )
            
            self.performance_monitor.stop_monitoring()
            
            # Extraer datos
            summary = final_report['processing_summary']
            resources = final_report['resource_consumption']
            metrics = final_report['performance_metrics']
            
            # Usar el PICO MÁXIMO observado por la GUI (que incluye hijos)
            # en lugar del reportado por el monitor (que a veces solo ve el padre)
            final_peak_memory = max(resources['peak_memory_mb'], self.peak_memory_observed)
            
            # --- Construcción del Texto para la Interfaz (FORMATO SOLICITADO) ---
            details_text = (
                "DETALLES COMPLETOS DEL PROCESAMIENTO\n"
                "======================================\n"
                "INFORMACIÓN DE LA IMAGEN:\n"
                f"   • Archivo: {os.path.basename(image_info['path'])}\n"
                f"   • Dimensiones: {image_info['dimensions']}\n"
                f"   • Tamaño archivo: {image_info['size_mb']:.1f} MB\n\n"
                
                "RENDIMIENTO DEL PROCESAMIENTO:\n"
                f"   • Modo usado: {summary['mode_used']}\n"
                f"   • Estado: {'EXITOSO' if summary['success'] else 'FALLIDO'}\n"
                f"   • Tiempo total: {summary['total_processing_time']}\n"
                f"   • Fecha/hora: {summary['completion_time'].split('T')[0]} {summary['completion_time'].split('T')[1][:8]}\n\n"
                
                "CONSUMO DE RECURSOS:\n"
                f"   • Memoria máxima alcanzada: {final_peak_memory:.1f} MB\n"
                f"   • Memoria final: {resources['final_memory_mb']:.1f} MB\n"
                f"   • CPU promedio: {resources['average_cpu_percent']:.1f}%\n"
                f"   • Eficiencia memoria: {resources['memory_efficiency']}\n\n"
                
                "ESTADÍSTICAS DETALLADAS:\n"
                f"   • Total de pasos: {metrics['total_steps']}\n"
                f"   • Memoria promedio: {metrics['average_memory_usage_mb']:.1f} MB\n"
                f"   • Picos de memoria detectados: {len(metrics['memory_usage_peaks'])}\n"
                "======================================"
            )
            
            # 1. Mostrar en la Consola (Log existente)
            self.log("\n" + details_text)
            
            # 2. Mostrar en el Panel Derecho dentro del ScrollArea
            self.lbl_performance_details.setText(details_text)
            self.scroll_details.setVisible(True) # Hacemos visible el scroll container
            self.lbl_performance_details.setVisible(True)
            
            # Solo aseguramos que el panel sea visible
            self.stats_panel.setVisible(True)
            
            # Actualizamos el reporte JSON con el dato real
            final_report['resource_consumption']['peak_memory_mb'] = final_peak_memory
            self.save_performance_report(final_report, image_info['path'])
            
        except Exception as e:
            self.log(f"Error generando reporte final: {e}")

    def save_performance_report(self, report, image_path):
        try:
            base_name = os.path.basename(image_path).split('.')[0]
            report_file = os.path.join(self.processor.config["output"]["directory"], 
                                       f"{base_name}_performance_report.json")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
                
            self.log(f"Reporte guardado: {report_file}")
            
        except Exception as e:
            self.log(f"Error guardando reporte: {e}")

    def process_image(self):
        if not self.current_image:
            self.log("❌ ERROR: No hay imagen seleccionada")
            QMessageBox.warning(self, "Error", "Por favor, selecciona una imagen primero")
            return
            
        if not os.path.exists(self.current_image):
            self.log("❌ ERROR: La imagen seleccionada ya no existe")
            QMessageBox.warning(self, "Error", "La imagen seleccionada ya no existe")
            return
        
        try:
            # Obtener información de la imagen
            image_info = {
                'path': self.current_image,
                'size_mb': os.path.getsize(self.current_image) / (1024 * 1024),
                'dimensions': self.get_image_dimensions(self.current_image)
            }
            
            # Obtener modo de procesamiento
            processing_mode = self.get_current_processing_mode()
            
            # Modo Optimizado: Usar scripts de tiles avanzados
            if processing_mode == 'optimized':
                self.log("🎯 MODO OPTIMIZADO ACTIVADO - Usando scripts de tiles avanzados")
                self.log("📋 Estrategia: Ejecutar process_with_tiles.py e instancias_tiles.py")
                
                if not os.path.exists("scripts/process_with_tiles.py"):
                    self.log("❌ ERROR: Script process_with_tiles.py no encontrado")
                    QMessageBox.warning(self, "Error", "El script process_with_tiles.py no se encuentra en la carpeta scripts/")
                    return
                
                if not os.path.exists("scripts/instancias_tiles.py"):
                    self.log("❌ ERROR: Script instancias_tiles.py no encontrado")
                    QMessageBox.warning(self, "Error", "El script instancias_tiles.py no se encuentra en la carpeta scripts/")
                    return
                
                # Configurar processor para usar tiles optimizados
                self.processor.config["optimization"]["memory_management"]["low_ram_mode"] = False
                
                # --- NUEVO: REINICIAR INDICADORES ---
                self.chk_seg.setChecked(False)
                self.chk_inst.setChecked(False)
                self.chk_done.setChecked(False)
                
                # Mostrar advertencia
                self.lbl_warning.setVisible(True)
                # ------------------------------------
                
                self.start_performance_tracking(processing_mode, image_info)
                
                output_dir = self.processor.config["output"]["directory"]
                preview_path = self.main_viewer.save_preview_image(output_dir)
                if preview_path:
                    self.log(f"🖼️ Vista previa guardada: {os.path.basename(preview_path)}")
                
                self.main_viewer.clear()
                self.main_viewer.setText("🧩 Procesamiento Optimizado...")
                self.main_viewer.data_ref = None
                
                self.process_btn.setEnabled(False)
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(1, 100)
                self.progress_bar.setValue(1)
                # self.progress_label.setText("1%")  <--- BORRAR ESTA LÍNEA
                
                # Crear thread para procesamiento optimizado
                self.thread = ProcessingThread(
                    self.current_image, 
                    self.processor,
                    use_optimized_tiles=True,  # ¡IMPORTANTE! Esto activa el modo optimizado
                    force_tiling=False,
                    processing_mode=processing_mode,
                    image_info=image_info
                )
                self.thread.update_signal.connect(self.log)
                self.thread.progress_signal.connect(self.update_progress_with_eta)
                self.thread.finished_signal.connect(self.processing_finished)
                self.thread.start()
                return
            
            # Resto del código existente para otros modos...
            if processing_mode == 'low_memory':
                self.processor.config["optimization"]["memory_management"]["low_ram_mode"] = True
                self.log("🔋 MODO BAJA MEMORIA - Configurando procesador...")
            else:
                self.processor.config["optimization"]["memory_management"]["low_ram_mode"] = False
            
            if processing_mode == 'auto':
                self.log("🔍 Realizando análisis de memoria previo...")
                analysis = self.processor.analyze_memory_requirements(self.current_image)
                
                if 'error' not in analysis:
                    strategy = analysis['strategy']
                    self.log(f"📋 Análisis: {strategy['reason']}")
                    if strategy.get('optimal_tile_size'):
                        self.log(f"🧱 Tile óptimo: {strategy['optimal_tile_size']}")
            
            # --- NUEVO: REINICIAR INDICADORES ---
            self.chk_seg.setChecked(False)
            self.chk_inst.setChecked(False)
            self.chk_done.setChecked(False)
            
            # Mostrar advertencia
            self.lbl_warning.setVisible(True)
            # ------------------------------------
            
            self.start_performance_tracking(processing_mode, image_info)
            
            output_dir = self.processor.config["output"]["directory"]
            preview_path = self.main_viewer.save_preview_image(output_dir)
            if preview_path:
                self.log(f"🖼️ Vista previa guardada: {os.path.basename(preview_path)}")
            
            self.main_viewer.clear()
            self.main_viewer.setText("🔄 Procesando...")
            self.main_viewer.data_ref = None
            
            self.process_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(1, 100)
            self.progress_bar.setValue(1)
            # self.progress_label.setText("1%")  <--- BORRAR ESTA LÍNEA
            
            # Determinar si usar tiles optimizados (solo para modo optimizado)
            use_optimized_tiles = (processing_mode == 'optimized')
            force_tiling = (processing_mode == 'low_memory')
            
            self.thread = ProcessingThread(
                self.current_image, 
                self.processor,
                use_optimized_tiles=use_optimized_tiles,
                force_tiling=force_tiling,
                processing_mode=processing_mode,
                image_info=image_info
            )
            self.thread.update_signal.connect(self.log)
            self.thread.progress_signal.connect(self.update_progress_with_eta)
            self.thread.finished_signal.connect(self.processing_finished)
            self.thread.start()
            
        except Exception as e:
            self.log(f"❌ Error iniciando procesamiento: {e}")
            QMessageBox.critical(self, "Error", f"Error iniciando procesamiento:\n{e}")

    def processing_finished(self, success, message, processing_mode, image_info):
        self.force_windows_memory_clean()
        
        self.progress_bar.setVisible(False)
        # self.progress_label.setText("")  <--- BORRAR ESTA LÍNEA
        self.process_btn.setEnabled(True)
        
        self.processor.config["optimization"]["memory_management"]["low_ram_mode"] = False
        
        # --- NUEVO: ACTUALIZAR ESTADO FINAL ---
        self.lbl_warning.setVisible(False) # Ocultar advertencia
        
        if success:
            self.chk_seg.setChecked(True)
            self.chk_inst.setChecked(True)
            self.chk_done.setChecked(True)
            # Ponerlos en verde
            self.chk_seg.setStyleSheet("QCheckBox { color: green; font-weight: bold; }")
            self.chk_inst.setStyleSheet("QCheckBox { color: green; font-weight: bold; }")
            self.chk_done.setStyleSheet("QCheckBox { color: green; font-weight: bold; }")
        else:
            self.lbl_warning.setText("❌ PROCESO FALLIDO")
            self.lbl_warning.setVisible(True)
        # --------------------------------------
        
        self.show_final_performance_report(success, processing_mode, image_info)
        
        if success:
            self.log("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
            self.status_label.setText("Procesamiento completado")
            
            if self.current_image:
                base_name = os.path.basename(self.current_image).split('.')[0]
                output_dir = self.processor.config["output"]["directory"]
                
                # === BLOQUE CORREGIDO (solo esta parte se modifica) ===
                # Buscar archivo de estadísticas con todos los nombres posibles
                posibles_csv = [
                    f"{base_name}_predicted_summary.csv",     # NUEVO: resumen directo (prioridad alta)
                    f"{base_name}_atributos.csv",             # NUEVO: lista detallada (modo tiles)
                    f"{base_name}_predicted_atributos.csv",   # Antiguo (otros modos)
                    "resultados_atributos.csv"
                ]
                
                csv_encontrado = None
                for csv_file in posibles_csv:
                    csv_path = os.path.join(output_dir, csv_file)
                    if os.path.exists(csv_path):
                        csv_encontrado = csv_path
                        self.log(f"📊 CSV encontrado: {csv_file}")
                        break

                if csv_encontrado:
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_encontrado)
                        self.log(f"   Columnas del CSV: {list(df.columns)}")

                        # Caso 1: CSV de resumen (tiene columna 'CONTEO')
                        if 'CONTEO' in [c.upper() for c in df.columns]:
                            self.log("   Tipo: Resumen directo")
                            df.columns = [c.strip().upper() for c in df.columns]
                            
                            mau = int(df.loc[df['ESPECIE'].str.contains('MAURITIA', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('MAURITIA', case=False)].empty else 0
                            eut = int(df.loc[df['ESPECIE'].str.contains('EUTERPE', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('EUTERPE', case=False)].empty else 0
                            oeno = int(df.loc[df['ESPECIE'].str.contains('OENOCARPUS', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('OENOCARPUS', case=False)].empty else 0
                            
                            total_row = df[df['ESPECIE'].str.contains('TOTAL', case=False, na=False)]
                            total = int(total_row['CONTEO'].iloc[0]) if not total_row.empty else (mau + eut + oeno)
                        
                        # Caso 2: CSV de listado (una filra por palmera)
                        else:
                            self.log("   Tipo: Listado detallado")
                            if 'ESPECIE' in df.columns:
                                mau = len(df[df['ESPECIE'].str.contains('Mauritia flexuosa', case=False, na=False)])
                                eut = len(df[df['ESPECIE'].str.contains('Euterpe precatoria', case=False, na=False)])
                                oeno = len(df[df['ESPECIE'].str.contains('Oenocarpus bataua', case=False, na=False)])
                                total = len(df)
                            else:
                                self.log("   No se encontró columna ESPECIE")
                                mau = eut = oeno = total = len(df)

                        # Actualizar el panel de estadísticas
                        stats_text = (
                            f"Mauritia flexuosa: {mau}\n"
                            f"Euterpe precatoria: {eut}\n"
                            f"Oenocarpus bataua: {oeno}"
                        )
                        self.update_statistics_panel(stats_text)
                        
                        self.log(f"✅ Estadísticas mostradas: Total {total} palmeras")

                    except Exception as e:
                        self.log(f"Error leyendo CSV: {e}")
                else:
                    self.log("⚠️ No se encontró ningún archivo de estadísticas")
                # === FIN DEL BLOQUE CORREGIDO ===
            
            self.view_selector.setCurrentIndex(2)
            self.change_view("Conteo de Instancias")
            
            QTimer.singleShot(1000, self.force_interface_refresh)
                    
            QMessageBox.information(self, "Éxito", 
                                f"Procesamiento completado.\n"
                                f"Ver resultados en el panel derecho y consola.")
        else:
            self.log(f"ERROR en procesamiento: {message}")
            self.status_label.setText("ERROR en procesamiento")
            QMessageBox.critical(self, "Error", f"Error en procesamiento:\n{message}")

    def get_current_processing_mode(self):
        """Obtener el modo de procesamiento seleccionado"""
        if self.mode_optimized.isChecked():
            return 'optimized'
        elif self.mode_low_memory.isChecked():
            return 'low_memory'
        elif self.mode_auto.isChecked():
            return 'auto'
        else:
            # Por defecto, auto
            return 'auto'

    def get_image_dimensions(self, image_path):
        try:
            dataset = gdal.Open(image_path)
            if dataset:
                width = dataset.RasterXSize
                height = dataset.RasterYSize
                dataset = None
                return f"{width}x{height}"
        except:
            pass
        return "Desconocido"

    def force_interface_refresh(self):
        try:
            if self.current_image:
                base_name = os.path.basename(self.current_image).split('.')[0]
                output_dir = self.processor.config["output"]["directory"]
                
                files_to_check = {
                    "Segmentación": f"{base_name}_balanced_argmax.png",
                    "Conteo": f"{base_name}_predicted.png",
                    "Resumen CSV": f"{base_name}_predicted_summary.csv",  # NUEVO
                    "Atributos CSV": f"{base_name}_atributos.csv",        # NUEVO
                    "Atributos CSV (antiguo)": f"{base_name}_predicted_atributos.csv"
                }
                
                for file_type, file_name in files_to_check.items():
                    file_path = os.path.join(output_dir, file_name)
                    if os.path.exists(file_path):
                        self.log(f"{file_type}: {file_name} - DISPONIBLE")
                    else:
                        self.log(f"{file_type}: {file_name} - NO ENCONTRADO")
                
                # === BLOQUE CORREGIDO - Reemplaza self.update_statistics_from_files() ===
                # Buscar archivo de estadísticas con todos los nombres posibles
                posibles_csv = [
                    f"{base_name}_predicted_summary.csv",     # NUEVO: resumen directo (prioridad alta)
                    f"{base_name}_atributos.csv",             # NUEVO: lista detallada (modo tiles)
                    f"{base_name}_predicted_atributos.csv",   # Antiguo (otros modos)
                    "resultados_atributos.csv"
                ]
                
                csv_encontrado = None
                for csv_file in posibles_csv:
                    csv_path = os.path.join(output_dir, csv_file)
                    if os.path.exists(csv_path):
                        csv_encontrado = csv_path
                        self.log(f"📊 CSV encontrado: {csv_file}")
                        break

                if csv_encontrado:
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_encontrado)
                        self.log(f"   Columnas del CSV: {list(df.columns)}")

                        # Caso 1: CSV de resumen (tiene columna 'CONTEO')
                        if 'CONTEO' in [c.upper() for c in df.columns]:
                            self.log("   Tipo: Resumen directo")
                            df.columns = [c.strip().upper() for c in df.columns]
                            
                            mau = int(df.loc[df['ESPECIE'].str.contains('MAURITIA', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('MAURITIA', case=False)].empty else 0
                            eut = int(df.loc[df['ESPECIE'].str.contains('EUTERPE', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('EUTERPE', case=False)].empty else 0
                            oeno = int(df.loc[df['ESPECIE'].str.contains('OENOCARPUS', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('OENOCARPUS', case=False)].empty else 0
                            
                            total_row = df[df['ESPECIE'].str.contains('TOTAL', case=False, na=False)]
                            total = int(total_row['CONTEO'].iloc[0]) if not total_row.empty else (mau + eut + oeno)
                        
                        # Caso 2: CSV de listado (una fila por palmera)
                        else:
                            self.log("   Tipo: Listado detallado")
                            if 'ESPECIE' in df.columns:
                                mau = len(df[df['ESPECIE'].str.contains('Mauritia flexuosa', case=False, na=False)])
                                eut = len(df[df['ESPECIE'].str.contains('Euterpe precatoria', case=False, na=False)])
                                oeno = len(df[df['ESPECIE'].str.contains('Oenocarpus bataua', case=False, na=False)])
                                total = len(df)
                            else:
                                self.log("   No se encontró columna ESPECIE")
                                mau = eut = oeno = total = len(df)

                        # Actualizar el panel de estadísticas
                        stats_text = (
                            f"Mauritia flexuosa: {mau}\n"
                            f"Euterpe precatoria: {eut}\n"
                            f"Oenocarpus bataua: {oeno}"
                        )
                        self.update_statistics_panel(stats_text)
                        
                        self.log(f"✅ Estadísticas actualizadas: Total {total} palmeras")

                    except Exception as e:
                        self.log(f"Error leyendo CSV: {e}")
                else:
                    self.log("⚠️ No se encontró ningún archivo de estadísticas")
                # === FIN DEL BLOQUE CORREGIDO ===
                    
        except Exception as e:
            self.log(f"Error en actualización de interfaz: {e}")

    def update_statistics_from_files(self):
        try:
            if not self.current_image:
                return
                
            base_name = os.path.basename(self.current_image).split('.')[0]
            output_dir = self.processor.config["output"]["directory"]
            
            # Buscar archivo de atributos en diferentes ubicaciones posibles
            posibles_csv = [
                f"{base_name}_predicted_atributos.csv",
                f"atributos_{base_name}.csv",
                "resultados_atributos.csv",
                os.path.join(output_dir, f"{base_name}_predicted_atributos.csv")
            ]
            
            csv_encontrado = None
            for csv_path in posibles_csv:
                if os.path.exists(csv_path):
                    csv_encontrado = csv_path
                    break
            
            if csv_encontrado:
                import pandas as pd
                df = pd.read_csv(csv_encontrado)
                
                mau_count = len(df[df['ESPECIE'] == 'Mauritia flexuosa'])
                eut_count = len(df[df['ESPECIE'] == 'Euterpe precatoria'])
                oeno_count = len(df[df['ESPECIE'] == 'Oenocarpus bataua'])
                total_count = mau_count + eut_count + oeno_count
                
                stats_text = f"Mauritia flexuosa: {mau_count}\nEuterpe precatoria: {eut_count}\nOenocarpus bataua: {oeno_count}"
                self.update_statistics_panel(stats_text)
                
                self.log(f"Estadísticas cargadas: Total {total_count} palmeras")
                
        except Exception as e:
            self.log(f"Error actualizando estadísticas: {e}")

    def init_ui(self):
        self.setWindowTitle("Segmentación de Palmeras - IA")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        file_group = QGroupBox("Selección de Imagen")
        file_layout = QVBoxLayout(file_group)
        
        file_btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("Seleccionar Imagen")
        self.select_btn.clicked.connect(self.select_image)
        self.select_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        
        self.image_label = QLabel("No se ha seleccionado ninguna imagen")
        self.image_label.setStyleSheet("QLabel { padding: 8px; background-color: #f0f0f0; border: 1px solid #ccc; }")
        
        file_btn_layout.addWidget(self.select_btn)
        file_btn_layout.addWidget(self.image_label, 1)
        file_layout.addLayout(file_btn_layout)
        
        self.load_progress_bar = QProgressBar()
        self.load_progress_bar.setVisible(False)
        self.load_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 20px;
            }
        """)
        file_layout.addWidget(self.load_progress_bar)
        
        processing_mode_group = QGroupBox("Modo de Procesamiento")
        processing_mode_layout = QVBoxLayout(processing_mode_group)
        
        self.mode_auto = QRadioButton("Automático (Recomendado)")
        self.mode_auto.setToolTip("El sistema decide automáticamente la mejor estrategia")
        self.mode_auto.setChecked(True)
        
        self.mode_low_memory = QRadioButton("Modo Baja Memoria")
        self.mode_low_memory.setToolTip("Forzar uso de tiles para máximo ahorro de RAM")
        
        self.mode_optimized = QRadioButton("Modo Optimizado")
        self.mode_optimized.setToolTip("Usa scripts optimizados de procesamiento por tiles para mejor rendimiento")
        self.mode_optimized.setStyleSheet("color: #2196F3; font-weight: bold;")
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.mode_auto)
        self.mode_group.addButton(self.mode_low_memory)
        self.mode_group.addButton(self.mode_optimized)
        
        self.mode_auto.toggled.connect(self.on_processing_mode_changed)
        self.mode_low_memory.toggled.connect(self.on_processing_mode_changed)
        self.mode_optimized.toggled.connect(self.on_processing_mode_changed)
        
        processing_mode_layout.addWidget(self.mode_auto)
        processing_mode_layout.addWidget(self.mode_low_memory)
        processing_mode_layout.addWidget(self.mode_optimized)
        
        self.mode_info_label = QLabel("Modo automático: El sistema decide la mejor estrategia")
        self.mode_info_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic; padding: 5px;")
        self.mode_info_label.setWordWrap(True)
        processing_mode_layout.addWidget(self.mode_info_label)
        
        optimization_group = QGroupBox("Optimización de Memoria")
        optimization_layout = QVBoxLayout(optimization_group)
        
        self.low_memory_cb = QCheckBox("Forzar Procesamiento por Tiles")
        self.low_memory_cb.setToolTip("Activar para procesar todas las imágenes por tiles")
        self.low_memory_cb.stateChanged.connect(self.toggle_force_tiling)
        
        self.memory_info_label = QLabel("Memoria: calculando...")
        self.update_memory_info()
        
        optimization_layout.addWidget(self.low_memory_cb)
        optimization_layout.addWidget(self.memory_info_label)
        
        drag_label = QLabel("Tambien puedes arrastrar y soltar una imagen directamente sobre la ventana")
        drag_label.setStyleSheet("QLabel { color: #666; font-style: italic; padding: 5px; }")
        file_layout.addWidget(drag_label)
        
        left_layout.addWidget(file_group)
        left_layout.addWidget(processing_mode_group)
        left_layout.addWidget(optimization_group)

        diag_group = QGroupBox("Diagnóstico del Sistema")
        diag_layout = QVBoxLayout(diag_group)
        
        self.btn_check_image = QPushButton("Calcular Procesamiento de Imagen")
        self.btn_check_image.setToolTip("Analiza si hay suficiente RAM para la imagen seleccionada")
        self.btn_check_image.clicked.connect(self.check_image_feasibility)
        self.btn_check_image.setStyleSheet("text-align: left; padding: 6px;")
        
        self.btn_check_system = QPushButton("Analizar Requisitos del Equipo")
        self.btn_check_system.setToolTip("Compara tu PC con los requisitos mínimos")
        self.btn_check_system.clicked.connect(self.show_system_analysis)
        self.btn_check_system.setStyleSheet("text-align: left; padding: 6px;")
        
        # self.btn_analyze_tiles = QPushButton("Analizar Procesamiento por Tiles")
        # self.btn_analyze_tiles.setToolTip("Plan óptimo de procesamiento para imágenes grandes")
        # self.btn_analyze_tiles.clicked.connect(self.analyze_tile_processing)
        # self.btn_analyze_tiles.setStyleSheet("text-align: left; padding: 6px;")
        
        # self.btn_memory_analysis = QPushButton("Análisis Detallado de Memoria")
        # self.btn_memory_analysis.setToolTip("Análisis completo de requisitos de memoria")
        # self.btn_memory_analysis.clicked.connect(self.analyze_memory_requirements)
        # self.btn_memory_analysis.setStyleSheet("text-align: left; padding: 6px;")
        
        diag_layout.addWidget(self.btn_check_image)
        diag_layout.addWidget(self.btn_check_system)
        # diag_layout.addWidget(self.btn_analyze_tiles)
        # diag_layout.addWidget(self.btn_memory_analysis)
        
        left_layout.addWidget(diag_group)
        
        process_group = QGroupBox("Procesamiento")
        process_layout = QVBoxLayout(process_group)
        
        self.process_btn = QPushButton("Procesar Imagen")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton { 
                padding: 12px; 
                font-weight: bold; 
                font-size: 14px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
        """)
        process_layout.addWidget(self.process_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(1, 100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)
        process_layout.addWidget(self.progress_bar)
        
        # --- NUEVO: INDICADORES DE PROCESO ---
        status_layout = QHBoxLayout()
        
        self.chk_seg = QCheckBox("Segmentación")
        self.chk_inst = QCheckBox("Instancias")
        self.chk_done = QCheckBox("Finalizado")
        
        # Estilo para que se vean bien pero el usuario no los toque
        for chk in [self.chk_seg, self.chk_inst, self.chk_done]:
            chk.setEnabled(False) # Deshabilitado para que el usuario no haga clic (se ven gris, pero se marcarán)
            # Opcional: Truco para que se vean habilitados pero no clicables:
            # chk.setAttribute(Qt.WA_TransparentForMouseEvents) 
            # chk.setFocusPolicy(Qt.NoFocus)
            
            # Estilo personalizado para el check
            chk.setStyleSheet("""
                QCheckBox::indicator:checked {
                    background-color: #4CAF50;
                    border: 1px solid #4CAF50;
                    border-radius: 3px;
                }
                QCheckBox { font-weight: bold; font-size: 11px; }
            """)
            status_layout.addWidget(chk)
            
        process_layout.addLayout(status_layout)
        
        # --- NUEVO: ETIQUETA DE ADVERTENCIA CORREGIDA ---
        self.lbl_warning = QLabel("⚠️ POR FAVOR, NO CIERRE LA APLICACIÓN\nEl proceso se está ejecutando...")
        self.lbl_warning.setAlignment(Qt.AlignCenter)
        
        # 1. Permitir que el texto se ajuste automáticamente si falta espacio horizontal
        self.lbl_warning.setWordWrap(True) 
        
        # 2. Establecer una altura mínima para asegurar que quepan las dos líneas cómodamente
        # 50 pixeles debería ser suficiente para el texto, el padding y el borde.
        self.lbl_warning.setMinimumHeight(50) 
        
        self.lbl_warning.setStyleSheet("""
            QLabel { 
                color: #c62828; /* Un rojo un poco más oscuro para mejor contraste */
                font-weight: bold; 
                font-size: 12px; /* Aumenté un poco la fuente para que se lea mejor */
                background-color: #ffebee;
                border: 2px solid #ef9a9a; /* Borde un poco más grueso */
                border-radius: 6px;
                padding: 8px; /* Un poco más de espacio interno */
                margin-top: 10px; /* Más separación de la barra de progreso y los checks */
            }
        """)
        self.lbl_warning.setVisible(False) # Oculto al inicio
        process_layout.addWidget(self.lbl_warning)
        # ------------------------------------------------
        
        # self.progress_label = QLabel("")  <--- (RECORDATORIO: ESTO LO BORRASTE ANTES)
        # self.progress_label.setAlignment(Qt.AlignCenter)
        # self.progress_label.setStyleSheet("QLabel { color: #333; font-weight: bold; }")
        # process_layout.addWidget(self.progress_label)
        
        left_layout.addWidget(process_group)
        
        output_group = QGroupBox("Gestión de Salida y Logs")
        output_layout = QVBoxLayout(output_group)
        
        path_layout = QHBoxLayout()
        
        current_out = self.processor.config["output"]["directory"]
        if not os.path.isabs(current_out):
            current_out = os.path.abspath(current_out)
            
        self.lbl_output_path = QLabel(f"Salida: {current_out}")
        self.lbl_output_path.setStyleSheet("color: #555; font-size: 11px; font-style: italic;")
        self.lbl_output_path.setWordWrap(True)
        
        btn_change_dir = QPushButton("Cambiar...")
        btn_change_dir.setToolTip("Cambiar carpeta donde se guardan los resultados")
        btn_change_dir.clicked.connect(self.select_output_folder)
        btn_change_dir.setFixedWidth(70)
        
        btn_open_dir = QPushButton("Abrir")
        btn_open_dir.setToolTip("Abrir carpeta de resultados")
        btn_open_dir.setStyleSheet("font-weight: bold; color: #2196F3;")
        btn_open_dir.clicked.connect(self.open_output_folder)
        btn_open_dir.setFixedWidth(50)
        
        path_layout.addWidget(self.lbl_output_path)
        path_layout.addWidget(btn_change_dir)
        path_layout.addWidget(btn_open_dir)
        
        output_layout.addLayout(path_layout)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                border: 1px solid #444;
                border-radius: 5px;
            }
        """)
        output_layout.addWidget(self.console)
        
        left_layout.addWidget(output_group)
        
        self.status_label = QLabel("Listo para comenzar")
        self.status_label.setStyleSheet("QLabel { padding: 8px; background-color: #e8f5e8; border: 1px solid #4CAF50; }")
        left_layout.addWidget(self.status_label)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.main_display_group = QGroupBox("Visualización y Resultados")
        display_container_layout = QHBoxLayout(self.main_display_group)
        
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        
        view_label = QLabel("Vista:")
        self.view_selector = QComboBox()
        self.view_selector.addItems(["Imagen Original", "Segmentación", "Conteo de Instancias"])
        self.view_selector.currentTextChanged.connect(self.change_view)
        self.view_selector.setEnabled(False)
        
        zoom_in_btn = QPushButton("+")
        zoom_out_btn = QPushButton("-") 
        reset_zoom_btn = QPushButton("Reset")
        
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        
        for btn in [zoom_in_btn, zoom_out_btn, reset_zoom_btn]:
            btn.setFixedWidth(40)
        
        controls_layout.addWidget(view_label)
        controls_layout.addWidget(self.view_selector)
        controls_layout.addStretch()
        controls_layout.addWidget(zoom_in_btn)
        controls_layout.addWidget(zoom_out_btn) 
        controls_layout.addWidget(reset_zoom_btn)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #202020; border: 2px dashed #444;")
        
        self.main_viewer = SmartImageViewer()
        self.main_viewer.set_scroll_area(self.scroll_area)
        self.scroll_area.setWidget(self.main_viewer)
        
        viewer_layout.addWidget(controls_widget)
        viewer_layout.addWidget(self.scroll_area)
        
        self.stats_panel = QWidget()
        self.stats_panel.setFixedWidth(250)
        self.stats_panel.setVisible(False)
        stats_layout = QVBoxLayout(self.stats_panel)
        stats_layout.setAlignment(Qt.AlignTop)
        
        lbl_title = QLabel("RESULTADOS")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        stats_layout.addWidget(lbl_title)

        self.lbl_mauritia = QLabel()
        self.lbl_euterpe = QLabel()
        self.lbl_oenocarpus = QLabel()
        self.lbl_total = QLabel()
        
        self.lbl_files_info = QLabel()
        self.lbl_files_info.setWordWrap(True)
        self.lbl_files_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        stats_layout.addWidget(self.lbl_mauritia)
        stats_layout.addWidget(self.lbl_euterpe)
        stats_layout.addWidget(self.lbl_oenocarpus)
        
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #ccc;")
        stats_layout.addWidget(line)
        
        stats_layout.addWidget(self.lbl_total)
        
        line2 = QWidget()
        line2.setFixedHeight(1)
        line2.setStyleSheet("background-color: #ccc; margin-top: 10px;")
        stats_layout.addWidget(line2)
        stats_layout.addWidget(self.lbl_files_info)
        
        # --- NUEVO COMPONENTE: ETIQUETA PARA DETALLES DE RENDIMIENTO CON SCROLL ---
        self.lbl_performance_details = QLabel()
        self.lbl_performance_details.setWordWrap(True)
        self.lbl_performance_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_performance_details.setStyleSheet("""
            QLabel { 
                font-family: 'Consolas', 'Courier New', monospace; 
                font-size: 13px; 
                color: #333; 
                background-color: #fffbe6; 
                padding: 8px; 
                border: 1px solid #e0e0e0; 
                border-radius: 4px; 
                margin-top: 10px; 
            }
        """)
        
        # Envolver en un QScrollArea para permitir desplazamiento
        self.scroll_details = QScrollArea()
        self.scroll_details.setWidget(self.lbl_performance_details)
        self.scroll_details.setWidgetResizable(True)
        self.scroll_details.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        self.scroll_details.setVisible(False) # Oculto al inicio
        
        stats_layout.addWidget(self.scroll_details)
        # -------------------------------------------------------------
        
        stats_layout.addStretch()

        display_container_layout.addWidget(viewer_container, stretch=1)
        display_container_layout.addWidget(self.stats_panel)

        right_layout.addWidget(self.main_display_group)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 1050]) 
        
        main_layout.addWidget(splitter)
        
        self.log("Aplicacion iniciada. Interfaz actualizada.")

    def on_processing_mode_changed(self):
        if self.mode_auto.isChecked():
            self.use_optimized_tiles = False
            self.force_tiling = False
            self.mode_info_label.setText("Modo automático: El sistema decide la mejor estrategia basada en análisis de memoria")
            self.low_memory_cb.setEnabled(True)
            self.log("Modo automático activado")
            
        elif self.mode_low_memory.isChecked():
            self.use_optimized_tiles = False
            self.force_tiling = True
            self.mode_info_label.setText("Modo baja memoria: Siempre usar tiles para máximo ahorro de RAM (~900-1000MB)")
            self.low_memory_cb.setChecked(True)
            self.low_memory_cb.setEnabled(False)
            self.log("Modo baja memoria activado")
            
        elif self.mode_optimized.isChecked():
            self.use_optimized_tiles = True
            self.force_tiling = False
            self.mode_info_label.setText("Modo optimizado: Usa scripts optimizados de procesamiento por tiles (process_with_tiles.py e instancias_tiles.py)")
            self.low_memory_cb.setEnabled(True)
            self.log("🎯 Modo optimizado activado - Se usarán scripts de tiles avanzados")

    def toggle_force_tiling(self, state):
        self.force_tiling = (state == Qt.Checked)
        
        if self.force_tiling:
            self.log("Forzando procesamiento por tiles para todas las imágenes")
        else:
            self.log("Procesamiento automático por tamaño de imagen")

    def analyze_memory_requirements(self):
        if not self.current_image:
            QMessageBox.warning(self, "Atención", "Primero selecciona una imagen")
            return
        
        try:
            analysis = self.processor.analyze_memory_requirements(self.current_image)
            
            if 'error' in analysis:
                QMessageBox.warning(self, "Error en análisis", analysis['error'])
                return
            
            self.show_memory_analysis_dialog(analysis)
            
        except Exception as e:
            self.log(f"Error en análisis de memoria: {e}")
            QMessageBox.warning(self, "Error", f"Error en análisis: {e}")

    def show_memory_analysis_dialog(self, analysis):
        strategy = analysis['strategy']
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Análisis Detallado de Memoria")
        dialog.setMinimumWidth(700)
        layout = QVBoxLayout(dialog)
        
        title = QLabel("ANÁLISIS DETALLADO DE MEMORIA")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(title)
        
        img_info = QLabel(
            f"Imagen: {os.path.basename(self.current_image)}<br>"
            f"Dimensiones: {strategy['image_info']['width']}x{strategy['image_info']['height']}<br>"
            f"Memoria requerida: {strategy['image_info']['memory_mb']:.1f} MB<br>"
            f"Bandas: {strategy['image_info']['bands']}"
        )
        img_info.setStyleSheet("background-color: #f0f8ff; padding: 10px; border-radius: 5px;")
        layout.addWidget(img_info)
        
        sys_info = QLabel(
            f"Memoria disponible: {strategy['system_info']['available_ram_mb']:.1f} MB<br>"
            f"Umbral seguro: {strategy['system_info']['threshold_mb']:.1f} MB<br>"
            f"Margen de seguridad: 70%"
        )
        sys_info.setStyleSheet("background-color: #fff0f5; padding: 10px; border-radius: 5px; margin-top: 10px;")
        layout.addWidget(sys_info)
        
        rec_info = QLabel(
            f"Decisión: {'Usar TILES' if strategy['use_tiles'] else 'Procesamiento DIRECTO'}<br>"
            f"Tile óptimo: {strategy['tile_size']}<br>"
            f"Razón: {strategy['reason']}"
        )
        color = "#f0fff0" if not strategy['use_tiles'] else "#fff0f0"
        rec_info.setStyleSheet(f"background-color: {color}; padding: 10px; border-radius: 5px; margin-top: 10px;")
        layout.addWidget(rec_info)
        
        btn_layout = QHBoxLayout()
        
        if strategy['use_tiles']:
            optimize_btn = QPushButton("Usar Modo Optimizado")
            optimize_btn.clicked.connect(lambda: self.use_optimized_processing(dialog))
            optimize_btn.setStyleSheet("padding: 8px; font-weight: bold; background-color: #2196F3; color: white;")
            btn_layout.addWidget(optimize_btn)
        
        close_btn = QPushButton("Entendido")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("padding: 8px; font-weight: bold;")
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        dialog.exec_()

    def use_optimized_processing(self, dialog):
        dialog.accept()
        self.mode_optimized.setChecked(True)
        self.log("Cambiando a modo optimizado basado en análisis")
        self.on_processing_mode_changed()

    def analyze_tile_processing(self):
        if not self.current_image:
            QMessageBox.warning(self, "Atención", "Primero debes seleccionar una imagen")
            return
        
        try:
            plan = self.tile_optimizer.get_processing_plan(self.current_image)
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Análisis de Procesamiento por Tiles")
            dialog.setMinimumWidth(600)
            layout = QVBoxLayout(dialog)
            
            title = QLabel("PLAN DE PROCESAMIENTO POR TILES")
            title.setStyleSheet("font-size: 16px; margin-bottom: 15px;")
            layout.addWidget(title)
            
            img_info = QLabel(
                f"Imagen: {os.path.basename(self.current_image)}<br>"
                f"Tamaño: {plan['image_info']['image_size_mb']:.1f} MB<br>"
                f"Estrategia: {plan['image_info']['strategy']}<br>"
                f"Tile recomendado: {plan['image_info']['recommended_tile_size']}px"
            )
            img_info.setStyleSheet("background-color: #f0f8ff; padding: 10px; border-radius: 5px;")
            layout.addWidget(img_info)
            
            steps_label = QLabel("Pasos de procesamiento:")
            layout.addWidget(steps_label)
            
            for i, step in enumerate(plan['steps'], 1):
                step_label = QLabel(f"{i}. {step}")
                step_label.setStyleSheet("margin-left: 10px;")
                layout.addWidget(step_label)
            
            if plan['warnings']:
                warnings_label = QLabel("Advertencias:")
                layout.addWidget(warnings_label)
                
                for warning in plan['warnings']:
                    warn_label = QLabel(f"• {warning}")
                    warn_label.setStyleSheet("color: orange; margin-left: 10px;")
                    layout.addWidget(warn_label)
            
            close_btn = QPushButton("Entendido - Usar este plan")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setStyleSheet("padding: 8px; font-weight: bold;")
            layout.addWidget(close_btn)
            
            dialog.exec_()
            self.log("Análisis de tiles completado")
            
        except Exception as e:
            self.log(f"Error en análisis de tiles: {e}")

    def toggle_low_memory_mode(self, state):
        self.force_tiling = (state == Qt.Checked)
        self.toggle_force_tiling(state)

    def select_output_folder(self):
        current_dir = self.processor.config["output"]["directory"]
        new_dir = QFileDialog.getExistingDirectory(
            self, 
            "Seleccionar Carpeta de Salida",
            current_dir
        )
        
        if new_dir:
            self.processor.config["output"]["directory"] = new_dir
            self.lbl_output_path.setText(f"Salida: {new_dir}")
            self.log(f"Carpeta de salida cambiada a: {new_dir}")

    def open_output_folder(self):
        path = self.processor.config["output"]["directory"]
        if not os.path.isabs(path):
            path = os.path.abspath(path)
            
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                
            if os.name == 'nt':
                os.startfile(path)
            elif os.name == 'posix':
                os.system(f'xdg-open "{path}"')
                
            self.log(f"Abriendo carpeta: {path}")
        except Exception as e:
            self.log(f"Error al abrir carpeta: {e}")
            QMessageBox.warning(self, "Error", f"No se pudo abrir la carpeta:\n{e}")
        
    def check_image_feasibility(self):
        if not self.current_image:
            QMessageBox.warning(self, "Atención", "Primero debes seleccionar una imagen.")
            return

        self.log("Analizando viabilidad de procesamiento...")
        can_run, message = self.memory_manager.can_process_image(self.current_image)
        
        if can_run:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Análisis Favorable")
            msg_box.setText("El procesamiento es viable")
            msg_box.setInformativeText(f"El sistema tiene recursos suficientes.\n\n{message}")
            msg_box.exec_()
            self.log("Diagnóstico imagen: VIABLE")
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Advertencia de Recursos")
            msg_box.setText("Riesgo de falta de memoria")
            msg_box.setInformativeText(f"Es posible que el proceso falle o sea lento.\n\n{message}\n\n¿Deseas intentar activar el 'Modo Baja Memoria'?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            ret = msg_box.exec_()
            
            if ret == QMessageBox.Yes:
                self.mode_low_memory.setChecked(True)
                self.log("Diagnóstico imagen: RIESGO (Usuario activó modo baja memoria)")
            else:
                self.log("Diagnóstico imagen: RIESGO")

    def show_system_analysis(self):
        meets, data = self.resource_monitor.check_min_requirements()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Análisis de Requisitos del Sistema")
        dialog.setMinimumWidth(500)
        layout = QVBoxLayout(dialog)
        
        if meets:
            header = QLabel("Tu equipo CUMPLE con los requisitos recomendados")
            header.setStyleSheet("color: green; font-size: 14px; margin-bottom: 10px;")
        else:
            header = QLabel("Tu equipo NO CUMPLE algunos requisitos mínimos")
            header.setStyleSheet("color: orange; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(header)
        
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Componente", "Mínimo", "Tu Equipo", "Estado"])
        table.verticalHeader().setVisible(False)
        table.setRowCount(3)
        
        row = 0
        for key, info in data.items():
            table.setItem(row, 0, QTableWidgetItem(info['item']))
            table.setItem(row, 1, QTableWidgetItem(info['min']))
            table.setItem(row, 2, QTableWidgetItem(info['actual']))
            
            status_item = QTableWidgetItem("OK" if info['status'] else "BAJO")
            status_item.setBackground(Qt.green if info['status'] else Qt.red)
            status_item.setForeground(Qt.white)
            status_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 3, status_item)
            row += 1
            
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(table)
        
        close_btn = QPushButton("Entendido")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
        self.log("Diagnóstico de sistema realizado.")
        
    def update_statistics_panel(self, result_text):
        try:
            # Si el texto contiene información de procesamiento (no solo conteo de palmeras)
            if "ARCHIVO:" in result_text or "DIMENSIONES:" in result_text:
                # Es un reporte completo, mostrarlo directamente
                self.lbl_files_info.setText(result_text)
                self.lbl_files_info.setStyleSheet("color: #333; font-size: 11px; margin-top: 10px; background-color: #f8f9fa; padding: 10px; border-radius: 5px;")
                self.stats_panel.setVisible(True)
                return
            
            # Código original para conteo de palmeras...
            mau, eut, oeno = 0, 0, 0
            m_mau = re.search(r"Mauritia flexuosa: (\d+)", result_text)
            if m_mau: mau = int(m_mau.group(1))
            m_eut = re.search(r"Euterpe precatoria: (\d+)", result_text)
            if m_eut: eut = int(m_eut.group(1))
            m_oeno = re.search(r"Oenocarpus bataua: (\d+)", result_text)
            if m_oeno: oeno = int(m_oeno.group(1))
            
            total = mau + eut + oeno

            style_template = "QLabel { background-color: %s; color: %s; padding: 10px; border-radius: 5px; font-weight: bold; margin-bottom: 5px; }"
            
            self.lbl_mauritia.setText(f"Mauritia flexuosa\n{mau} und.")
            self.lbl_mauritia.setStyleSheet(style_template % ("#ff7f00", "white"))
            
            self.lbl_euterpe.setText(f"Euterpe precatoria\n{eut} und.")
            self.lbl_euterpe.setStyleSheet(style_template % ("#08F6EB", "black"))
            
            self.lbl_oenocarpus.setText(f"Oenocarpus bataua\n{oeno} und.")
            self.lbl_oenocarpus.setStyleSheet(style_template % ("#E008F6", "white"))
            
            self.lbl_total.setText(f"TOTAL DETECTADO\n{total} palmeras")
            self.lbl_total.setStyleSheet("padding: 10px; font-size: 12px; font-weight: bold; background-color: #e8f5e8; border-radius: 5px;")

            output_dir = os.path.abspath(self.processor.config["output"]["directory"])
            
            msg_panel = f"ARCHIVOS GUARDADOS EN:\n{output_dir}\n\n"
            msg_panel += "• _predicted.png (Imagen)\n"
            msg_panel += "• _poly.gpkg (Vectores)\n"
            msg_panel += "• _atributos.csv (Excel)\n"
            msg_panel += "• _performance_report.json (Métricas)"
            
            self.lbl_files_info.setText(msg_panel)
            self.lbl_files_info.setStyleSheet("color: #333; font-size: 11px; margin-top: 10px; background-color: #f0f8ff; padding: 10px; border-radius: 5px;")

            self.stats_panel.setVisible(True)
            
            self.log("="*40)
            self.log("RESUMEN DE ARCHIVOS GENERADOS:")
            self.log(f"Directorio: {output_dir}")
            if self.current_image:
                base = os.path.basename(self.current_image).split('.')[0]
                self.log(f" -> {base}_predicted.png")
                self.log(f" -> {base}_predicted.tif")
                self.log(f" -> {base}_predicted_poly.gpkg")
                self.log(f" -> {base}_predicted_atributos.csv")
            self.log("="*40)

            if self.current_image:
                report_path = os.path.join(output_dir, f"{os.path.basename(self.current_image).split('.')[0]}_reporte_conteo.txt")
                with open(report_path, 'w') as f:
                    f.write(f"Total: {total}\nMauritia: {mau}\nEuterpe: {eut}\nOenocarpus: {oeno}")

        except Exception as e:
            self.log(f"Error actualizando estadísticas: {e}")
        
    def _load_statistics_csv(self, base_name, output_dir):
        """
        Función auxiliar unificada para cargar estadísticas desde cualquier CSV generado
        Prioriza: _predicted_summary.csv → _atributos.csv → _predicted_atributos.csv
        """
        posibles_csv = [
            f"{base_name}_predicted_summary.csv",      # Resumen directo (modo optimizado)
            f"{base_name}_atributos.csv",              # Lista detallada (modo tiles)
            f"{base_name}_predicted_atributos.csv",    # Nombre antiguo (otros modos)
            "resultados_atributos.csv"
        ]

        csv_encontrado = None
        for csv_file in posibles_csv:
            csv_path = os.path.join(output_dir, csv_file)
            if os.path.exists(csv_path):
                csv_encontrado = csv_path
                self.log(f"📊 CSV encontrado: {csv_file}")
                break

        if not csv_encontrado:
            self.log("⚠️ No se encontró ningún archivo de estadísticas")
            return

        try:
            import pandas as pd
            df = pd.read_csv(csv_encontrado)
            self.log(f"   Columnas del CSV: {list(df.columns)}")

            # Caso 1: CSV de resumen (tiene columna 'CONTEO')
            if 'CONTEO' in [c.upper() for c in df.columns]:
                self.log("   Tipo: Resumen directo")
                df.columns = [c.strip().upper() for c in df.columns]
                
                mau = int(df.loc[df['ESPECIE'].str.contains('MAURITIA', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('MAURITIA', case=False)].empty else 0
                eut = int(df.loc[df['ESPECIE'].str.contains('EUTERPE', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('EUTERPE', case=False)].empty else 0
                oeno = int(df.loc[df['ESPECIE'].str.contains('OENOCARPUS', case=False, na=False), 'CONTEO'].iloc[0]) if not df[df['ESPECIE'].str.contains('OENOCARPUS', case=False)].empty else 0
                
                total_row = df[df['ESPECIE'].str.contains('TOTAL', case=False, na=False)]
                total = int(total_row['CONTEO'].iloc[0]) if not total_row.empty else (mau + eut + oeno)
            
            # Caso 2: CSV de listado (una fila por palmera)
            else:
                self.log("   Tipo: Listado detallado")
                if 'ESPECIE' in df.columns:
                    mau = len(df[df['ESPECIE'].str.contains('Mauritia flexuosa', case=False, na=False)])
                    eut = len(df[df['ESPECIE'].str.contains('Euterpe precatoria', case=False, na=False)])
                    oeno = len(df[df['ESPECIE'].str.contains('Oenocarpus bataua', case=False, na=False)])
                    total = len(df)
                else:
                    self.log("   No se encontró columna ESPECIE")
                    mau = eut = oeno = total = len(df)

            # Construir texto para el panel
            stats_text = (
                f"Mauritia flexuosa: {mau}\n"
                f"Euterpe precatoria: {eut}\n"
                f"Oenocarpus bataua: {oeno}"
            )
            
            self.update_statistics_panel(stats_text)
            self.log(f"✅ Estadísticas mostradas: Total {total} palmeras (Mauritia: {mau}, Euterpe: {eut}, Oenocarpus: {oeno})")

        except Exception as e:
            self.log(f"❌ Error leyendo CSV: {e}")    

    def change_view(self, view_name):
        if not self.current_image:
            return
            
        base_name = os.path.basename(self.current_image).split('.')[0]
        output_dir = self.processor.config["output"]["directory"]
        
        try:
            if view_name == "Imagen Original":
                self.main_viewer.load_image_smart(self.current_image)
                self.current_view = "original"
                
            elif view_name == "Segmentación":
                seg_image_path = os.path.join(output_dir, f"{base_name}_balanced_argmax.png")
                if os.path.exists(seg_image_path):
                    self.main_viewer.load_image_smart(seg_image_path)
                    self.current_view = "segmentacion"
                else:
                    self.log(f"Imagen de segmentación no encontrada: {seg_image_path}")
                    
            elif view_name == "Conteo de Instancias":
                inst_image_path = os.path.join(output_dir, f"{base_name}_predicted.png")
                if os.path.exists(inst_image_path):
                    self.main_viewer.load_image_smart(inst_image_path)
                    self.current_view = "instancias"
                else:
                    self.log(f"Imagen de conteo no encontrada: {inst_image_path}")
                    
            self.log(f"Cambiando a vista: {view_name}")
            
        except Exception as e:
            self.log(f"Error cambiando vista: {str(e)}")
        
    def update_memory_info(self):
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            self.memory_info_label.setText(
                f"Memoria: {available_gb:.1f}GB libres de {total_gb:.1f}GB total"
            )
            
            QTimer.singleShot(5000, self.update_memory_info)
        except:
            self.memory_info_label.setText("Memoria: info no disponible")
    
    def zoom_in(self):
        self.main_viewer.zoom_in()
    
    def zoom_out(self):
        self.main_viewer.zoom_out()
    
    def reset_zoom(self):
        self.main_viewer.reset_zoom()
        
    def setup_drag_drop(self):
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                if self.is_valid_image(file_path):
                    event.acceptProposedAction()
                    self.setStyleSheet("MainWindow { background-color: #e8f5e8; }")
                else:
                    event.ignore()
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("")
        event.accept()

    def dropEvent(self, event):
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            if self.is_valid_image(file_path):
                self.load_image_file(file_path)
                event.acceptProposedAction()
            else:
                QMessageBox.warning(self, "Formato no valido", 
                                    "Por favor, selecciona una imagen valida (PNG, JPG, JPEG, TIFF, BMP)")
                event.ignore()
        else:
            event.ignore()

    def is_valid_image(self, file_path):
        valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        if not os.path.isfile(file_path):
            return False
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in valid_extensions

    def load_image_file(self, file_path):
        try:
            self.load_progress_bar.setVisible(True)
            self.load_progress_bar.setValue(0)
            
            self.current_image = file_path
            filename = os.path.basename(file_path)
            self.image_label.setText(f"Cargando {filename}...")
            self.process_btn.setEnabled(False)
            self.log(f"Cargando imagen: {filename}")
            
            self.load_progress_bar.setValue(30)
            file_size = os.path.getsize(file_path) / 1024 / 1024
            self.log(f"Tamaño: {file_size:.2f} MB")
            
            self.load_progress_bar.setValue(60)
            success = self.main_viewer.load_image_smart(file_path)
                
            if not success:
                self.log("Usando método de carga tradicional...")
                self.display_image_fallback(file_path, self.main_viewer, "original")
            
            self.load_progress_bar.setValue(90)
            self.view_selector.setCurrentIndex(0)
            self.view_selector.setEnabled(True)
            self.stats_panel.setVisible(False)
            
            # --- LIMPIAR DETALLES ANTERIORES AL CARGAR NUEVA IMAGEN ---
            if hasattr(self, 'scroll_details'):
                self.scroll_details.setVisible(False)
                self.lbl_performance_details.setText("")
            # ----------------------------------------------------------
            
            self.load_progress_bar.setValue(100)
            self.image_label.setText(f"OK {filename}")
            self.process_btn.setEnabled(True)
            self.log(f"OK Imagen cargada: {filename}")
            self.log(f"Ruta: {file_path}")
            
            QTimer.singleShot(1000, lambda: self.load_progress_bar.setVisible(False))
            
        except Exception as e:
            error_msg = f"Error al cargar imagen: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
            self.load_progress_bar.setVisible(False)

    def display_image_fallback(self, image_path, viewer, image_type):
        try:
            if image_path.lower().endswith(('.tif', '.tiff')):
                try:
                    pil_image = Image.open(image_path)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    pil_image.thumbnail((600, 600))
                    data = pil_image.tobytes("raw", "RGB")
                    qimage = QImage(data, pil_image.size[0], pil_image.size[1], QImage.Format_RGB888)
                    
                    if not qimage.isNull():
                        pixmap = QPixmap.fromImage(qimage)
                        viewer.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        viewer.setText("")
                        viewer.setToolTip(f"{image_type.capitalize()}\nTamaño: {pil_image.size[0]}x{pil_image.size[1]}")
                        viewer.data_ref = None 
                        return True
                        
                except Exception as pil_error:
                    self.log(f"PIL no pudo cargar el TIFF: {pil_error}")
                    return False
            else:
                qimage = QImage(image_path)
                if not qimage.isNull():
                    pixmap = QPixmap.fromImage(qimage)
                    viewer.setPixmap(pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    viewer.setText("")
                    original_size = f"{pixmap.width()}x{pixmap.height()}"
                    viewer.setToolTip(f"{image_type.capitalize()}\nTamaño: {original_size}")
                    viewer.data_ref = None
                    return True
            
            return False
                
        except Exception as e:
            self.log(f"ERROR mostrando imagen {image_type}: {str(e)}")
            return False

    
    def select_image(self):
        try:
            file_filters = (
                "Todos los archivos de imagen (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;"
                "Imágenes TIFF (*.tiff *.tif);;"
                "Imágenes JPEG (*.jpg *.jpeg);;"
                "Imágenes PNG (*.png);;"
                "Imágenes BMP (*.bmp);;"
                "Todos los archivos (*.*)"
            )
            
            initial_dir = os.path.expanduser("~") 
            if os.path.exists("C:/"): 
                initial_dir = "C:/"
            elif os.path.exists("/home"): 
                initial_dir = "/home"
            
            file_path, selected_filter = QFileDialog.getOpenFileName(
                self, 
                "Seleccionar imagen - Palmaria", 
                initial_dir,
                file_filters,
                "Todos los archivos de imagen (*.png *.jpg *.jpeg *.tiff *.tif *.bmp)"
            )
            
            self.log(f"Dialogo retorno: {file_path if file_path else 'Ninguna seleccion'}")
            self.log(f"Filtro seleccionado: {selected_filter}")
            
            if file_path:
                if not os.path.exists(file_path):
                    QMessageBox.warning(self, "Archivo no encontrado", 
                                        f"El archivo no existe:\n{file_path}")
                    return
                    
                if not os.path.isfile(file_path):
                    QMessageBox.warning(self, "Error", "La ruta seleccionada no es un archivo")
                    return
                    
                valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext not in valid_extensions:
                    QMessageBox.warning(self, "Formato no soportado",
                                        f"Formato de archivo no soportado: {file_ext}\n"
                                        f"Formatos válidos: {', '.join(valid_extensions)}")
                    return
                
                self.load_image_file(file_path)
                
            else:
                self.log("No se selecciono ninguna imagen")
                
        except Exception as e:
            error_msg = f"Error al seleccionar imagen: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)

    def open_in_explorer(self):
        try:
            current_dir = os.getcwd()
            if os.name == 'nt': 
                os.system(f'explorer "{current_dir}"')
            elif os.name == 'posix': 
                os.system(f'xdg-open "{current_dir}"')
            self.log(f"Explorador abierto en: {current_dir}")
        except Exception as e:
            error_msg = f"No se pudo abrir el explorador: {str(e)}"
            self.log(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"{value}%")
            
    def log(self, message):
        self.console.append(f"{message}")
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum()
        )
        
    def check_permissions(self):
        try:
            self.log("Verificando permisos del sistema...")
            current_dir = os.getcwd()
            self.log(f"Directorio actual: {current_dir}")
            
            test_file = os.path.join(current_dir, "test_permission.txt")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                self.log("Permisos de escritura: OK")
            except:
                self.log("Advertencia: Permisos de escritura limitados")
            
            important_dirs = ['scripts', 'output', 'app', 'models']
            for dir_name in important_dirs:
                if os.path.exists(dir_name):
                    self.log(f"Directorio {dir_name}: Existe")
                else:
                    self.log(f"Directorio {dir_name}: No existe")
                    
            scripts_to_check = ['segmentacion.py', 'instancias.py', 'process_with_tiles.py', 'instancias_tiles.py']
            for script in scripts_to_check:
                script_path = os.path.join('scripts', script)
                if os.path.exists(script_path):
                    self.log(f"Script {script}: Encontrado")
                else:
                    self.log(f"Script {script}: No encontrado")
                    
            self.log("Verificacion de permisos completada")
            
        except Exception as e:
            self.log(f"ERROR en verificacion de permisos: {str(e)}")

    def closeEvent(self, event):
        # --- NUEVO: VERIFICAR SI HAY UN PROCESO CORRIENDO ---
        if hasattr(self, 'thread') and self.thread.isRunning():
            warning = QMessageBox.warning(self, 'Proceso en Ejecución',
                                        '⚠️ ¡CUIDADO!\n\n'
                                        'Se está ejecutando un proceso de segmentación.\n'
                                        'Si cierra la aplicación ahora, el proceso se interrumpirá y podría perder datos.\n\n'
                                        '¿Realmente desea forzar el cierre?',
                                        QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.No)
            if warning == QMessageBox.No:
                event.ignore()
                return
        # ----------------------------------------------------

        reply = QMessageBox.question(self, 'Salir',
                                     '¿Estás seguro de que quieres salir?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.log("Cerrando aplicacion...")
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        
        try:
            from processor import PalmProcessor
        except ImportError:
            print("Error: El módulo 'processor' no se encontró. Asegúrate de que processor.py existe.")
            sys.exit(1)
            
        try:
            import numpy as np
        except ImportError:
            print("Error: La librería 'numpy' no se encontró. Instálala con 'pip install numpy'.")
            sys.exit(1)

        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())
    except ImportError as e:
        print(f"Error de importación: {e}")
        print("Asegúrate de tener instaladas las librerías 'PyQt5', 'Pillow', 'gdal', y 'numpy'.")
        sys.exit(1)
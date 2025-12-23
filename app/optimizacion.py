# E:\IIAP\IIAPFLEX\aplicativo\app\optimizacion.py
"""
Módulo de Optimización de Recursos para la Aplicación de Segmentación de Palmeras
Compatible con la configuración existente en config.json
"""

import os
import gc
import psutil
import time
import threading
import json
from datetime import datetime
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
import onnxruntime as ort

class MemoryManager(QObject):
    """
    Gestor de memoria para optimizar el uso de RAM durante el procesamiento
    """
    memory_warning = pyqtSignal(str)  # Señal para advertencias de memoria
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.max_memory_usage = self.config.get("optimization", {}).get("memory_management", {}).get("max_memory_usage", 0.85)  # Aumentado a 85%
        self.monitoring = False
        self.monitor_thread = None
        self.last_warning_time = 0  # Para evitar advertencias repetidas
        self.warning_cooldown = 30  # Segundos entre advertencias
        
    def get_memory_info(self):
        """Obtener información detallada de la memoria"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent,
            'available_percent': 100 - memory.percent
        }
    
    def can_process_image(self, image_path):
        """
        Verificar si hay suficiente memoria para procesar una imagen
        Considera el tamaño del archivo y la memoria disponible
        """
        try:
            if not os.path.exists(image_path):
                return False, "Archivo no encontrado"
            
            image_size_gb = os.path.getsize(image_path) / (1024**3)
            memory_info = self.get_memory_info()
            
            # Estimación más realista: imagen puede requerir 2x su tamaño en memoria
            estimated_need_gb = image_size_gb * 2  # Reducido de 4x a 2x
            available_gb = memory_info['available_gb']
            
            # Verificar memoria disponible
            if available_gb < estimated_need_gb:
                return False, f"Memoria insuficiente. Necesario: {estimated_need_gb:.2f}GB, Disponible: {available_gb:.2f}GB"
            
            # Verificar porcentaje máximo de uso (más permisivo)
            if memory_info['percent_used'] > (self.max_memory_usage * 100):
                return False, f"Uso de memoria muy alto: {memory_info['percent_used']:.1f}%"
            
            return True, f"OK - Imagen: {image_size_gb:.2f}GB, Memoria disponible: {available_gb:.2f}GB"
            
        except Exception as e:
            return False, f"Error verificando memoria: {str(e)}"
    
    def optimize_memory(self):
        """
        Liberar memoria no utilizada y ejecutar garbage collection
        """
        # Forzar garbage collection
        gc.collect()
        return self.get_memory_info()
    
    def start_memory_monitoring(self, interval=5):  # Aumentado intervalo a 5 segundos
        """Iniciar monitoreo continuo de memoria"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._memory_monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print("Monitor de memoria iniciado")
    
    def stop_memory_monitoring(self):
        """Detener monitoreo de memoria"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        print("Monitor de memoria detenido")
    
    def _memory_monitor_loop(self, interval):
        """Loop de monitoreo de memoria CORREGIDO - MENOS SENSIBLE"""
        warning_count = 0
        max_warnings = 3  # Máximo de advertencias antes de silenciar
        
        while self.monitoring:
            try:
                # MONITOREO CORREGIDO: Usar memoria del proceso actual
                process = psutil.Process()
                process_memory_mb = process.memory_info().rss / (1024 * 1024)
                
                current_time = time.time()
                
                # Umbral más realista para NUESTRO proceso (2GB)
                if process_memory_mb > 2000:  # 2GB para nuestro proceso
                    if (current_time - self.last_warning_time > self.warning_cooldown and
                        warning_count < max_warnings):
                        
                        warning_msg = f"Alto uso de memoria de la aplicación: {process_memory_mb:.1f} MB"
                        self.memory_warning.emit(warning_msg)
                        self.last_warning_time = current_time
                        warning_count += 1
                        print(f"Advertencia de memoria #{warning_count}: {process_memory_mb:.1f} MB")
                
                # Reiniciar contador si la memoria mejora
                if process_memory_mb < 1000:
                    warning_count = 0
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error en monitoreo de memoria: {e}")
                time.sleep(interval)


class PerformanceOptimizer:
    """
    Optimizador de rendimiento para el procesamiento de imágenes
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.performance_metrics = {
            'segmentacion_times': [],
            'instancias_times': [],
            'memory_peak': [],
            'cpu_peak': []
        }
        self.current_process = None
        
    def optimize_onnx_session(self, model_path):
        """
        Configurar sesión ONNX optimizada para el hardware disponible
        """
        try:
            # Opciones de optimización para ONNX Runtime
            sess_options = ort.SessionOptions()
            
            # Configurar nivel de optimización
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Configurar para mejor rendimiento en CPU
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_profiling = False
            
            # Configurar número de hilos (ajustar según CPU)
            cpu_count = os.cpu_count() or 4
            sess_options.intra_op_num_threads = min(cpu_count, 4)  # Reducido a 4 hilos máximo
            sess_options.inter_op_num_threads = 1
            
            # Crear sesión optimizada
            providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, sess_options, providers=providers)
            
            print(f"Sesión ONNX optimizada - Hilos: {sess_options.intra_op_num_threads}")
            return session
            
        except Exception as e:
            print(f"Error optimizando sesión ONNX: {e}")
            return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    def get_optimal_parameters(self, image_path, process_type):
        """
        Calcular parámetros óptimos según tamaño de imagen y memoria disponible
        """
        try:
            if not os.path.exists(image_path):
                return None
                
            image_size_gb = os.path.getsize(image_path) / (1024**3)
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            
            # Tamaños base según memoria disponible (más conservadores)
            if available_gb > 16:
                base_window = 400  # Reducido
            elif available_gb > 8:
                base_window = 300  # Reducido
            else:
                base_window = 256  # Mantenido
            
            # Ajustar según tamaño de imagen
            if image_size_gb > 0.5:  # Más sensible
                base_window = max(200, base_window - 50)
            elif image_size_gb < 0.05:  # Imágenes muy pequeñas
                base_window = min(400, base_window + 50)
            
            return {
                "window_radius": base_window,
                "internal_window_radius": int(base_window * 0.75)
            }
            
        except Exception as e:
            print(f"Error calculando parámetros óptimos: {e}")
            return None
    
    def log_performance_metric(self, stage, duration, memory_usage=None, cpu_usage=None):
        """Registrar métricas de rendimiento"""
        metric = {
            'stage': stage,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'memory_usage': memory_usage or psutil.virtual_memory().percent,
            'cpu_usage': cpu_usage or psutil.cpu_percent()
        }
        
        if stage == 'segmentacion':
            self.performance_metrics['segmentacion_times'].append(metric)
        elif stage == 'instancias':
            self.performance_metrics['instancias_times'].append(metric)
        
        # Mantener solo las últimas 5 mediciones
        if len(self.performance_metrics['segmentacion_times']) > 5:
            self.performance_metrics['segmentacion_times'].pop(0)
        if len(self.performance_metrics['instancias_times']) > 5:
            self.performance_metrics['instancias_times'].pop(0)
    
    def get_performance_report(self):
        """Generar reporte de rendimiento"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'memory': psutil.virtual_memory()._asdict(),
                'cpu_count': os.cpu_count(),
                'cpu_percent': psutil.cpu_percent()
            },
            'average_times': {},
            'recommendations': []
        }
        
        # Calcular promedios
        if self.performance_metrics['segmentacion_times']:
            seg_times = [m['duration'] for m in self.performance_metrics['segmentacion_times']]
            report['average_times']['segmentacion'] = sum(seg_times) / len(seg_times)
        
        if self.performance_metrics['instancias_times']:
            inst_times = [m['duration'] for m in self.performance_metrics['instancias_times']]
            report['average_times']['instancias'] = sum(inst_times) / len(inst_times)
        
        return report


class ResourceMonitor(QObject):
    """
    Monitor integral de recursos del sistema
    """
    status_update = pyqtSignal(dict)  # Señal para actualizaciones de estado
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval=5):  # Aumentado intervalo
        """Iniciar monitoreo completo del sistema"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print("Monitor de sistema iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        print("Monitor de sistema detenido")
    
    def _monitor_loop(self, interval):
        """Loop principal de monitoreo"""
        while self.monitoring:
            try:
                status = self.get_system_status()
                self.status_update.emit(status)
                time.sleep(interval)
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                time.sleep(interval)
    
    def get_system_status(self):
        """Obtener estado completo del sistema"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'total_gb': round(memory.total / (1024**3), 1),
                'used_gb': round(memory.used / (1024**3), 1),
                'available_gb': round(memory.available / (1024**3), 1),
                'percent': memory.percent
            },
            'cpu': {
                'percent': cpu,
                'cores': os.cpu_count()
            },
            'disk': {
                'total_gb': round(disk.total / (1024**3), 1),
                'free_gb': round(disk.free / (1024**3), 1),
                'percent': disk.percent
            },
            'process': {
                'memory_mb': round(psutil.Process().memory_info().rss / (1024**2), 1),
                'cpu_percent': psutil.Process().cpu_percent()
            }
        }

    def check_min_requirements(self):
        """
        Compara el sistema actual con los requisitos mínimos recomendados
        Retorna un diccionario con la comparación
        """
        # Requisitos mínimos definidos
        MIN_RAM_GB = 8
        MIN_CORES = 4
        MIN_DISK_GB = 10
        
        # Obtener estado actual
        current = self.get_system_status()
        
        # Realizar comparación
        analysis = {
            "ram": {
                "item": "Memoria RAM",
                "min": f"{MIN_RAM_GB} GB",
                "actual": f"{current['memory']['total_gb']} GB",
                "status": current['memory']['total_gb'] >= MIN_RAM_GB
            },
            "cpu": {
                "item": "Procesador (Núcleos)",
                "min": f"{MIN_CORES} Núcleos",
                "actual": f"{current['cpu']['cores']} Núcleos",
                "status": current['cpu']['cores'] >= MIN_CORES
            },
            "disk": {
                "item": "Espacio en Disco (Libre)",
                "min": f"{MIN_DISK_GB} GB",
                "actual": f"{current['disk']['free_gb']} GB",
                "status": current['disk']['free_gb'] >= MIN_DISK_GB
            }
        }
        
        # Evaluación global
        meets_requirements = all(item['status'] for item in analysis.values())
        
        return meets_requirements, analysis


class PerformanceMonitor(QObject):
    """
    Monitor de rendimiento en tiempo real que calcula tiempo estimado y consumo de recursos
    """
    performance_update = pyqtSignal(dict)  # Señal para actualizaciones de rendimiento
    memory_alert = pyqtSignal(str)  # Señal para alertas de memoria
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.process = psutil.Process()
        
        # Estadísticas históricas
        self.performance_history = []
        self.memory_peaks = []
        
    def start_monitoring(self):
        """Iniciar monitoreo de rendimiento"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Detener monitoreo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitoring_loop(self):
        """Loop principal de monitoreo"""
        while self.monitoring:
            try:
                status = self.get_current_status()
                self.performance_update.emit(status)
                
                # Guardar en historial
                self.performance_history.append(status)
                self.memory_peaks.append(status['memory_used_mb'])
                
                # Mantener solo últimas 1000 mediciones
                if len(self.performance_history) > 1000:
                    self.performance_history.pop(0)
                if len(self.memory_peaks) > 1000:
                    self.memory_peaks.pop(0)
                
                # Verificar alertas de memoria - CORREGIDO
                process_memory_mb = self.process.memory_info().rss / (1024 * 1024)
                if process_memory_mb > 2000:  # 2GB para nuestro proceso
                    self.memory_alert.emit(f"Alto uso de memoria de la aplicación: {process_memory_mb:.1f} MB")
                    
                time.sleep(2)  # Actualizar cada 2 segundos
                
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                time.sleep(5)
    
    def get_current_status(self):
        """Obtener estado actual del sistema"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        # Memoria del proceso actual
        process_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        # Tiempo transcurrido
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': elapsed_time,
            'elapsed_time_formatted': self._format_time(elapsed_time),
            'memory_used_mb': process_memory,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu,
            'process_memory_mb': process_memory
        }
    
    # En la clase PerformanceMonitor, mejorar el método calculate_eta:

    def calculate_eta(self, progress_percent, elapsed_time):
        """Calcular tiempo estimado de finalización MEJORADO"""
        if progress_percent <= 0:
            return "Calculando..."
        
        # MEJORA: Cálculo más preciso considerando progreso no lineal
        if progress_percent < 10:
            # Al inicio, ser más conservador
            estimated_total = elapsed_time / (progress_percent / 100) * 1.5
        elif progress_percent > 90:
            # Al final, ser más optimista
            estimated_total = elapsed_time / (progress_percent / 100) * 0.9
        else:
            # En medio, cálculo normal
            estimated_total = elapsed_time / (progress_percent / 100)
        
        remaining_time = estimated_total - elapsed_time
        
        return self._format_time(max(0, remaining_time))
    
    def _format_time(self, seconds):
        """Formatear tiempo en formato legible"""
        if seconds < 60:
            return f"{int(seconds)} segundos"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def generate_final_report(self, processing_mode, image_info, success=True):
        """Generar reporte final del procesamiento"""
        final_status = self.get_current_status()
        
        report = {
            'processing_summary': {
                'mode_used': processing_mode,
                'success': success,
                'total_processing_time': final_status['elapsed_time_formatted'],
                'completion_time': datetime.now().isoformat()
            },
            'resource_consumption': {
                'peak_memory_mb': max(self.memory_peaks) if self.memory_peaks else final_status['memory_used_mb'],
                'average_cpu_percent': np.mean([s['cpu_percent'] for s in self.performance_history]) if self.performance_history else final_status['cpu_percent'],
                'final_memory_mb': final_status['memory_used_mb'],
                'memory_efficiency': f"{(final_status['memory_used_mb'] / (psutil.virtual_memory().total / (1024 * 1024)) * 100):.1f}%"
            },
            'image_info': image_info,
            'performance_metrics': {
                'total_steps': len(self.performance_history),
                'average_memory_usage_mb': np.mean([s['memory_used_mb'] for s in self.performance_history]) if self.performance_history else 0,
                'memory_usage_peaks': self.memory_peaks
            }
        }
        
        return report


# Instancias globales
memory_manager = None
performance_optimizer = None
resource_monitor = None
performance_monitor = PerformanceMonitor()  # Nueva instancia global

def initialize_optimization_system(config=None):
    """
    Inicializar todo el sistema de optimización
    """
    global memory_manager, performance_optimizer, resource_monitor
    
    print("Inicializando sistema de optimización...")
    
    try:
        # Crear instancias
        memory_manager = MemoryManager(config)
        performance_optimizer = PerformanceOptimizer(config)
        resource_monitor = ResourceMonitor(config)
        
        # Obtener configuración
        optimization_config = config.get("optimization", {}) if config else {}
        
        # Solo iniciar monitores si están explícitamente habilitados
        memory_monitoring = optimization_config.get("memory_management", {}).get("enable_memory_monitoring", False)  # Por defecto False
        system_monitoring = optimization_config.get("monitoring", {}).get("enable_system_monitoring", True)
        
        if memory_monitoring:
            memory_manager.start_memory_monitoring()
        else:
            print("Monitor de memoria deshabilitado")
        
        if system_monitoring:
            resource_monitor.start_monitoring()
        
        memory_info = memory_manager.get_memory_info()
        
        print("Sistema de optimización inicializado")
        print(f"   Memoria: {memory_info['available_gb']:.1f}GB disponible ({memory_info['available_percent']:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error inicializando sistema de optimización: {e}")
        return False

def shutdown_optimization_system():
    """
    Apagar sistema de optimización
    """
    global memory_manager, resource_monitor
    
    print("Apagando sistema de optimización...")
    
    try:
        if memory_manager:
            memory_manager.stop_memory_monitoring()
        if resource_monitor:
            resource_monitor.stop_monitoring()
        print("Sistema de optimización apagado")
    except Exception as e:
        print(f"Error apagando sistema de optimización: {e}")

if __name__ == "__main__":
    # Pruebas del módulo
    initialize_optimization_system()
    
    if resource_monitor:
        print("\nEstado del sistema:")
        status = resource_monitor.get_system_status()
        for category, data in status.items():
            if category != 'timestamp':
                print(f"   {category.upper()}: {data}")
    
    time.sleep(2)
    shutdown_optimization_system()
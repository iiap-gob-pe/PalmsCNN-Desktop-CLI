"""
Paquete de la aplicación de Segmentación de Palmeras
"""

__version__ = "3.1.0"
__author__ = "Tu Nombre"
__description__ = "Aplicación para segmentación y conteo de palmeras usando IA"

# Exportar clases principales
from .gui import MainWindow
from .processor import PalmProcessor
from .tile_processor import EfficientTileProcessor
from .memory_optimizer import memory_optimizer
from .optimizacion import (
    MemoryManager, 
    PerformanceOptimizer, 
    ResourceMonitor, 
    PerformanceMonitor,
    initialize_optimization_system
)
from .single_instance import instance_checker

# Inicializar sistema de optimización al importar
try:
    import json
    import os
    
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        initialize_optimization_system(config)
    else:
        initialize_optimization_system()
except:
    pass

__all__ = [
    'MainWindow',
    'PalmProcessor',
    'EfficientTileProcessor',
    'memory_optimizer',
    'MemoryManager',
    'PerformanceOptimizer',
    'ResourceMonitor',
    'PerformanceMonitor',
    'instance_checker'
]
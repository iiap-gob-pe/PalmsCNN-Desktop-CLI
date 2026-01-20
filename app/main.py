#!/usr/bin/env python3
"""
Punto de entrada principal de la aplicación de Segmentación de Palmeras
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

def main():
    """Función principal de la aplicación"""
    try:
        # Verificar instancia única antes de iniciar Qt
        try:
            from single_instance import instance_checker
            if not instance_checker.check():
                print("¡La aplicación ya está en ejecución!")
                return
        except ImportError:
            print("Advertencia: No se pudo verificar instancia única")
        
        # Crear aplicación Qt
        app = QApplication(sys.argv)
        app.setApplicationName("Segmentación de Palmeras")
        app.setApplicationVersion("3.0")
        
        # Importar y crear ventana principal
        from gui import MainWindow
        
        # Crear y mostrar ventana principal
        main_window = MainWindow()
        main_window.show()
        
        # Ejecutar aplicación
        print("Aplicación iniciada correctamente")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error crítico al iniciar la aplicación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ejecutar aplicación de segmentación de palmeras - VERSIÓN CON MODO OPTIMIZADO
"""

import sys
import os
import traceback

# Agregar directorio app al path
app_dir = os.path.join(os.path.dirname(__file__), 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Cambiar al directorio del proyecto
os.chdir(os.path.dirname(__file__))

def check_dependencies():
    """Verificar dependencias críticas"""
    required_modules = [
        'PyQt5',
        'numpy',
        'PIL',
        'rasterio',
        'geopandas',
        'onnxruntime'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            missing.append(f"{module}: {str(e)}")
    
    return missing

def main():
    """Función principal de la aplicación"""
    try:
        print("=" * 60)
        print("🌴 SEGMENTACIÓN DE PALMERAS - MODO OPTIMIZADO")
        print("=" * 60)
        
        # Verificar instancia única antes de iniciar Qt
        try:
            from app.single_instance import instance_checker
            if not instance_checker.check():
                print("¡La aplicación ya está en ejecución!")
                print("Cierra la instancia anterior o reinicia el sistema.")
                input("Presiona Enter para salir...")
                return
        except ImportError:
            print("⚠️ Advertencia: No se pudo verificar instancia única")
        
        # Verificar dependencias
        missing_deps = check_dependencies()
        if missing_deps:
            print("❌ ERROR: Faltan dependencias críticas:")
            for dep in missing_deps:
                print(f"   • {dep}")
            print("\nInstala las dependencias con:")
            print("   pip install -r requirements.txt")
            input("\nPresiona Enter para salir...")
            return
        
        # Crear aplicación Qt
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Segmentación de Palmeras")
        app.setApplicationVersion("3.1.0")
        
        # Importar y crear ventana principal
        from app.gui import MainWindow
        
        # Crear y mostrar ventana principal
        main_window = MainWindow()
        main_window.show()
        
        print("✅ Aplicación iniciada correctamente")
        print(f"📁 Directorio actual: {os.getcwd()}")
        print("=" * 60)
        
        # Ejecutar aplicación
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ Error crítico al iniciar la aplicación: {e}")
        traceback.print_exc()
        
        # Mensaje para el usuario
        import tkinter as tk
        from tkinter import messagebox
        
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Error Crítico",
                f"No se pudo iniciar la aplicación:\n\n{str(e)}\n\n"
                f"Verifica que todas las dependencias estén instaladas."
            )
        except:
            pass
        
        input("\nPresiona Enter para salir...")
        sys.exit(1)

if __name__ == "__main__":
    main()
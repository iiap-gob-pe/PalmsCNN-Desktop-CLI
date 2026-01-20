# Archivo: aplicativo2/main_cli.py
import sys
import os
import argparse
import time

# --- 1. Configuración de rutas para encontrar tus módulos 'app' ---
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from app.processor import PalmProcessor  # <--- AGREGA "app."
except ImportError as e:
    print(f"❌ Error crítico: No se encuentra el módulo 'app'. {e}")
    sys.exit(1)

# --- 2. Función visual para ver el progreso en la consola ---
def cli_progress(value, message=""):
    """Dibuja una barra de carga: [██████----] 60%"""
    bar_len = 30
    filled = int(bar_len * value // 100)
    bar = '█' * filled + '-' * (bar_len - filled)
    # \r permite sobrescribir la línea para animar
    sys.stdout.write(f'\r[{bar}] {int(value)}% {message}')
    sys.stdout.flush()
    if value >= 100: print()

def main():
    # --- 3. Definir los "Parámetros" (Argumentos) ---
    parser = argparse.ArgumentParser(description="CLI Segmentación de Palmeras (IIAPFLEX)")
    
    # Parámetro 1: La imagen (Obligatorio)
    parser.add_argument("image", help="Ruta de la imagen .tif a procesar")
    
    # Parámetro 2: Modo de uso (Opcional, por defecto 'auto')
    parser.add_argument("--mode", choices=['auto', 'optimized', 'low_memory'], default='auto',
                        help="Estrategia de procesamiento")
    
    # Parámetro 3: Carpeta de salida (Opcional)
    parser.add_argument("--output", default="output", help="Carpeta de resultados")

    args = parser.parse_args()

    # --- 4. Validaciones ---
    if not os.path.exists(args.image):
        print(f"❌ Error: La imagen '{args.image}' no existe.")
        sys.exit(1)

    print("\n" + "="*50)
    print(f"🌴 PROCESANDO: {os.path.basename(args.image)}")
    print(f"⚙️  MODO:       {args.mode.upper()}")
    print(f"📂 SALIDA:     {args.output}")
    print("="*50 + "\n")

    try:
        # Inicializar el procesador (Tu cerebro existente)
        processor = PalmProcessor()
        
        # Configurar carpeta de salida
        processor.config["output"]["directory"] = args.output
        os.makedirs(args.output, exist_ok=True)

        # Configurar lógica según el parámetro --mode
        use_opt = False
        force_tile = False
        
        if args.mode == 'optimized':
            use_opt = True # Usa tus scripts nuevos (process_with_tiles.py)
            print("🚀 Usando scripts optimizados...")
        elif args.mode == 'low_memory':
            force_tile = True
            processor.config["optimization"]["memory_management"]["low_ram_mode"] = True
            print("🔋 Usando modo ahorro de RAM...")

        # --- 5. EJECUTAR ---
        start_time = time.time()
        
        result = processor.process_image(
            args.image,
            force_tiling=force_tile,
            use_optimized_tiles=use_opt,
            progress_callback=cli_progress # Conectar la barra visual
        )

        elapsed = time.time() - start_time
        print("\n" + "="*50)
        print(f"✅ FINALIZADO en {elapsed:.2f} segundos")
        print(f"📝 Resultado: {result}")
        print("="*50)

    except KeyboardInterrupt:
        print("\n⚠️ Proceso cancelado por el usuario.")
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
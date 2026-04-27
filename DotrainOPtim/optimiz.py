import os
import sys
import shutil
import subprocess
from pathlib import Path
from ultralytics import YOLO
import torch
import warnings

warnings.filterwarnings('ignore')

MODEL_NAME = "26n.pt" 
OUTPUT_DIR = "optimized_models"
IMGSZ = 640
INT8_DATA = None  

def get_script_dir():
    return Path(__file__).parent.absolute()

def get_file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)

def check_package(package_name):
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def install_package(package):
    print(f"   Устанавливаю {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        return True
    except:
        return False

def optimize_fp16(model_path, output_dir, base_name):
    print("\n" + "-"*40)
    print("🔧 FP16 ОПТИМИЗАЦИЯ")
    print("-"*40)
    
    try:
        model = YOLO(str(model_path))
        output_path = output_dir / f"{base_name}_fp16.pt"
        
        model.model.half()
        model.save(str(output_path))
        
        new_size = get_file_size_mb(output_path)
        compression = (1 - new_size / get_file_size_mb(model_path)) * 100
        
        print(f"FP16 сохранен!")
        print(f"Размер: {new_size:.2f} MB (сжатие {compression:.1f}%)")
        print(f"Файл: {output_path.name}")
        return True, output_path
        
    except Exception as e:
        print(f"Ошибка FP16: {e}")
        return False, None

def optimize_tensorrt(model_path, output_dir, base_name):
    print("\n" + "-"*40)
    print("TENSORRT ОПТИМИЗАЦИЯ")
    print("-"*40)
    
    if not torch.cuda.is_available():
        print("TensorRT требует GPU")
        return False, None
    
    try:
        try:
            import tensorrt as trt
            print(f"TensorRT версия: {trt.__version__}")
        except ImportError as e:
            print(f"TensorRT не найден: {e}")
            return False, None
        
        model = YOLO(str(model_path))
        original_dir = os.getcwd()
        os.chdir(get_script_dir())
        
        output_path = output_dir / f"{base_name}_tensorrt.engine"
        
        print("   Экспорт в TensorRT (может занять 2-5 минут)...")
        model.export(
            format='engine',
            imgsz=IMGSZ,
            half=True,
            device=0,
            workspace=4,
            verbose=True
        )
        engine_files = list(get_script_dir().glob("*.engine"))
        if engine_files:
            shutil.move(str(engine_files[0]), str(output_path))
            new_size = get_file_size_mb(output_path)
            compression = (1 - new_size / get_file_size_mb(model_path)) * 100
            
            print(f"TensorRT сохранен!")
            print(f"Размер: {new_size:.2f} MB (сжатие {compression:.1f}%)")
            print(f"Файл: {output_path.name}")
            print(f"Ожидаемое ускорение: 2-5x")
            os.chdir(original_dir)
            return True, output_path
        else:
            print("TensorRT файл не создан")
            engine_files = list(get_script_dir().glob("*.engine"))
            if engine_files:
                shutil.move(str(engine_files[0]), str(output_path))
                new_size = get_file_size_mb(output_path)
                compression = (1 - new_size / get_file_size_mb(model_path)) * 100
                print(f"Найден! TensorRT сохранен в {output_path.name}")
                os.chdir(original_dir)
                return True, output_path
            os.chdir(original_dir)
            return False, None
            
    except Exception as e:
        print(f"Ошибка TensorRT: {e}")
        return False, None

def optimize_onnx(model_path, output_dir, base_name):
    """ONNX конвертация (кросс-платформенный)"""
    print("\n" + "-"*40)
    print("ONNX КОНВЕРТАЦИЯ")
    print("-"*40)  
    try:
        model = YOLO(str(model_path))
        original_dir = os.getcwd()
        os.chdir(get_script_dir())
        
        output_path = output_dir / f"{base_name}.onnx"
        
        print("   Экспорт в ONNX...")
        model.export(
            format='onnx',
            imgsz=IMGSZ,
            opset=12,
            simplify=True
        )
        onnx_files = list(get_script_dir().glob("*.onnx"))
        if onnx_files:
            shutil.move(str(onnx_files[0]), str(output_path))
            new_size = get_file_size_mb(output_path)
            compression = (1 - new_size / get_file_size_mb(model_path)) * 100
            
            print(f"ONNX сохранен!")
            print(f"Размер: {new_size:.2f} MB (сжатие {compression:.1f}%)")
            print(f"Файл: {output_path.name}")
            os.chdir(original_dir)
            return True, output_path
        else:
            print("ONNX файл не создан")
            os.chdir(original_dir)
            return False, None
            
    except Exception as e:
        print(f"Ошибка ONNX: {e}")
        return False, None

def optimize_int8(model_path, output_dir, base_name, data_yaml):
    print("\n" + "-"*40)
    print("INT8 КВАНТИЗАЦИЯ")
    print("-"*40)
    
    if not torch.cuda.is_available():
        print("INT8")
        return False, None
    
    if not data_yaml or not Path(data_yaml).exists():
        print(f"Нет калибровочных данных: {data_yaml}")
        print("   INT8 пропущен")
        return False, None
    
    try:
        model = YOLO(str(model_path))
        original_dir = os.getcwd()
        os.chdir(get_script_dir())
        
        output_path = output_dir / f"{base_name}_int8.engine"
        
        print("   Экспорт в INT8 (может занять несколько минут)...")
        print(f"   Калибровка на: {data_yaml}")
        
        model.export(
            format='engine',
            imgsz=IMGSZ,
            half=False,
            int8=True,
            data=data_yaml,
            workspace=4,
            device=0
        )
    
        engine_files = list(get_script_dir().glob("*.engine"))
        if engine_files:
            shutil.move(str(engine_files[0]), str(output_path))
            new_size = get_file_size_mb(output_path)
            compression = (1 - new_size / get_file_size_mb(model_path)) * 100
            
            print(f"INT8 сохранен!")
            print(f"Размер: {new_size:.2f} MB (сжатие {compression:.1f}%)")
            print(f"Файл: {output_path.name}")
            print(f"Максимальное сжатие!")
            os.chdir(original_dir)
            return True, output_path
        else:
            print("INT8 файл не создан")
            os.chdir(original_dir)
            return False, None
            
    except Exception as e:
        print(f"Ошибка INT8: {e}")
        return False, None

def optimize_copy(model_path, output_dir, base_name):
    print("\n" + "-"*40)
    print("КОПИРОВАНИЕ")
    print("-"*40)
    
    try:
        output_path = output_dir / f"{base_name}_copy.pt"
        shutil.copy2(model_path, output_path)
        
        print(f"Копия сохранена!")
        print(f"   Размер: {get_file_size_mb(output_path):.2f} MB")
        print(f"   Файл: {output_path.name}")
        return True, output_path
        
    except Exception as e:
        print(f"Ошибка: {e}")
        return False, None

def main():
    print("="*60)
    print("ОПТИМИЗАЦИЯ МОДЕЛИ YOLO (ВСЕ МЕТОДЫ)")
    print("="*60)
    
    script_dir = get_script_dir()
    print(f"Папка скрипта: {script_dir}")
    
    model_path = script_dir / MODEL_NAME
    
    if not model_path.exists():
        print(f"\nМодель не найдена: {MODEL_NAME}")
        pt_files = list(script_dir.glob("*.pt"))
        if pt_files:
            print("\nНайденные .pt файлы:")
            for i, f in enumerate(pt_files, 1):
                size = get_file_size_mb(f)
                print(f"   {i}. {f.name} ({size:.2f} MB)")
            print(f"\n Измените MODEL_NAME в скрипте на одно из имен выше")
        return
    
    original_size = get_file_size_mb(model_path)
    print(f"\nМодель: {model_path.name}")
    print(f"Размер: {original_size:.2f} MB")
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Сохранение в: {output_dir}")
    
    base_name = model_path.stem
    
    print("\n" + "="*60)
    print("="*60)
    print("1. FP16 (полуточность) - работает сразу, небольшое сжатие")
    print("2. TensorRT (максимальная скорость) - требует GPU, 2-5x ускорение")
    print("3. ONNX (кросс-платформенный) - требует onnx")
    print("4. INT8 (максимальное сжатие) - требует калибровку")
    print("5. Копия модели")
    print("6. ВСЕ методы (создать все модели)")
    print("="*60)
    
    choice = input("\nВаш выбор (1-6): ").strip()
    
    results = []
    
    if choice in ['1', '6']:
        success, path = optimize_fp16(model_path, output_dir, base_name)
        if success:
            results.append(('FP16', path))
    
    if choice in ['2', '6']:
        success, path = optimize_tensorrt(model_path, output_dir, base_name)
        if success:
            results.append(('TensorRT', path))
    
    if choice in ['3', '6']:
        success, path = optimize_onnx(model_path, output_dir, base_name)
        if success:
            results.append(('ONNX', path))
    
    if choice in ['4', '6']:
        if INT8_DATA:
            data_path = INT8_DATA
        else:
            print("\n   Для INT8 нужен dataset.yaml")
            data_path = input("   Введите путь к dataset.yaml: ").strip()
        if data_path and Path(data_path).exists():
            success, path = optimize_int8(model_path, output_dir, base_name, data_path)
            if success:
                results.append(('INT8', path))
        else:
            print("\nINT8 пропущен (нет калибровочных данных)")
    
    if choice in ['5', '6']:
        success, path = optimize_copy(model_path, output_dir, base_name)
        if success:
            results.append(('COPY', path))
    
    print("\n" + "="*60)
    print("ИТОГИ ОПТИМИЗАЦИИ")
    print("="*60)
    
    if results:
        print(f"\nУспешно выполнено: {len(results)} методов\n")
        print(f"{'Метод':<12} {'Размер':<12} {'Сжатие':<12} {'Файл'}")
        print("-"*60)
        
        for method, path in results:
            size = get_file_size_mb(path)
            compression = (1 - size / original_size) * 100
            print(f"{method:<12} {size:.2f} MB     {compression:>5.1f}%      {path.name}")
        
        print("\n" + "="*60)
        print("💡 РЕКОМЕНДАЦИЯ:")
        
        if any(m == 'TensorRT' for m, _ in results):
            print("   TensorRT: МАКСИМАЛЬНАЯ СКОРОСТЬ (2-5x ускорение)")
            print(f"   Файл: {output_dir}/{base_name}_tensorrt.engine")
        elif any(m == 'INT8' for m, _ in results):
            print("   INT8: МАКСИМАЛЬНОЕ СЖАТИЕ")
        elif any(m == 'ONNX' for m, _ in results):
            print("   ONNX: КРОСС-ПЛАТФОРМЕННОСТЬ")
        elif any(m == 'FP16' for m, _ in results):
            print("   FP16: ХОРОШИЙ БАЛАНС")
        
        print(f"\nВсе файлы в: {output_dir}")
    else:
        print("\nНе удалось выполнить ни одну оптимизацию")
        print("\nВозможные причины:")
        print("  - Нет GPU для TensorRT/INT8")
        print("  - Не установлены библиотеки для ONNX")
        print("  - Нет калибровочных данных для INT8")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
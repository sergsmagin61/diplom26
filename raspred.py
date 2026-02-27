import os
import shutil
import random
from pathlib import Path

def create_yolo_structure(dataset_path, output_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"Ошибка: папка images не найдена в {dataset_path}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"Ошибка: папка labels не найдена в {dataset_path}")
        return
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("Ошибка: не найдены изображения в папке images")
        return
    
    print(f"Найдено изображений: {len(image_files)}")
    random.shuffle(image_files)
    
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        if files:
            split_images_dir = os.path.join(output_dir, split_name, 'images')
            split_labels_dir = os.path.join(output_dir, split_name, 'labels')
            
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_labels_dir, exist_ok=True)
            
            print(f"Создание {split_name} набора...")
            
            for image_file in files:
                src_image = os.path.join(images_dir, image_file)
                dst_image = os.path.join(split_images_dir, image_file)
                shutil.copy2(src_image, dst_image)
                
                label_name = os.path.splitext(image_file)[0] + '.txt'
                src_label = os.path.join(labels_dir, label_name)
                dst_label = os.path.join(split_labels_dir, label_name)
                
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    print(f"Предупреждение: аннотация не найдена для {image_file}")
    
    create_yolo_yaml(output_dir)
    classes_src = os.path.join(dataset_path, 'classes.txt')
    classes_dst = os.path.join(output_dir, 'classes.txt')
    if os.path.exists(classes_src):
        shutil.copy2(classes_src, classes_dst)
    
    print(f"\nСтруктура YOLO создана успешно!")
    print(f"Выходная директория: {output_dir}")
    print(f"Статистика:")
    print(f"  Train: {len(train_files)} изображений")
    print(f"  Val: {len(val_files)} изображений")
    print(f"  Test: {len(test_files)} изображений")
    print(f"  Всего: {total} изображений")

def create_yolo_yaml(output_dir):
    """
    Создает dataset.yaml файл для YOLO
    """
    classes_path = os.path.join(output_dir, 'classes.txt')
    class_names = []
    
    if os.path.exists(classes_path):
        with open(classes_path, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = ['crayfish']
        with open(classes_path, 'w', encoding='utf-8') as f:
            f.write("crayfish\n")
    
    yaml_content = f"""# YOLO dataset configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_names)}
names: {class_names}
"""
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"YAML файл создан: {yaml_path}")

def verify_dataset_structure(dataset_path):
    """
    Проверяет структуру датасета и выводит статистику
    """
    print("Проверка структуры датасета...")
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        images_dir = os.path.join(dataset_path, split, 'images')
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')] if os.path.exists(labels_dir) else []
            
            print(f"{split.upper():6}: {len(images):3} изображений, {len(labels):3} аннотаций")
            image_names = {os.path.splitext(f)[0] for f in images}
            label_names = {os.path.splitext(f)[0] for f in labels}
            
            missing_labels = image_names - label_names
            if missing_labels:
                print(f"  Предупреждение: отсутствуют аннотации для {len(missing_labels)} изображений")
            
            extra_labels = label_names - image_names
            if extra_labels:
                print(f"  Предупреждение: лишние аннотации для {len(extra_labels)} изображений")
        else:
            print(f"{split.upper():6}: папка не существует")

def quick_yolo_split():
    DATASET_PATH = "yolo_dataset"  # Папка где лежат images и labels
    OUTPUT_DIR = "yolo_structured" 
    
    print("Создание структуры YOLO...")
    print(f"Исходная папка: {DATASET_PATH}")
    print(f"Выходная папка: {OUTPUT_DIR}")
    
    create_yolo_structure(
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8, 
        val_ratio=0.2,   
        test_ratio=0.0  
    )
    print("\nПроверка результата:")
    verify_dataset_structure(OUTPUT_DIR)

if __name__ == "__main__":
    quick_yolo_split()

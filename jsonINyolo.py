import os
import json
import shutil
from PIL import Image
import random

def find_all_files(root_dir, extensions):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(root, filename))
    return files

def parse_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def extract_class_from_region(region_attrs):
    for key in ['1', 'object', 'name', 'class', 'label']:
        if key in region_attrs:
            return region_attrs[key]
    return 'unknown'

def convert_polygon_to_bbox(all_points_x, all_points_y, img_width, img_height):
    x_min = max(0, min(all_points_x))
    x_max = min(img_width, max(all_points_x))
    y_min = max(0, min(all_points_y))
    y_max = min(img_height, max(all_points_y))
    if x_min >= x_max or y_min >= y_max:
        return None
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
            0 <= width <= 1 and 0 <= height <= 1):
        return None
    
    return [x_center, y_center, width, height]

def process_single_json(json_path, images_base_dir, class_mapping):
    annotations = {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return annotations
    
    for image_key, image_data in data.items():
        filename = image_data['filename']
        image_path = find_image_file(images_base_dir, filename)
        if not image_path:
            print(f"Image not found: {filename}")
            continue
        img_size = parse_image_size(image_path)
        if not img_size:
            continue
            
        img_width, img_height = img_size
        
        yolo_annotations = []
        if 'regions' in image_data and image_data['regions']:
            for region_id, region in image_data['regions'].items():
                shape_attrs = region['shape_attributes']
                region_attrs = region['region_attributes']
                
                class_name = extract_class_from_region(region_attrs)
                class_id = class_mapping.get(class_name)
                
                if class_id is None:
                    print(f"Unknown class: {class_name} in {filename}")
                    continue
                
                if 'all_points_x' in shape_attrs and 'all_points_y' in shape_attrs:
                    bbox = convert_polygon_to_bbox(
                        shape_attrs['all_points_x'],
                        shape_attrs['all_points_y'],
                        img_width,
                        img_height
                    )
                    
                    if bbox:
                        yolo_annotations.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        
        annotations[filename] = {
            'image_path': image_path,
            'yolo_annotations': yolo_annotations,
            'image_size': (img_width, img_height)
        }
    
    return annotations

def find_image_file(images_base_dir, filename):
    for root, _, files in os.walk(images_base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def create_class_mapping_from_jsons(json_files, images_base_dir):
    all_classes = set()
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for image_data in data.values():
                if 'regions' in image_data:
                    for region in image_data['regions'].values():
                        class_name = extract_class_from_region(region['region_attributes'])
                        if class_name and class_name != 'unknown':
                            all_classes.add(class_name)
        except Exception as e:
            print(f"Error analyzing classes in {json_path}: {e}")
    
    sorted_classes = sorted(list(all_classes))
    return {cls: idx for idx, cls in enumerate(sorted_classes)}

def create_yolo_dataset(input_base_dir, output_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    print("Searching for JSON files and images...")
    json_files = find_all_files(input_base_dir, ['.json'])
    image_files = find_all_files(input_base_dir, ['.jpg', '.jpeg', '.png', '.bmp'])
    
    print(f"Found JSON files: {len(json_files)}")
    print(f"Found images: {len(image_files)}")
    
    if not json_files:
        print("No JSON files found!")
        return
    print("Creating class mapping...")
    class_mapping = create_class_mapping_from_jsons(json_files, input_base_dir)
    print(f"Found classes: {class_mapping}")
    
    all_annotations = {}
    for json_path in json_files:
        print(f"Processing {os.path.basename(json_path)}...")
        annotations = process_single_json(json_path, input_base_dir, class_mapping)
        all_annotations.update(annotations)
    
    print(f"Processed annotations for {len(all_annotations)} images")
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)
    
    image_filenames = list(all_annotations.keys())
    random.shuffle(image_filenames)
    
    total = len(image_filenames)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_files = image_filenames[:train_count]
    val_files = image_filenames[train_count:train_count + val_count]
    test_files = image_filenames[train_count + val_count:]
    
    def copy_to_split(filenames, split_dir):
        for filename in filenames:
            annotation_data = all_annotations[filename]
            
            src_image_path = annotation_data['image_path']
            dst_image_path = os.path.join(split_dir, 'images', filename)
            shutil.copy2(src_image_path, dst_image_path)
            
            base_name = os.path.splitext(filename)[0]
            dst_label_path = os.path.join(split_dir, 'labels', base_name + '.txt')
            
            with open(dst_label_path, 'w') as f:
                for ann in annotation_data['yolo_annotations']:
                    f.write(ann + '\n')
    
    print("Creating train set...")
    copy_to_split(train_files, train_dir)
    
    print("Creating val set...")
    copy_to_split(val_files, val_dir)
    
    if test_files:
        print("Creating test set...")
        copy_to_split(test_files, test_dir)
    
    classes_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_path, 'w') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")
    
    create_yolo_yaml(output_dir, class_mapping)
    
    print(f"SUCCESS: Dataset created successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Statistics:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    print(f"  Classes: {len(class_mapping)}")
    print(f"  Classes file: {classes_path}")

def create_yolo_yaml(output_dir, class_mapping):
    yaml_content = f"""# YOLO dataset configuration
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_mapping)}
names: {list(class_mapping.keys())}
"""
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"YAML config created: {yaml_path}")
def simple_create_dataset(input_dir, output_dir):
    all_data = {}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    all_data.update(data)
                    print(f"Loaded JSON: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    print(f"Found annotations: {len(all_data)}")
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0

    for image_key, image_data in all_data.items():
        filename = image_data['filename']
        image_path = None
        for root, dirs, files in os.walk(input_dir):
            if filename in files:
                image_path = os.path.join(root, filename)
                break
        
        if not image_path:
            print(f"Image not found: {filename}")
            error_count += 1
            continue
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening {filename}: {e}")
            error_count += 1
            continue
        yolo_lines = []
        regions = image_data.get('regions', {})
        
        if not regions:
            print(f"No annotations for {filename}")
            continue
            
        for region_id, region in regions.items():
            shape = region['shape_attributes']
            if 'all_points_x' in shape:
                x_points = shape['all_points_x']
                y_points = shape['all_points_y']
                
                x_min, x_max = min(x_points), max(x_points)
                y_min, y_max = min(y_points), max(y_points)
                if x_min >= x_max or y_min >= y_max:
                    continue
                
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                w = (x_max - x_min) / width
                h = (y_max - y_min) / height
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    continue
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        if yolo_lines:
            try:
                shutil.copy2(image_path, os.path.join(images_dir, filename))
                label_name = os.path.splitext(filename)[0] + '.txt'
                with open(os.path.join(labels_dir, label_name), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))
                
                processed_count += 1
                print(f"SUCCESS: Processed {filename}")
            except Exception as e:
                print(f"ERROR saving {filename}: {e}")
                error_count += 1
        else:
            print(f"WARNING: No valid annotations for {filename}")
            error_count += 1
    with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
        f.write("crayfish\n")
    
    print(f"DONE!")
    print(f"Created in: {output_dir}")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors: {error_count}")
if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\sergs\Desktop\rak"
    OUTPUT_DIR = "yolo_dataset"
    
    print("Starting YOLO dataset creation...")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    simple_create_dataset(INPUT_DIR, OUTPUT_DIR)
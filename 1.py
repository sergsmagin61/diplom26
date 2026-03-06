import os
from ultralytics import YOLO

def main():
    DATASET_PATH = "yolo_structured"
    MODEL_NAME = "yolo11m.pt"       
    EPOCHS = 100          
    IMGSZ = 640       
    
    print("Запуск обучения YOLO...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Ошибка: Датасет {DATASET_PATH} не найден")
        return
    
    config_content = f"""
path: {os.path.abspath(DATASET_PATH)}
train: train/images
val: val/images

names:
  0: crayfish
nc: 1
"""
    
    config_path = os.path.join(DATASET_PATH, "dataset.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    
    model = YOLO(MODEL_NAME)
    
    results = model.train(
        data=config_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=16,
        workers=4,
        project="training_results",
        name="train",
        patience=10,
        save=True
    )
    
    print("Обучение завершено!")

if __name__ == "__main__":
    main()
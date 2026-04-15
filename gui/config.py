# config.py - Конфигурация приложения

import os
from pathlib import Path

# Цветовая схема приложения
COLORS = {
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'accent': '#00ff88',
    'warning': '#ffcc00',
    'dark_bg': '#0f1a2b',
    'card_bg': '#1a2b3c',
    'text_primary': '#ffffff',
    'text_secondary': '#8899aa',
    'chart_1': '#00d4ff',
    'chart_2': '#ff6b6b',
    'chart_3': '#00ff88',
    'chart_4': '#ffcc00',
    'chart_5': '#9d4edd'
}

# Настройки калибровки по умолчанию
DEFAULT_CALIBRATION_FACTOR_X = 0.15
DEFAULT_CALIBRATION_FACTOR_Y = 0.15
DEFAULT_CALIBRATION_WIDTH_MM = 10.0
DEFAULT_CALIBRATION_HEIGHT_MM = 10.0

# Настройки детекции
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IMAGE_SIZE = (1600, 1000)

# Пути для поиска моделей
MODEL_SEARCH_PATHS = [
    "training_results/train/weights/",
    "training_results/train2/weights/",
    "models/",
    "optimized_models/"
]

# Поддерживаемые расширения моделей
MODEL_EXTENSIONS = ['*.pt', '*.onnx', '*.engine']

# Настройки графиков
CHART_FIGURE_SIZE = (16, 10)
CHART_DPI = 100

# Настройки Telegram бота
TELEGRAM_TEMP_DIR = "temp"
MAX_PHOTO_SIZE_MB = 10

# Настройки базы данных
DATABASE_NAME = "crayfish_data.db"

# Настройки окна калибровки
CALIBRATION_WINDOW_SIZE = (900, 700)
CALIBRATION_CANVAS_SIZE = (800, 600)
CALIBRATION_ZOOM_FACTOR = 2.0  # Увеличение при клике
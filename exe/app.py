
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from pathlib import Path
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from collections import defaultdict
import json

from database import DatabaseManager
from telegram_bot import TelegramBotManager
from models_manager import ModelOptimizer
from config import COLORS, DEFAULT_CALIBRATION_FACTOR_X, DEFAULT_CALIBRATION_FACTOR_Y
from config import DEFAULT_CONFIDENCE, MODEL_SEARCH_PATHS, MODEL_EXTENSIONS
from utils import setup_matplotlib_style, create_growth_plots, format_detection_results_text
from utils import create_database_stats_text, calculate_calibration_factors, detect_shape_from_points
from utils import calculate_distance

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime не установлен. ONNX модели не будут поддерживаться.")


class ModernCrayfishDetector:
    """Главный класс приложения для детекции раков"""
    
    def __init__(self, root):
        """Инициализация главного окна приложения"""
        self.root = root
        self.root.title("Crayfish AI Studio Pro")
        self.root.geometry("1600x1000")
        self.root.configure(bg=COLORS['dark_bg'])
        
        # Настройка стилей matplotlib
        setup_matplotlib_style()
        
        # Инициализация калибровочных коэффициентов
        self.calibration_factor_x = DEFAULT_CALIBRATION_FACTOR_X
        self.calibration_factor_y = DEFAULT_CALIBRATION_FACTOR_Y
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_known_width_mm = 10.0
        self.calibration_known_height_mm = 10.0
        self.calibration_shape = None
        
        # Инициализация модели и детекции
        self.model = None
        self.model_type = None
        self.models = {}
        self.current_model = ""
        self.confidence = DEFAULT_CONFIDENCE
        self.current_image = None
        self.processed_image = None
        self.detections = []
        self.current_image_path = None
        
        # Переменные для отображения изображений
        self.photo = None
        self.photo_original = None
        self.photo_result = None
        self.display_scale_factor = 1.0
        self.original_image_size = None
        self.canvas_image_id = None
        
        # Инициализация Telegram бота и БД
        self.telegram_bot = None
        self.bot_thread = None
        self.db = DatabaseManager()
        
        # Хранилище данных
        self.measurement_history = defaultdict(list)
        self.crayfish_counter = 1
        self.session_history = []
        
        # Загрузка существующих данных и настройка UI
        self.load_existing_data()
        self.setup_modern_ui()
        self.load_available_models()
        self.check_database_health()
    
    #ИНИЦИАЛИЗАЦИЯ БД И ДАННЫХ
    
    def check_database_health(self):
        """Проверка состояния базы данных и обновление статуса"""
        status = self.db.check_database_status()
        
        if status:
            message = f"БД: {status['crayfish_count']} раков, {status['measurements_count']} измерений"
            if hasattr(self, 'status_label'):
                self.status_label.config(text=message)
            print(f"Статус БД: {message}")
    
    def load_existing_data(self):
        """Загрузка существующих данных из базы данных"""
        try:
            crayfish_df = self.db.get_all_crayfish()
            
            if not crayfish_df.empty:
                if 'id' in crayfish_df.columns and not crayfish_df['id'].empty:
                    max_id = crayfish_df['id'].max()
                else:
                    max_id = 0
                self.crayfish_counter = max_id + 1
                
                print(f"Загружено {len(crayfish_df)} раков из БД")
                measurements_df = self.db.get_all_measurements()
                
                if not measurements_df.empty:
                    self.measurement_history.clear()
                    for _, row in measurements_df.iterrows():
                        if 'crayfish_id' in row and not pd.isna(row['crayfish_id']):
                            try:
                                crayfish_id = int(row['crayfish_id'])
                                measurement = {
                                    'timestamp': pd.to_datetime(row['timestamp']),
                                    'image_path': str(row['image_path']) if 'image_path' in row and not pd.isna(row['image_path']) else '',
                                    'width_mm': float(row['width_mm']) if 'width_mm' in row and not pd.isna(row['width_mm']) else 0.0,
                                    'height_mm': float(row['height_mm']) if 'height_mm' in row and not pd.isna(row['height_mm']) else 0.0,
                                    'angle': float(row['angle']) if 'angle' in row and not pd.isna(row['angle']) else 0.0,
                                    'confidence': float(row['confidence']) if 'confidence' in row and not pd.isna(row['confidence']) else 0.0,
                                    'width_px': float(row['width_px']) if 'width_px' in row and not pd.isna(row['width_px']) else 0.0,
                                    'height_px': float(row['height_px']) if 'height_px' in row and not pd.isna(row['height_px']) else 0.0
                                }
                                self.measurement_history[crayfish_id].append(measurement)
                            except (ValueError, TypeError) as e:
                                print(f"Ошибка загрузки измерения для рака {row.get('crayfish_id', 'unknown')}: {e}")
                    
                    print(f"Загружено {len(measurements_df)} измерений из БД")
            self.update_db_stats()
            
        except Exception as e:
            print(f"Ошибка загрузки данных из БД: {e}")
    
    #НАСТРОЙКА ПОЛЬЗОВАТЕЛЬСКОГО ИНТЕРФЕЙСа
    
    def setup_modern_ui(self):
        """Настройка современного пользовательского интерфейса"""
        self.setup_styles()
        main_container = tk.Frame(self.root, bg=COLORS['dark_bg'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        self.create_header(main_container)
        content_frame = tk.Frame(main_container, bg=COLORS['dark_bg'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Левая панель
        left_frame = tk.Frame(content_frame, bg=COLORS['dark_bg'], width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Центральная панель
        center_frame = tk.Frame(content_frame, bg=COLORS['dark_bg'])
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Правая панель
        right_frame = tk.Frame(content_frame, bg=COLORS['dark_bg'], width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        self.create_quick_actions(left_frame)
        self.create_image_viewer(center_frame)
        self.create_analytics_panel(right_frame)
    
    def setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Modern.TButton',
                       background=COLORS['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 11, 'bold'),
                       padding=(20, 12))
        
        style.map('Modern.TButton',
                 background=[('active', '#00b8e6'), ('pressed', '#0099cc')])
        
        style.configure('Card.TFrame', background=COLORS['card_bg'])
        
        style.configure('Title.TLabel', 
                       background=COLORS['dark_bg'],
                       foreground=COLORS['primary'],
                       font=('Arial', 16, 'bold'))
        
        style.configure('Vertical.TScrollbar',
                       background=COLORS['primary'],
                       troughcolor=COLORS['card_bg'],
                       arrowcolor='white',
                       bordercolor=COLORS['primary'],
                       lightcolor=COLORS['primary'],
                       darkcolor=COLORS['primary'])
    
    #ВЕРХНЯЯ ПАНЕЛЬ
    
    def create_header(self, parent):
        """Создание верхней панели с заголовком и индикаторами"""
        header_frame = tk.Frame(parent, bg=COLORS['dark_bg'], height=80)
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="CRAYFISH AI STUDIO PRO",
                              font=('Arial', 20, 'bold'),
                              fg=COLORS['primary'],
                              bg=COLORS['dark_bg'])
        title_label.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(header_frame,
                                   text="Система готова | БД: подключена",
                                   font=('Arial', 12),
                                   fg=COLORS['accent'],
                                   bg=COLORS['dark_bg'])
        self.status_label.pack(side=tk.RIGHT, padx=20)
        
        self.create_indicators(header_frame)
    
    def create_indicators(self, parent):
        """Создание индикаторов состояния системы"""
        indicators_frame = tk.Frame(parent, bg=COLORS['dark_bg'])
        indicators_frame.pack(side=tk.RIGHT, padx=20)
        
        self.db_indicator = tk.Label(indicators_frame, text="DB", 
                                   font=('Arial', 12),
                                   fg=COLORS['accent'], bg=COLORS['dark_bg'])
        self.db_indicator.pack(side=tk.LEFT, padx=5)
        
        self.model_indicator = tk.Label(indicators_frame, text="ML", 
                                      font=('Arial', 12),
                                      fg='#ff4444', bg=COLORS['dark_bg'])
        self.model_indicator.pack(side=tk.LEFT, padx=5)
        
        self.calibration_indicator = tk.Label(indicators_frame, text="CAL", 
                                            font=('Arial', 12),
                                            fg='#ffcc00', bg=COLORS['dark_bg'])
        self.calibration_indicator.pack(side=tk.LEFT, padx=5)
        
        self.bot_indicator = tk.Label(indicators_frame, text="BOT", 
                                     font=('Arial', 12),
                                     fg='#ff4444', bg=COLORS['dark_bg'])
        self.bot_indicator.pack(side=tk.LEFT, padx=5)
    
    #ЛЕВАЯ ПАНЕЛЬ (БЫСТРЫЕ ДЕЙСТВИЯ)
    
    def create_quick_actions(self, parent):
        """Создание левой панели с кнопками быстрых действий"""
        main_container = ttk.Frame(parent, style='Card.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(main_container, bg=COLORS['card_bg'], 
                          highlightthickness=1, 
                          highlightbackground=COLORS['primary'])
        scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL, 
                                 command=canvas.yview,
                                 style='Vertical.TScrollbar')
        self.scrollable_frame = ttk.Frame(canvas, style='Card.TFrame')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=280)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self._setup_scroll_binding(canvas)
        self._fill_quick_actions_content()
    
    def _setup_scroll_binding(self, canvas):
        """Настройка прокрутки для панели"""
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self.scrollable_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.scrollable_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
    
    def _fill_quick_actions_content(self):
        """Заполнение левой панели кнопками и элементами управления"""
        title = tk.Label(self.scrollable_frame, text="Быстрые действия", 
                        font=('Arial', 14, 'bold'),
                        fg=COLORS['text_primary'],
                        bg=COLORS['card_bg'])
        title.pack(pady=15)
        
        actions = [
            ("Загрузить фото", self.load_image),
            ("Выбрать модель", self.load_model),
            ("Запустить детекцию", self.start_detection),
            ("Показать результат", self.show_detection_result),
            ("Анализ роста по дням", self.show_growth_analysis),
            ("Статистика БД", self.show_database_stats),
            ("Экспорт данных", self.export_all_data),
            ("Выборочный экспорт", self.export_by_date),
            ("Калибровка КВАДРАТОМ", self.start_square_calibration),
            ("Очистить БД", self.clear_database_ui),
            ("Telegram Bot", self.setup_telegram_bot)
        ]
        
        for text, command in actions:
            btn = ttk.Button(self.scrollable_frame, text=text, command=command, style='Modern.TButton')
            btn.pack(fill=tk.X, padx=20, pady=5)
        
        self.create_confidence_slider(self.scrollable_frame)
        self.create_calibration_indicator(self.scrollable_frame)
        self.create_telegram_bot_ui(self.scrollable_frame)
        self.create_realtime_stats(self.scrollable_frame)
        
        spacer = tk.Frame(self.scrollable_frame, height=20, bg=COLORS['card_bg'])
        spacer.pack(fill=tk.X)
    
    # ЭЛЕМЕНТЫ УПРАВЛЕНИЯ
    
    def create_confidence_slider(self, parent):
        """Создание ползунка для настройки порога уверенности"""
        slider_frame = tk.Frame(parent, bg=COLORS['card_bg'])
        slider_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(slider_frame, text="Порог уверенности", 
                font=('Arial', 12, 'bold'),
                fg=COLORS['text_primary'],
                bg=COLORS['card_bg']).pack(anchor=tk.W)
        
        self.confidence_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        slider = ttk.Scale(slider_frame, from_=0.1, to=0.9, 
                          variable=self.confidence_var, orient=tk.HORIZONTAL)
        slider.pack(fill=tk.X, pady=10)
        slider.configure(command=self.on_slider_change)
        
        self.confidence_label = tk.Label(slider_frame, 
                                       text=f"Текущее значение: {DEFAULT_CONFIDENCE:.2f}",
                                       font=('Arial', 10),
                                       fg=COLORS['text_secondary'],
                                       bg=COLORS['card_bg'])
        self.confidence_label.pack()
    
    def on_slider_change(self, value):
        """Обработчик изменения значения ползунка уверенности"""
        self.confidence = float(value)
        self.confidence_label.config(text=f"Текущее значение: {self.confidence:.2f}")
    
    def create_calibration_indicator(self, parent):
        """индикаторы калибровки"""
        calibration_frame = tk.Frame(parent, bg=COLORS['card_bg'])
        calibration_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(calibration_frame, text="Калибровочные коэффициенты", 
                font=('Arial', 11, 'bold'),
                fg=COLORS['text_primary'],
                bg=COLORS['card_bg']).pack(anchor=tk.W)
        
        self.calibration_label_x = tk.Label(calibration_frame,
                                         text=f"X: 1 пиксель = {self.calibration_factor_x:.4f} мм",
                                         font=('Arial', 10),
                                         fg=COLORS['accent'],
                                         bg=COLORS['card_bg'])
        self.calibration_label_x.pack(pady=2)
        
        self.calibration_label_y = tk.Label(calibration_frame,
                                         text=f"Y: 1 пиксель = {self.calibration_factor_y:.4f} мм",
                                         font=('Arial', 10),
                                         fg=COLORS['accent'],
                                         bg=COLORS['card_bg'])
        self.calibration_label_y.pack(pady=2)
        
        reset_btn = tk.Button(calibration_frame, text="Сбросить калибровку",
                             font=('Arial', 9),
                             bg=COLORS['secondary'], fg='white',
                             command=self.reset_calibration)
        reset_btn.pack(pady=5)
    
    def reset_calibration(self):
        """Сброс калибровки к сток значениям"""
        self.calibration_factor_x = DEFAULT_CALIBRATION_FACTOR_X
        self.calibration_factor_y = DEFAULT_CALIBRATION_FACTOR_Y
        self.calibration_label_x.config(text=f"X: 1 пиксель = {self.calibration_factor_x:.4f} мм")
        self.calibration_label_y.config(text=f"Y: 1 пиксель = {self.calibration_factor_y:.4f} мм")
        messagebox.showinfo("Информация", f"Калибровка сброшена к стандартным значениям\n"
                                        f"По ширине: 1px = {self.calibration_factor_x:.4f} мм\n"
                                        f"По высоте: 1px = {self.calibration_factor_y:.4f} мм")
    
    def create_telegram_bot_ui(self, parent):
        """Создание интерфейса для управления тг ботом"""
        telegram_frame = tk.Frame(parent, bg=COLORS['card_bg'])
        telegram_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(telegram_frame, text="Telegram Bot API", 
                font=('Arial', 11, 'bold'),
                fg=COLORS['text_primary'],
                bg=COLORS['card_bg']).pack(anchor=tk.W)
        
        tk.Label(telegram_frame, text="Токен бота:", 
                font=('Arial', 9),
                fg=COLORS['text_secondary'],
                bg=COLORS['card_bg']).pack(anchor=tk.W, pady=(5,0))
        
        self.bot_token_var = tk.StringVar()
        token_entry = tk.Entry(telegram_frame, textvariable=self.bot_token_var, 
                              font=('Arial', 10), width=30)
        token_entry.pack(fill=tk.X, pady=5)
        
        btn_frame = tk.Frame(telegram_frame, bg=COLORS['card_bg'])
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="Запустить бота", 
                 font=('Arial', 9),
                 bg=COLORS['accent'], fg='white',
                 command=self.start_telegram_bot).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_frame, text="Остановить", 
                 font=('Arial', 9),
                 bg=COLORS['secondary'], fg='white',
                 command=self.stop_telegram_bot).pack(side=tk.LEFT, padx=2)
        
        self.bot_status_label = tk.Label(telegram_frame, text="Бот не запущен", 
                                       font=('Arial', 9),
                                       fg=COLORS['secondary'],
                                       bg=COLORS['card_bg'])
        self.bot_status_label.pack(pady=5)
    
    def create_realtime_stats(self, parent):
        """Создание панели статистики базы данных"""
        stats_frame = tk.Frame(parent, bg=COLORS['card_bg'])
        stats_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(stats_frame, text="Статистика БД", 
                font=('Arial', 12, 'bold'),
                fg=COLORS['text_primary'],
                bg=COLORS['card_bg']).pack(anchor=tk.W)
        
        self.stats_labels = {}
        stats_data = [
            ("Всего особей", "total_crayfish"),
            ("Всего измерений", "total_measurements"),
            ("ML запросов", "ml_requests"),
            ("Дней с данными", "days_with_data"),
            ("Средний размер", "avg_size"),
            ("Мин. размер", "min_size"),
            ("Макс. размер", "max_size")
        ]
        
        for text, key in stats_data:
            frame = tk.Frame(stats_frame, bg=COLORS['card_bg'])
            frame.pack(fill=tk.X, pady=5)
            tk.Label(frame, text=text, font=('Arial', 9),
                   fg=COLORS['text_secondary'], bg=COLORS['card_bg']).pack(side=tk.LEFT)
            value_label = tk.Label(frame, text="0", font=('Arial', 10, 'bold'),
                                fg=COLORS['accent'], bg=COLORS['card_bg'])
            value_label.pack(side=tk.RIGHT)
            self.stats_labels[key] = value_label
        
        self.update_db_stats()
    
    def update_db_stats(self):
        """Обновление статистики базы данных на панели"""
        try:
            crayfish_df = self.db.get_all_crayfish()
            measurements_df = self.db.get_all_measurements()
            ml_logs_df = self.db.get_ml_inference_stats()
            
            total_crayfish = len(crayfish_df) if not crayfish_df.empty else 0
            total_measurements = len(measurements_df) if not measurements_df.empty else 0
            
            ml_requests = 0
            if not ml_logs_df.empty and 'total_inferences' in ml_logs_df.columns:
                ml_requests = ml_logs_df['total_inferences'].sum()
            
            days_with_data = 0
            if not measurements_df.empty and 'timestamp' in measurements_df.columns:
                try:
                    measurements_df['date'] = pd.to_datetime(measurements_df['timestamp']).dt.date
                    days_with_data = measurements_df['date'].nunique()
                except:
                    days_with_data = 0
            
            # Статистика размеров
            avg_size = min_size = max_size = 0
            if not measurements_df.empty and 'width_mm' in measurements_df.columns:
                try:
                    sizes = pd.to_numeric(measurements_df['width_mm'], errors='coerce')
                    sizes = sizes.dropna()
                    if len(sizes) > 0:
                        avg_size = sizes.mean()
                        min_size = sizes.min()
                        max_size = sizes.max()
                except Exception as e:
                    print(f"Ошибка расчета статистики размеров: {e}")
            
            if hasattr(self, 'stats_labels'):
                self.stats_labels['total_crayfish'].config(text=str(total_crayfish))
                self.stats_labels['total_measurements'].config(text=str(total_measurements))
                self.stats_labels['ml_requests'].config(text=str(int(ml_requests)))
                self.stats_labels['days_with_data'].config(text=str(days_with_data))
                self.stats_labels['avg_size'].config(text=f"{avg_size:.1f} мм")
                self.stats_labels['min_size'].config(text=f"{min_size:.1f} мм")
                self.stats_labels['max_size'].config(text=f"{max_size:.1f} мм")
            
        except Exception as e:
            print(f"Ошибка обновления статистики БД: {e}")
    
    # ЦЕНТРАЛЬНАЯ ПАНЕЛь
    
    def create_image_viewer(self, parent):
        """Создание центральной панели для просмотра изображений"""
        viewer_frame = ttk.Frame(parent, style='Card.TFrame')
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(viewer_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        original_tab = ttk.Frame(self.notebook)
        self.notebook.add(original_tab, text="Исходное")
        
        result_tab = ttk.Frame(self.notebook)
        self.notebook.add(result_tab, text="Результат")
        
        self.original_canvas = tk.Canvas(original_tab, bg=COLORS['card_bg'], 
                                       highlightthickness=1, highlightbackground=COLORS['primary'])
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_canvas = tk.Canvas(result_tab, bg=COLORS['card_bg'],
                                     highlightthickness=1, highlightbackground=COLORS['primary'])
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.original_canvas.bind("<Configure>", self.on_canvas_resize)
        self.result_canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.original_label = tk.Label(original_tab, text="Загрузите изображение...",
                                     fg=COLORS['text_secondary'], bg=COLORS['card_bg'])
        self.original_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.result_label = tk.Label(result_tab, text="Результат появится здесь...",
                                   fg=COLORS['text_secondary'], bg=COLORS['card_bg'])
        self.result_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def on_canvas_resize(self, event):
        """Обработчик изменения размера canvas для перерисовки изображений"""
        if event.widget == self.original_canvas and hasattr(self, 'photo_original') and self.photo_original:
            self.redraw_original_image()
        elif event.widget == self.result_canvas and hasattr(self, 'photo_result') and self.photo_result:
            self.redraw_result_image()
    
    def redraw_original_image(self):
        """Перерисовка исходного изображения"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            return
        try:
            image = Image.open(self.current_image_path)
            self._display_image_on_canvas(image, self.original_canvas, 'original')
        except Exception as e:
            print(f"Ошибка перерисовки исходного изображения: {e}")
    
    def redraw_result_image(self):
        """Перерисовка изображения результата"""
        if self.processed_image is None:
            return
        try:
            if isinstance(self.processed_image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
                self._display_image_on_canvas(image, self.result_canvas, 'result')
        except Exception as e:
            print(f"Ошибка перерисовки результата: {e}")
    
    def _display_image_on_canvas(self, image, canvas, img_type):
        """Отображение изображения на canvas с масштабированием"""
        try:
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 10 or canvas_height <= 10:
                canvas_width = 780
                canvas_height = 580
            
            padding = 10
            canvas_width -= padding * 2
            canvas_height -= padding * 2
            
            img_width, img_height = image.size
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            scale = min(width_ratio, height_ratio)
            
            new_width = max(1, int(img_width * scale))
            new_height = max(1, int(img_height * scale))
            
            if img_type == 'original':
                self.display_scale_factor = scale
            
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(resized_image)
            
            if img_type == 'original':
                self.photo_original = photo
            else:
                self.photo_result = photo
            
            canvas.delete("all")
            x = padding + (canvas_width - new_width) // 2
            y = padding + (canvas_height - new_height) // 2
            self.canvas_image_id = canvas.create_image(x, y, anchor=tk.NW, image=photo)
            
            if img_type == 'original':
                self.original_label.place_forget()
            else:
                self.result_label.place_forget()
            
        except Exception as e:
            print(f"Ошибка отображения изображения: {e}")
            if img_type == 'original':
                self.original_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            else:
                self.result_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    def display_image_original(self, image_path):
        """Отображение исходного изображения"""
        try:
            image = Image.open(image_path)
            self.original_image_size = image.size
            self._display_image_on_canvas(image, self.original_canvas, 'original')
            self.status_label.config(text="Изображение загружено")
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
    
    def display_image_result(self, image):
        """Отображение обработанного изображения с результатами"""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self._display_image_on_canvas(image, self.result_canvas, 'result')
        except Exception as e:
            print(f"Ошибка отображения обработанного изображения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось отобразить результат: {e}")
    
    # ПРАВАЯ ПАНЕЛЬ
    
    def create_analytics_panel(self, parent):
        """Создание правой панели с детальной аналитикой"""
        analytics_frame = ttk.Frame(parent, style='Card.TFrame')
        analytics_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        tk.Label(analytics_frame, text="Детальная аналитика", 
                font=('Arial', 14, 'bold'),
                fg=COLORS['text_primary'],
                bg=COLORS['card_bg']).pack(pady=15)
        
        details_frame = tk.Frame(analytics_frame, bg=COLORS['card_bg'])
        details_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.detections_text = tk.Text(details_frame, height=20, width=35,
                                     bg='#1e2b3a', fg=COLORS['text_primary'],
                                     font=('Arial', 9), relief='flat', wrap=tk.WORD)
        
        scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.detections_text.yview)
        self.detections_text.configure(yscrollcommand=scrollbar.set)
        
        self.detections_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        initial_text = """РЕЗУЛЬТАТЫ ДЕТЕКЦИИ

Загрузите изображение и запустите детекцию для получения результатов.

Будут показаны:
• Количество обнаруженных раков
• Уровень уверенности для каждого
• Размеры в мм и пикселях
• ID особи (из БД)

ПОДДЕРЖИВАЕМЫЕ ФОРМАТЫ МОДЕЛЕЙ:
• .pt - нативный формат YOLO
• .onnx - оптимизированный кросс-платформенный
• .engine - максимальная скорость (TensorRT)
"""
        
        self.detections_text.insert(1.0, initial_text)
        self.detections_text.config(state=tk.DISABLED)
    
    # УПРАВЛЕНИЕ МОДЕЛЯМИ
    
    def load_available_models(self):
        """Загрузка доступных моделей из директорий"""
        found_models = []
        
        for ext in MODEL_EXTENSIONS:
            for path in Path('.').glob(ext):
                if path.exists():
                    model_name = path.name
                    self.models[model_name] = str(path)
                    found_models.append(model_name)
                    print(f"Найдена модель: {model_name}")
        
        for search_path in MODEL_SEARCH_PATHS:
            for ext in MODEL_EXTENSIONS:
                for path in Path(search_path).glob(ext):
                    if path.exists():
                        model_name = f"[{search_path}]{path.name}"
                        self.models[model_name] = str(path)
                        found_models.append(model_name)
                        print(f"Найдена модель: {model_name}")
        
        if self.models:
            self.current_model = list(self.models.keys())[0]
            self.load_model_thread(self.models[self.current_model])
            self.model_indicator.config(fg=COLORS['accent'])
        else:
            self.status_label.config(text="Модели не найдены")
            print("Модели не найдены. Поместите .pt, .onnx или .engine файлы в папку программы")
    
    def load_model_thread(self, model_path):
        """Загрузка модели в отдельном потоке"""
        def load():
            try:
                ext = Path(model_path).suffix.lower()
                
                if ext == '.pt':
                    self.model = YOLO(model_path, task='obb')
                    self.model_type = 'yolo'
                    print(f"Загружена YOLO OBB модель: {model_path}")
                    
                elif ext == '.onnx':
                    if ONNX_AVAILABLE:
                        self.model = YOLO(model_path, task='obb')
                        self.model_type = 'yolo'
                        print(f"Загружена ONNX OBB модель: {model_path}")
                    else:
                        raise ImportError("ONNX Runtime не установлен")
                        
                elif ext == '.engine':
                    self.model = YOLO(model_path, task='obb')
                    self.model_type = 'yolo'
                    print(f"Загружена TensorRT OBB модель: {model_path}")
                    
                else:
                    raise ValueError(f"Неподдерживаемый формат: {ext}")
                
                self.current_model = os.path.basename(model_path)
                self.root.after(0, lambda: self.status_label.config(text=f"Модель загружена: {self.current_model} (OBB)"))
                self.root.after(0, self.update_db_stats)
                
                if ext == '.engine':
                    self.root.after(0, lambda: self.model_indicator.config(fg='#00ff88', text="TRT-OBB"))
                elif ext == '.onnx':
                    self.root.after(0, lambda: self.model_indicator.config(fg='#00ff88', text="ONNX-OBB"))
                else:
                    self.root.after(0, lambda: self.model_indicator.config(fg=COLORS['accent'], text="OBB"))
                
            except Exception as e:
                error_msg = str(e)
                print(f"Ошибка загрузки модели: {error_msg}")
                self.root.after(0, lambda: self.status_label.config(text=f"Ошибка загрузки: {error_msg[:50]}"))
                self.root.after(0, lambda: self.model_indicator.config(fg='#ff4444'))
        
        threading.Thread(target=load, daemon=True).start()
    
    def load_model(self):
        """Выбор и загрузка модели через диалоговое окно"""
        file_path = filedialog.askopenfilename(
            title="Выберите модель YOLO OBB",
            filetypes=[
                ("Все модели", "*.pt;*.onnx;*.engine"),
                ("YOLO models", "*.pt"),
                ("ONNX models", "*.onnx"),
                ("TensorRT models", "*.engine"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            threading.Thread(target=self.load_model_thread, args=(file_path,), daemon=True).start()
    
    # ЗАГРУЗКА ИЗОБРАЖЕНИЙ
    
    def load_image(self):
        """Загрузка изображения через диалоговое окно"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image_original(file_path)
    
    # ДЕТЕКЦИЯ И ОБРАБОТКА РЕЗУЛЬТАТОВ 
    
    def start_detection(self):
        """Запуск процесса детекции на загруженном изображении"""
        if not self.model:
            messagebox.showwarning("Внимание", "Сначала загрузите модель!")
            return
        
        if not self.current_image_path:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
            return
        
        self.status_label.config(text="Выполняется детекция...")
        threading.Thread(target=self.run_detection, daemon=True).start()
    
    def run_detection(self):
        """Выполнение детекции на загруженном изображении"""
        try:
            start_time = time.perf_counter()
            results = self.model.predict(
                source=self.current_image_path,
                conf=self.confidence,
                save=False
            )
            
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            processing_time = end_time - start_time
            
            for result in results:
                img = result.orig_img.copy()
                
                # Обработка OBB детекций
                if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                    print(f"Найдены OBB детекции: {len(result.obb)}")
                    self.detections = result.obb
                    img = self._draw_obb_detections(img, result.obb)
                    self.processed_image = img
                    
                # Обработка обычных детекций
                elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    print(f"Найдены обычные детекции: {len(result.boxes)}")
                    self.detections = result.boxes
                    img = self._draw_box_detections(img, result.boxes)
                    self.processed_image = img
                else:
                    print("Детекции не найдены")
                    self.processed_image = img
                    self.detections = []
                
                # Сохранение сессии
                session_data = {
                    'timestamp': datetime.now(),
                    'image_path': self.current_image_path,
                    'total_detections': len(self.detections) if self.detections else 0,
                    'processing_time': processing_time,
                    'confidence_threshold': self.confidence
                }
                
                session_id = self.db.save_session(session_data)
                
                if session_id and self.detections and len(self.detections) > 0:
                    self._save_inference_log(inference_time_ms, img.shape)
                    self.save_detections_to_db(session_id)
                    
                    self.root.after(0, lambda: self.update_ui_after_detection(
                        processing_time, session_id, inference_time_ms))
                else:
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Информация", "Объекты не обнаружены на изображении"))
                
        except Exception as e:
            error_msg = str(e)
            print(f"Ошибка детекции: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda msg=error_msg: messagebox.showerror(
                "Ошибка", f"Ошибка детекции: {msg[:100]}"))
    
    def _draw_obb_detections(self, img, obb_detections):
        """Отрисовка OBB детекций на изображении"""
        for i, obb in enumerate(obb_detections):
            try:
                if hasattr(obb, 'conf'):
                    conf = obb.conf[0].item() if hasattr(obb.conf, '__len__') else obb.conf
                else:
                    conf = 0.5
                
                if hasattr(obb, 'xywhr') and obb.xywhr is not None:
                    xywhr = obb.xywhr[0].cpu().numpy()
                    cx, cy, w, h, angle = xywhr
                    
                    rect = ((int(cx), int(cy)), (int(w), int(h)), angle * 180 / np.pi)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 3)
                    
                    text = f"Rak #{i+1} ({conf:.2f})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    thickness = 3
                    color = (0, 255, 0)
                    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = int(cx - text_w // 2)
                    text_y = int(cy - h//2 - 20)
                    
                    cv2.rectangle(img, 
                                (text_x - 10, text_y - text_h - 10),
                                (text_x + text_w + 10, text_y + 10),
                                (0, 0, 0), -1)
                    
                    cv2.putText(img, text, (text_x, text_y),
                              font, font_scale, color, thickness)
                    
                    if abs(angle) > 0.01:
                        angle_text = f"{angle:.1f}°"
                        cv2.putText(img, angle_text, 
                                  (int(cx + w//2 + 15), int(cy - h//2)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                
            except Exception as e:
                print(f"Ошибка отрисовки OBB {i}: {e}")
                continue
        
        return img
    
    def _draw_box_detections(self, img, box_detections):
        """Отрисовка обычных box детекций на изображении"""
        for i, box in enumerate(box_detections):
            try:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                conf = box.conf[0].item() if hasattr(box.conf, '__len__') else box.conf
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                text = f"Rak #{i+1} ({conf:.2f})"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                color = (0, 255, 0)
                
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x1
                text_y = y1 - 20
                
                cv2.rectangle(img,
                            (text_x - 10, text_y - text_h - 10),
                            (text_x + text_w + 10, text_y + 10),
                            (0, 0, 0), -1)
                
                cv2.putText(img, text, (text_x, text_y),
                          font, font_scale, color, thickness)
                
                width_px = x2 - x1
                height_px = y2 - y1
                size_text = f"{width_px}x{height_px}px"
                cv2.putText(img, size_text, (x1, y2 + 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
            except Exception as e:
                print(f"Ошибка отрисовки box {i}: {e}")
                continue
        
        return img
    
    def _save_inference_log(self, inference_time_ms, image_shape):
        """Сохранение лога ML инференса в БД"""
        inference_data = {
            'timestamp': datetime.now(),
            'image_path': self.current_image_path,
            'image_width': image_shape[1],
            'image_height': image_shape[0],
            'inference_time_ms': inference_time_ms,
            'detections_count': len(self.detections),
            'avg_confidence': float(np.mean([det.conf.cpu().numpy() for det in self.detections])),
            'model_name': self.current_model,
            'model_type': self.model_type,
            'confidence_threshold': self.confidence
        }
        self.db.save_ml_inference_log(inference_data)
    
    def save_detections_to_db(self, session_id):
        """Сохранение детекций в базу данных"""
        if not self.detections:
            return
        
        print(f"Сохранение {len(self.detections)} обнаружений в БД...")
        
        for i, det in enumerate(self.detections):
            try:
                # Извлечение уверенности
                if hasattr(det, 'conf'):
                    if hasattr(det.conf, '__len__') and len(det.conf) > 0:
                        confidence = float(det.conf[0].item())
                    else:
                        confidence = float(det.conf)
                else:
                    confidence = 0.5
                
                # Извлечение координат
                if hasattr(det, 'xywhr') and det.xywhr is not None:
                    coords = det.xywhr[0].cpu().numpy()
                    cx, cy, w, h, angle = coords
                    width_px = float(w)
                    height_px = float(h)
                    angle = float(angle)
                    bbox_coords = [float(cx), float(cy), float(w), float(h), float(angle)]
                elif hasattr(det, 'xyxy') and det.xyxy is not None:
                    coords = det.xyxy[0].cpu().numpy()
                    width_px = float(coords[2] - coords[0])
                    height_px = float(coords[3] - coords[1])
                    angle = 0.0
                    bbox_coords = coords.tolist()
                else:
                    print(f"Неизвестный формат детекции для элемента {i}: {type(det)}")
                    continue
                
                # Расчет размеров в мм
                estimated_width_mm = float(width_px * self.calibration_factor_x)
                estimated_height_mm = float(height_px * self.calibration_factor_y)
                
                # Получение или создание ID рака
                crayfish_id = self.get_or_create_crayfish(estimated_width_mm)
                
                if crayfish_id:
                    measurement_data = {
                        'timestamp': datetime.now(),
                        'image_path': self.current_image_path,
                        'width_mm': estimated_width_mm,
                        'height_mm': estimated_height_mm,
                        'width_px': width_px,
                        'height_px': height_px,
                        'angle': angle,
                        'confidence': confidence,
                        'bounding_box': json.dumps(bbox_coords)
                    }
                    measurement_id = self.db.save_measurement(crayfish_id, measurement_data, session_id)
                    
                    if measurement_id:
                        measurement_data['crayfish_id'] = crayfish_id
                        measurement_data['session_id'] = session_id
                        self.measurement_history[crayfish_id].append(measurement_data)
                        print(f"Сохранено измерение {i+1}: рак {crayfish_id}, размер {estimated_width_mm:.1f}мм, угол {angle:.1f}°")
                
            except Exception as e:
                print(f"Ошибка сохранения обнаружения {i+1}: {e}")
                import traceback
                traceback.print_exc()
    
    def get_or_create_crayfish(self, current_size):
        """Получение существующего ID рака или создание нового"""
        crayfish_df = self.db.get_all_crayfish()
        
        if crayfish_df.empty:
            unique_id = f"crayfish_{self.crayfish_counter}"
            crayfish_id = self.db.save_crayfish(unique_id, float(current_size))
            self.crayfish_counter += 1
            return crayfish_id
        
        best_match_id = None
        min_size_diff = float('inf')
        
        for _, row in crayfish_df.iterrows():
            crayfish_id = int(row['id'])
            history = self.measurement_history.get(crayfish_id, [])
            
            if history:
                last_measurement = history[-1]
                last_size = last_measurement['width_mm']
                size_diff = abs(current_size - last_size)
                if size_diff < last_size * 0.2 and size_diff < min_size_diff:
                    min_size_diff = size_diff
                    best_match_id = crayfish_id
        
        if best_match_id is not None:
            return best_match_id
        else:
            unique_id = f"crayfish_{self.crayfish_counter}"
            crayfish_id = self.db.save_crayfish(unique_id, float(current_size))
            self.crayfish_counter += 1
            return crayfish_id
    
    def update_ui_after_detection(self, processing_time, session_id, inference_time_ms):
        """Обновление UI после завершения детекции"""
        try:
            self.display_image_result(self.processed_image)
            self.update_detections_info(processing_time, session_id, inference_time_ms)
            self.update_db_stats()
            self.check_database_health()
            
            model_type_str = ""
            if self.current_model:
                ext = Path(self.current_model).suffix.lower()
                if ext == '.engine':
                    model_type_str = " [TensorRT]"
                elif ext == '.onnx':
                    model_type_str = " [ONNX]"
            
            self.status_label.config(text=f"Детекция завершена ({processing_time:.2f}с, ML: {inference_time_ms:.1f}мс){model_type_str}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обновления UI: {e}")
    
    def update_detections_info(self, processing_time, session_id, inference_time_ms):
        """Обновление текстовой информации о детекции"""
        self.detections_text.config(state=tk.NORMAL)
        self.detections_text.delete(1.0, tk.END)
        
        text = format_detection_results_text(
            self.detections, self.confidence, self.calibration_factor_x,
            self.calibration_factor_y, self.current_image_path, session_id,
            processing_time, inference_time_ms, self.current_model, self.db
        )
        
        self.detections_text.insert(1.0, text)
        self.detections_text.config(state=tk.DISABLED)
    
    def show_detection_result(self):
        """Отображение результата детекции в отдельном окне"""
        if self.processed_image is None:
            messagebox.showinfo("Информация", "Сначала выполните детекцию!")
            return
        
        result_window = tk.Toplevel(self.root)
        result_window.title("Результат детекции")
        result_window.geometry("800x600")
        
        try:
            image = Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(result_window, image=photo)
            label.image = photo
            label.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось отобразить результат: {e}")
    
    # ==================== АНАЛИЗ РОСТА ====================
    
    def show_growth_analysis(self):
        """Отображение анализа роста популяции"""
        self.create_simple_growth_analysis()
    
    def create_simple_growth_analysis(self):
        """Создание окна с графиками анализа роста"""
        try:
            daily_stats = self.db.get_daily_statistics()
            
            if daily_stats.empty:
                messagebox.showwarning("Внимание", "Нет данных в базе данных!\nСначала выполните детекцию на нескольких изображениях.")
                return
            
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Анализ роста раков по дням")
            analysis_window.geometry("1400x900")
            analysis_window.configure(bg=COLORS['dark_bg'])
            
            title_label = tk.Label(analysis_window, 
                                  text="АНАЛИЗ РОСТА РАКОВ ПО ДНЯМ",
                                  font=('Arial', 18, 'bold'),
                                  fg=COLORS['primary'],
                                  bg=COLORS['dark_bg'])
            title_label.pack(pady=20)
            
            info_text = f"Анализ основан на данных за {len(daily_stats)} дней\n"
            info_text += f"Период: {daily_stats['date'].min().strftime('%d.%m.%Y')} - {daily_stats['date'].max().strftime('%d.%m.%Y')}"
            
            info_label = tk.Label(analysis_window, text=info_text,
                                 font=('Arial', 12),
                                 fg=COLORS['text_primary'],
                                 bg=COLORS['dark_bg'])
            info_label.pack(pady=10)
            
            fig = create_growth_plots(daily_stats, self.db, COLORS)
            
            canvas = FigureCanvasTkAgg(fig, analysis_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            toolbar = NavigationToolbar2Tk(canvas, analysis_window)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            export_btn = tk.Button(analysis_window, text="Экспортировать график", 
                                  font=('Arial', 11, 'bold'),
                                  bg=COLORS['primary'], fg='white',
                                  command=lambda: self.export_plot_as_image(fig))
            export_btn.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания анализа: {e}")
    
    def export_plot_as_image(self, fig):
        """Экспорт графика в файл изображения"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Экспортировать график как изображение",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='#0f1a2b')
                messagebox.showinfo("Успех", f"График экспортирован в:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    # ==================== КАЛИБРОВКА ====================
    
    def start_square_calibration(self):
        """Запуск калибровки квадратом/прямоугольником"""
        if not self.current_image_path:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
            return
        
        self.calibration_window = tk.Toplevel(self.root)
        self.calibration_window.title("Калибровка квадратом/прямоугольником")
        self.calibration_window.geometry("700x700")
        self.calibration_window.configure(bg=COLORS['dark_bg'])
        
        tk.Label(self.calibration_window, 
                text="КАЛИБРОВКА КВАДРАТОМ/ПРЯМОУГОЛЬНИКОМ",
                font=('Arial', 16, 'bold'),
                fg=COLORS['primary'],
                bg=COLORS['dark_bg']).pack(pady=20)
        
        instructions = tk.Label(self.calibration_window,
                              text="1. Выберите объект известного размера на фото\n"
                                   "2. Кликните мышкой по 4 углам объекта (по часовой стрелке)\n"
                                   "3. Введите реальные размеры объекта в мм\n"
                                   "4. Программа рассчитает раздельные коэффициенты X и Y",
                              font=('Arial', 11),
                              fg=COLORS['text_primary'],
                              bg=COLORS['dark_bg'],
                              justify=tk.LEFT)
        instructions.pack(pady=10)
        
        canvas_frame = tk.Frame(self.calibration_window, bg=COLORS['dark_bg'])
        canvas_frame.pack(pady=10)
        
        self.calibration_canvas = tk.Canvas(canvas_frame, 
                                          bg=COLORS['card_bg'],
                                          highlightthickness=1,
                                          highlightbackground=COLORS['primary'],
                                          width=600, height=400)
        self.calibration_canvas.pack()
        
        size_frame = tk.Frame(self.calibration_window, bg=COLORS['dark_bg'])
        size_frame.pack(pady=15)
        
        tk.Label(size_frame, text="Ширина объекта (мм):", 
                font=('Arial', 11),
                fg=COLORS['text_primary'],
                bg=COLORS['dark_bg']).pack(side=tk.LEFT, padx=10)
        
        self.calibration_width_var = tk.StringVar(value="10.0")
        width_entry = tk.Entry(size_frame, textvariable=self.calibration_width_var,
                            font=('Arial', 11), width=10)
        width_entry.pack(side=tk.LEFT, padx=10)
        
        tk.Label(size_frame, text="Высота объекта (мм):", 
                font=('Arial', 11),
                fg=COLORS['text_primary'],
                bg=COLORS['dark_bg']).pack(side=tk.LEFT, padx=10)
        
        self.calibration_height_var = tk.StringVar(value="10.0")
        height_entry = tk.Entry(size_frame, textvariable=self.calibration_height_var,
                              font=('Arial', 11), width=10)
        height_entry.pack(side=tk.LEFT, padx=10)
        
        info_frame = tk.Frame(self.calibration_window, bg=COLORS['card_bg'])
        info_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.calibration_info_label = tk.Label(info_frame,
                                             text="Жду выбора 4 точек... (0/4)",
                                             font=('Arial', 10),
                                             fg=COLORS['accent'],
                                             bg=COLORS['card_bg'])
        self.calibration_info_label.pack(pady=5)
        
        button_frame = tk.Frame(self.calibration_window, bg=COLORS['dark_bg'])
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="Завершить калибровку", 
                 font=('Arial', 11, 'bold'),
                 bg=COLORS['accent'], fg='white',
                 command=self.finish_square_calibration).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Очистить точки", 
                 font=('Arial', 11),
                 bg=COLORS['warning'], fg='white',
                 command=self.clear_calibration_points).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Отмена", 
                 font=('Arial', 11),
                 bg=COLORS['secondary'], fg='white',
                 command=self.cancel_calibration).pack(side=tk.LEFT, padx=10)
        
        self.calibration_mode = True
        self.calibration_points = []
        self.calibration_shape = None
        self.calibration_indicator.config(fg=COLORS['accent'])
        
        self.load_square_calibration_image()
        self.calibration_canvas.bind("<Button-1>", self.on_square_calibration_click)
        self.calibration_photo_ref = self.calibration_photo
    
    def load_square_calibration_image(self):
        """Загрузка изображения для калибровки"""
        try:
            image = Image.open(self.current_image_path)
            canvas_width = 600
            canvas_height = 400
            img_width, img_height = image.size
            width_ratio = canvas_width / img_width
            height_ratio = canvas_height / img_height
            ratio = min(width_ratio, height_ratio, 1.0)
            
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            self.calibration_scale_factor = ratio
            self.calibration_photo = ImageTk.PhotoImage(image)
            self.calibration_canvas.delete("all")
            self.calibration_canvas.create_image(0, 0, anchor=tk.NW, image=self.calibration_photo)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение для калибровки: {e}")
    
    def on_square_calibration_click(self, event):
        """Обработчик клика для калибровки"""
        if not self.calibration_mode:
            return
        
        x, y = event.x, event.y
        
        if len(self.calibration_points) >= 4:
            self.calibration_info_label.config(
                text=f"Уже выбрано 4 точки. Нажмите 'Очистить точки' чтобы начать заново.",
                fg=COLORS['warning']
            )
            return
        
        self.calibration_points.append((x, y))
        
        radius = 4
        point_id = self.calibration_canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                                                      fill='red', outline='white', width=2)
        text_id = self.calibration_canvas.create_text(x, y-15,
                                                     text=str(len(self.calibration_points)),
                                                     fill='yellow', font=('Arial', 12, 'bold'))
        
        self.calibration_info_label.config(
            text=f"Выбрано точек: {len(self.calibration_points)}/4",
            fg=COLORS['accent']
        )
        
        if not hasattr(self, 'calibration_objects'):
            self.calibration_objects = []
        self.calibration_objects.extend([point_id, text_id])
        
        # Рисование линий между точками
        if len(self.calibration_points) >= 2:
            for i in range(len(self.calibration_points) - 1):
                x1, y1 = self.calibration_points[i]
                x2, y2 = self.calibration_points[i + 1]
                line_id = self.calibration_canvas.create_line(x1, y1, x2, y2,
                                                            fill='cyan', width=2, dash=(4, 2))
                self.calibration_objects.append(line_id)
        
        if len(self.calibration_points) == 4:
            x1, y1 = self.calibration_points[3]
            x2, y2 = self.calibration_points[0]
            line_id = self.calibration_canvas.create_line(x1, y1, x2, y2,
                                                        fill='cyan', width=2, dash=(4, 2))
            self.calibration_objects.append(line_id)
            self.show_calibration_preview()
    
    def show_calibration_preview(self):
        """Показ предварительного просмотра калибровки"""
        if len(self.calibration_points) != 4:
            return
        
        shape = detect_shape_from_points(self.calibration_points)
        if shape:
            shape_text = {"square": "КВАДРАТ", "rectangle": "ПРЯМОУГОЛЬНИК", "rhombus": "РОМБ"}.get(shape, "НЕОПРЕДЕЛЕНО")
            self.calibration_info_label.config(text=f"Фигура определена как: {shape_text}", fg=COLORS['chart_3'])
        
        try:
            width_mm = float(self.calibration_width_var.get())
            height_mm = float(self.calibration_height_var.get())
            
            factor_x, factor_y, actual_width_px, actual_height_px = calculate_calibration_factors(
                self.calibration_points, self.calibration_scale_factor, width_mm, height_mm
            )
            
            info_text = f"Предварительные коэффициенты:\n"
            info_text += f"• По ширине: 1px = {factor_x:.4f} мм\n"
            info_text += f"• По высоте: 1px = {factor_y:.4f} мм\n"
            info_text += f"• Измерено: {actual_width_px:.1f}x{actual_height_px:.1f} px"
            
            self.calibration_info_label.config(text=info_text)
            
        except ValueError:
            self.calibration_info_label.config(
                text="Введите корректные размеры для расчета",
                fg=COLORS['warning']
            )
    
    def clear_calibration_points(self):
        """Очистка всех точек калибровки"""
        self.calibration_points = []
        self.calibration_shape = None
        if hasattr(self, 'calibration_objects'):
            for obj_id in self.calibration_objects:
                try:
                    self.calibration_canvas.delete(obj_id)
                except:
                    pass
            self.calibration_objects = []
        
        self.calibration_info_label.config(
            text="Точки очищены. Выберите 4 точки заново.",
            fg=COLORS['accent']
        )
        
        self.calibration_canvas.delete("all")
        self.calibration_canvas.create_image(0, 0, anchor=tk.NW, image=self.calibration_photo)
    
    def finish_square_calibration(self):
        """Завершение калибровки и сохранение коэффициентов"""
        if len(self.calibration_points) != 4:
            messagebox.showwarning("Внимание", "Выберите 4 точки для калибровки!")
            return
        
        try:
            known_width_mm = float(self.calibration_width_var.get())
            known_height_mm = float(self.calibration_height_var.get())
            
            if known_width_mm <= 0 or known_height_mm <= 0:
                messagebox.showwarning("Внимание", "Размеры должны быть больше 0!")
                return
            
            factor_x, factor_y, actual_width_px, actual_height_px = calculate_calibration_factors(
                self.calibration_points, self.calibration_scale_factor, known_width_mm, known_height_mm
            )
            
            self.calibration_factor_x = factor_x
            self.calibration_factor_y = factor_y
            
            self.calibration_label_x.config(text=f"X: 1 пиксель = {self.calibration_factor_x:.4f} мм")
            self.calibration_label_y.config(text=f"Y: 1 пиксель = {self.calibration_factor_y:.4f} мм")
            self.calibration_indicator.config(fg=COLORS['accent'])
            
            self.calibration_window.destroy()
            self.calibration_mode = False
            
            shape = detect_shape_from_points(self.calibration_points)
            
            success_text = f"Калибровка завершена!\n\n"
            success_text += f"Результаты:\n"
            success_text += f"• По ширине: 1px = {self.calibration_factor_x:.4f} мм\n"
            success_text += f"• По высоте: 1px = {self.calibration_factor_y:.4f} мм\n\n"
            success_text += f"Измеренные размеры:\n"
            success_text += f"• В пикселях: {actual_width_px:.1f} x {actual_height_px:.1f} px\n"
            success_text += f"• В миллиметрах: {known_width_mm:.1f} x {known_height_mm:.1f} мм\n\n"
            success_text += f"Форма: {shape.upper() if shape else 'НЕ ОПРЕДЕЛЕНА'}"
            
            messagebox.showinfo("Успех", success_text)
            
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числа для размеров!")
    
    def cancel_calibration(self):
        """Отмена калибровки"""
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_shape = None
        if hasattr(self, 'calibration_window'):
            self.calibration_window.destroy()
        if hasattr(self, 'calibration_objects'):
            del self.calibration_objects
        messagebox.showinfo("Информация", "Калибровка отменена")
    
    # ==================== СТАТИСТИКА И ЭКСПОРТ ====================
    
    def show_database_stats(self):
        """Отображение статистики базы данных"""
        try:
            status = self.db.check_database_status()
            
            if not status:
                messagebox.showinfo("Статистика БД", "Не удалось получить статистику базы данных")
                return
            
            daily_stats = self.db.get_daily_statistics()
            available_dates = self.db.get_available_dates()
            
            stats_text = create_database_stats_text(status, daily_stats, available_dates)
            messagebox.showinfo("Статистика БД", stats_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка получения статистики: {e}")
    
    def export_all_data(self):
        """Экспорт всех данных в файл"""
        try:
            export_df = self.db.get_crayfish_export_data()
            
            if export_df.empty:
                messagebox.showwarning("Экспорт", "В базе данных нет данных для экспорта")
                return
            
            file_path = filedialog.asksaveasfilename(
                title="Экспорт данных (размер, номер рака, дата)",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ],
                initialfile="crayfish_data_export"
            )
            
            if file_path:
                self._save_export_file(file_path, export_df)
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    def export_by_date(self):
        """Экспорт данных по выбранной дате"""
        try:
            available_dates = self.db.get_available_dates()
            
            if not available_dates:
                messagebox.showwarning("Экспорт", "В базе данных нет данных для экспорта")
                return
            
            self._create_date_selection_dialog(available_dates)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выборе даты: {e}")
    
    def _save_export_file(self, file_path, export_df):
        """Сохранение экспортированных данных в файл"""
        if file_path.endswith('.xlsx'):
            try:
                import openpyxl
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Данные_раков', index=False)
            except ImportError:
                csv_path = file_path.replace('.xlsx', '.csv')
                export_df.to_csv(csv_path, index=False, encoding='utf-8')
                file_path = csv_path
        else:
            export_df.to_csv(file_path, index=False, encoding='utf-8')
        
        messagebox.showinfo("Успех", 
                          f"Данные экспортированы в:\n{file_path}\n\n"
                          f"Экспортировано:\n"
                          f"• Строк: {len(export_df)}\n"
                          f"• Колонки: Номер рака, Ширина (мм), Высота (мм), Угол поворота, Дата, Время, Уверенность")
    
    def _create_date_selection_dialog(self, available_dates):
        """Создание диалога выбора даты для экспорта"""
        date_window = tk.Toplevel(self.root)
        date_window.title("Выбор даты для экспорта")
        date_window.geometry("400x300")
        date_window.configure(bg=COLORS['dark_bg'])
        
        tk.Label(date_window, 
                text="ВЫБЕРИТЕ ДАТУ ДЛЯ ЭКСПОРТА",
                font=('Arial', 14, 'bold'),
                fg=COLORS['primary'],
                bg=COLORS['dark_bg']).pack(pady=20)
        
        listbox_frame = tk.Frame(date_window, bg=COLORS['card_bg'])
        listbox_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        listbox = tk.Listbox(listbox_frame, bg='#1e2b3a', fg='white',
                           font=('Arial', 11), selectbackground=COLORS['primary'])
        
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        
        for date_str in available_dates:
            listbox.insert(tk.END, date_str)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        button_frame = tk.Frame(date_window, bg=COLORS['dark_bg'])
        button_frame.pack(pady=20)
        
        def export_selected_date():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("Внимание", "Выберите дату для экспорта!")
                return
            
            selected_date = listbox.get(selection[0])
            date_window.destroy()
            self.perform_export_by_date(selected_date)
        
        tk.Button(button_frame, text="Экспортировать выбранную дату", 
                 font=('Arial', 11, 'bold'),
                 bg=COLORS['accent'], fg='white',
                 command=export_selected_date).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Отмена", 
                 font=('Arial', 11),
                 bg=COLORS['secondary'], fg='white',
                 command=date_window.destroy).pack(side=tk.LEFT, padx=10)
    
    def perform_export_by_date(self, selected_date):
        """Выполнение экспорта данных за выбранную дату"""
        try:
            export_df = self.db.get_crayfish_export_by_date(selected_date)
            
            if export_df.empty:
                messagebox.showwarning("Экспорт", f"Нет данных за дату {selected_date}")
                return
            
            file_path = filedialog.asksaveasfilename(
                title=f"Экспорт данных за {selected_date}",
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("Excel files", "*.xlsx"),
                    ("All files", "*.*")
                ],
                initialfile=f"crayfish_data_{selected_date.replace('-', '_')}"
            )
            
            if file_path:
                self._save_export_file(file_path, export_df)
                
                messagebox.showinfo("Успех", 
                                  f"Данные за {selected_date} экспортированы в:\n{file_path}\n\n"
                                  f"Статистика экспорта:\n"
                                  f"• Строк: {len(export_df)}\n"
                                  f"• Уникальных особей: {export_df['Номер рака'].nunique() if 'Номер рака' in export_df.columns else 0}\n"
                                  f"• Средняя ширина: {export_df['Ширина (мм)'].mean():.1f} мм\n"
                                  f"• Средняя высота: {export_df['Высота (мм)'].mean():.1f} мм")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    def clear_database_ui(self):
        """Очистка всей базы данных после подтверждения"""
        if messagebox.askyesno("Подтверждение", 
                              "Вы уверены, что хотите полностью очистить базу данных?\nВсе данные будут безвозвратно удалены!"):
            success = self.db.clear_database()
            
            if success:
                self.measurement_history.clear()
                self.crayfish_counter = 1
                self.session_history.clear()
                self.update_db_stats()
                messagebox.showinfo("Успех", "База данных очищена")
            else:
                messagebox.showerror("Ошибка", "Не удалось очистить базу данных")
    
    # ==================== TELEGRAM БОТ ====================
    
    def setup_telegram_bot(self):
        """Настройка и запуск Telegram бота"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Настройка Telegram Bot")
        dialog.geometry("500x400")
        dialog.configure(bg=COLORS['dark_bg'])
        
        tk.Label(dialog, text="НАСТРОЙКА TELEGRAM BOT", 
                font=('Arial', 16, 'bold'),
                fg=COLORS['primary'],
                bg=COLORS['dark_bg']).pack(pady=20)
        
        instructions = tk.Text(dialog, height=8, width=50, wrap=tk.WORD,
                             bg='#1e2b3a', fg='white', font=('Arial', 10))
        instructions.pack(pady=10, padx=20)
        
        instructions_text = """Как создать Telegram бота:
1. Откройте Telegram и найдите @BotFather
2. Отправьте команду /newbot
3. Выберите имя для бота
4. Получите токен
5. Вставьте токен ниже

Бот будет предоставлять API:
/detect - анализ фото
/stats - статистика
/charts - графики
/export - экспорт данных
/status - статус системы
/info - информация о боте"""
        
        instructions.insert(1.0, instructions_text)
        instructions.config(state=tk.DISABLED)
        
        token_frame = tk.Frame(dialog, bg=COLORS['dark_bg'])
        token_frame.pack(pady=10)
        
        tk.Label(token_frame, text="Токен бота:", 
                font=('Arial', 11),
                fg=COLORS['text_primary'],
                bg=COLORS['dark_bg']).pack(side=tk.LEFT)
        
        token_entry = tk.Entry(token_frame, textvariable=self.bot_token_var,
                             font=('Arial', 10), width=40)
        token_entry.pack(side=tk.LEFT, padx=10)
        
        btn_frame = tk.Frame(dialog, bg=COLORS['dark_bg'])
        btn_frame.pack(pady=20)
        
        def save_and_start():
            token = token_entry.get().strip()
            if not token:
                messagebox.showerror("Ошибка", "Введите токен бота!")
                return
            self.bot_token_var.set(token)
            self.start_telegram_bot()
            dialog.destroy()
        
        tk.Button(btn_frame, text="Сохранить и запустить", 
                 font=('Arial', 11, 'bold'),
                 bg=COLORS['accent'], fg='white',
                 command=save_and_start).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Отмена", 
                 font=('Arial', 11),
                 bg=COLORS['secondary'], fg='white',
                 command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def start_telegram_bot(self):
        """Запуск Telegram бота в отдельном потоке"""
        token = self.bot_token_var.get().strip()
        
        if not token:
            messagebox.showerror("Ошибка", "Введите токен Telegram бота!")
            return
        
        if self.telegram_bot and self.telegram_bot.running:
            messagebox.showinfo("Информация", "Бот уже запущен!")
            return
        
        try:
            self.telegram_bot = TelegramBotManager(self, token)
            
            if not self.telegram_bot.bot:
                messagebox.showerror("Ошибка", "Не удалось инициализировать бота")
                return
            
            self.bot_thread = threading.Thread(
                target=self.telegram_bot.start_polling,
                daemon=True
            )
            self.bot_thread.start()
            
            self.bot_status_label.config(text="Бот запущен", fg=COLORS['accent'])
            self.bot_indicator.config(fg=COLORS['accent'])
            self.status_label.config(text="Telegram Bot запущен")
            
            messagebox.showinfo("Успех", "Telegram Bot успешно запущен!\n\nБот доступен по командам:\n/detect - анализ фото\n/stats - статистика\n/charts - графики\n/export - экспорт данных\n/status - статус системы\n/info - информация о боте")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Ошибка запуска бота: {error_msg}")
            messagebox.showerror("Ошибка", f"Не удалось запустить бота: {error_msg}")
            self.bot_status_label.config(text="Ошибка запуска", fg=COLORS['secondary'])
    
    def stop_telegram_bot(self):
        """Остановка Telegram бота"""
        if self.telegram_bot:
            self.telegram_bot.stop()
            self.telegram_bot = None
            self.bot_thread = None
            
            self.bot_status_label.config(text="Бот остановлен", fg=COLORS['secondary'])
            self.bot_indicator.config(fg='#ff4444')
            self.status_label.config(text="Telegram Bot остановлен")
            
            messagebox.showinfo("Информация", "Telegram бот остановлен")
        else:
            messagebox.showinfo("Информация", "Бот не запущен")
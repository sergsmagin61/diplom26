import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw
import os
from pathlib import Path
from ultralytics import YOLO
import threading
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from collections import defaultdict
from matplotlib import rcParams
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
import warnings
import math
import sys
import sqlite3
import json
from database import DatabaseManager
from telegram_bot import TelegramBotManager
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime не установлен. ONNX модели не будут поддерживаться.")
class ModernCrayfishDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Crayfish AI Studio Pro")
        self.root.geometry("1600x1000") 
        self.root.configure(bg='#0f1a2b') 
        self.calibration_factor_x = 0.15 
        self.calibration_factor_y = 0.15  
        self.calibration_mode = False      
        self.calibration_points = []       
        self.calibration_known_width_mm = 10.0   
        self.calibration_known_height_mm = 10.0  
        self.calibration_shape = None   
        
        self.setup_matplotlib_style()
        
        self.colors = {
            'primary': '#00d4ff',     #основные элементы
            'secondary': '#ff6b6b',   #ошибки/предупреждения
            'accent': '#00ff88',      #успех/активность
            'warning': '#ffcc00',     #предупреждения
            'dark_bg': '#0f1a2b',     #основной фон
            'card_bg': '#1a2b3c',     #карточки/панели
            'text_primary': '#ffffff',   #основной текст
            'text_secondary': '#8899aa', #второстепенный текст
            'chart_1': '#00d4ff',     #график 1
            'chart_2': '#ff6b6b',     #график 2
            'chart_3': '#00ff88',     #график 3
            'chart_4': '#ffcc00',     #график 4
            'chart_5': '#9d4edd'      #график 5
        }
        
        self.model = None          
        self.model_type = None    
        self.models = {}         
        self.current_model = ""   
        self.confidence = 0.5   
        self.current_image = None  
        self.processed_image = None  
        self.detections = []        
        self.current_image_path = None  
        self.photo = None          
        self.photo_original = None      
        self.photo_result = None      
        
        self.telegram_bot = None  
        self.bot_thread = None 
        self.db = DatabaseManager() 
        
        self.measurement_history = defaultdict(list)
        self.crayfish_counter = 1   
        self.session_history = [] 
        self.display_scale_factor = 1.0 
        self.original_image_size = None    
        self.canvas_image_id = None  
        
        self.load_existing_data()
        
        self.setup_modern_ui() 
        self.load_available_models()
        
        self.check_database_health()
    
    def setup_matplotlib_style(self):
        try:
            plt.style.use('dark_background')
        except:
            pass
        
        rcParams['axes.facecolor'] = '#1a2b3c'      # Цвет фона осей
        rcParams['axes.edgecolor'] = '#00d4ff'      # Цвет границ осей
        rcParams['axes.labelcolor'] = 'white'       # Цвет подписей осей
        rcParams['text.color'] = 'white'            # Цвет текста
        rcParams['xtick.color'] = 'white'           # Цвет меток по X
        rcParams['ytick.color'] = 'white'           # Цвет меток по Y
        rcParams['grid.color'] = '#2a3b4c'          # Цвет сетки
        rcParams['figure.facecolor'] = '#0f1a2b'    # Цвет фона фигуры
        rcParams['figure.edgecolor'] = '#0f1a2b'    # Цвет границ фигуры
    
    def check_database_health(self):
        status = self.db.check_database_status()
        
        if status:
            message = f"БД: {status['crayfish_count']} раков, {status['measurements_count']} измерений"
            if hasattr(self, 'status_label'):
                self.status_label.config(text=message)
            print(f"Статус БД: {message}")
    def load_existing_data(self):
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
    def setup_modern_ui(self):
        self.setup_styles()
        main_container = tk.Frame(self.root, bg=self.colors['dark_bg'])
        main_container.pack(fill=tk.BOTH, expand=True)
        self.create_header(main_container)
        content_frame = tk.Frame(main_container, bg=self.colors['dark_bg'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        #ЛЕВАЯ ПАНЕЛЬ
        left_frame = tk.Frame(content_frame, bg=self.colors['dark_bg'], width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        #ЦЕНТРАЛЬНАЯ ПАНЕЛЬ
        center_frame = tk.Frame(content_frame, bg=self.colors['dark_bg'])
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        #ПРАВАЯ ПАНЕЛЬ
        right_frame = tk.Frame(content_frame, bg=self.colors['dark_bg'], width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)  # Фиксируем ширину
        
        self.create_quick_actions(left_frame)      # Левая панель
        self.create_image_viewer(center_frame)     # Центр
        self.create_analytics_panel(right_frame)   # Правая панель
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Modern.TButton',
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 11, 'bold'),
                       padding=(20, 12))
        
        style.map('Modern.TButton',
                 background=[('active', '#00b8e6'), ('pressed', '#0099cc')])
        
        style.configure('Card.TFrame', background=self.colors['card_bg'])
        
        style.configure('Title.TLabel', 
                       background=self.colors['dark_bg'],
                       foreground=self.colors['primary'],
                       font=('Arial', 16, 'bold'))
        
        style.configure('Vertical.TScrollbar',
                       background=self.colors['primary'],
                       troughcolor=self.colors['card_bg'],
                       arrowcolor='white',
                       bordercolor=self.colors['primary'],
                       lightcolor=self.colors['primary'],
                       darkcolor=self.colors['primary'])
    
    #ВЕРХНЯЯ ПАНЕЛЬ
    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg=self.colors['dark_bg'], height=80)
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        header_frame.pack_propagate(False)
        title_label = tk.Label(header_frame, 
                              text="CRAYFISH AI STUDIO PRO v2.0 (OBB SUPPORT)",
                              font=('Arial', 20, 'bold'),
                              fg=self.colors['primary'],
                              bg=self.colors['dark_bg'])
        title_label.pack(side=tk.LEFT, padx=20)
        self.status_label = tk.Label(header_frame,
                                   text="Система готова | БД: подключена",
                                   font=('Arial', 12),
                                   fg=self.colors['accent'],
                                   bg=self.colors['dark_bg'])
        self.status_label.pack(side=tk.RIGHT, padx=20)
        self.create_indicators(header_frame)
    
    #ИНДИКАТОРЫ СОСТОЯНИЯ
    def create_indicators(self, parent):
        indicators_frame = tk.Frame(parent, bg=self.colors['dark_bg'])
        indicators_frame.pack(side=tk.RIGHT, padx=20)
        
        self.db_indicator = tk.Label(indicators_frame, text="DB", 
                                   font=('Arial', 12),
                                   fg=self.colors['accent'], bg=self.colors['dark_bg'])
        self.db_indicator.pack(side=tk.LEFT, padx=5)
        
        self.model_indicator = tk.Label(indicators_frame, text="ML", 
                                      font=('Arial', 12),
                                      fg='#ff4444', bg=self.colors['dark_bg'])
        self.model_indicator.pack(side=tk.LEFT, padx=5)
        
        self.calibration_indicator = tk.Label(indicators_frame, text="CAL", 
                                            font=('Arial', 12),
                                            fg='#ffcc00', bg=self.colors['dark_bg'])
        self.calibration_indicator.pack(side=tk.LEFT, padx=5)
        
        self.bot_indicator = tk.Label(indicators_frame, text="BOT", 
                                     font=('Arial', 12),
                                     fg='#ff4444', bg=self.colors['dark_bg'])
        self.bot_indicator.pack(side=tk.LEFT, padx=5)
    
    #ЛЕВАЯ ПАНЕЛЬ
    def create_quick_actions(self, parent):
        main_container = ttk.Frame(parent, style='Card.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas = tk.Canvas(main_container, bg=self.colors['card_bg'], 
                          highlightthickness=1, 
                          highlightbackground=self.colors['primary'])
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
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self.scrollable_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.scrollable_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        self._fill_quick_actions_content()
    def _fill_quick_actions_content(self):
        title = tk.Label(self.scrollable_frame, text="Быстрые действия", 
                        font=('Arial', 14, 'bold'),
                        fg=self.colors['text_primary'],
                        bg=self.colors['card_bg'])
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
        
        # Отступ внизу
        spacer = tk.Frame(self.scrollable_frame, height=20, bg=self.colors['card_bg'])
        spacer.pack(fill=tk.X)
    
    #ИНДИКАТОР КАЛИБРОВКИ
    def create_calibration_indicator(self, parent):
        calibration_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        calibration_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(calibration_frame, text="Калибровочные коэффициенты", 
                font=('Arial', 11, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['card_bg']).pack(anchor=tk.W)
        self.calibration_label_x = tk.Label(calibration_frame,
                                         text=f"X: 1 пиксель = {self.calibration_factor_x:.4f} мм",
                                         font=('Arial', 10),
                                         fg=self.colors['accent'],
                                         bg=self.colors['card_bg'])
        self.calibration_label_x.pack(pady=2)
        
        # Коэффициент Y (длина)
        self.calibration_label_y = tk.Label(calibration_frame,
                                         text=f"Y: 1 пиксель = {self.calibration_factor_y:.4f} мм",
                                         font=('Arial', 10),
                                         fg=self.colors['accent'],
                                         bg=self.colors['card_bg'])
        self.calibration_label_y.pack(pady=2)
        
        # Кнопка сброса калибровки
        reset_btn = tk.Button(calibration_frame, text="Сбросить калибровку",
                             font=('Arial', 9),
                             bg=self.colors['secondary'], fg='white',
                             command=self.reset_calibration)
        reset_btn.pack(pady=5)
    
    def create_confidence_slider(self, parent):
        slider_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        slider_frame.pack(fill=tk.X, padx=20, pady=20)
        tk.Label(slider_frame, text="Порог уверенности", 
                font=('Arial', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['card_bg']).pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.5)
        slider = ttk.Scale(slider_frame, from_=0.1, to=0.9, 
                          variable=self.confidence_var, orient=tk.HORIZONTAL)
        slider.pack(fill=tk.X, pady=10)
        slider.configure(command=self.on_slider_change)
        self.confidence_label = tk.Label(slider_frame, 
                                       text=f"Текущее значение: 0.50",
                                       font=('Arial', 10),
                                       fg=self.colors['text_secondary'],
                                       bg=self.colors['card_bg'])
        self.confidence_label.pack()
    
    #ОБРАБОТЧИК ПОЛЗУНКА
    def on_slider_change(self, value):
        self.confidence = float(value)
        self.confidence_label.config(text=f"Текущее значение: {self.confidence:.2f}")
    def create_telegram_bot_ui(self, parent):
        telegram_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        telegram_frame.pack(fill=tk.X, padx=20, pady=10)
        tk.Label(telegram_frame, text="Telegram Bot API", 
                font=('Arial', 11, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['card_bg']).pack(anchor=tk.W)
        
        # Поле для токена
        tk.Label(telegram_frame, text="Токен бота:", 
                font=('Arial', 9),
                fg=self.colors['text_secondary'],
                bg=self.colors['card_bg']).pack(anchor=tk.W, pady=(5,0))
        
        self.bot_token_var = tk.StringVar()
        token_entry = tk.Entry(telegram_frame, textvariable=self.bot_token_var, 
                              font=('Arial', 10), width=30)
        token_entry.pack(fill=tk.X, pady=5)
        
        btn_frame = tk.Frame(telegram_frame, bg=self.colors['card_bg'])
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="Запустить бота", 
                 font=('Arial', 9),
                 bg=self.colors['accent'], fg='white',
                 command=self.start_telegram_bot).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_frame, text="Остановить", 
                 font=('Arial', 9),
                 bg=self.colors['secondary'], fg='white',
                 command=self.stop_telegram_bot).pack(side=tk.LEFT, padx=2)
        
        self.bot_status_label = tk.Label(telegram_frame, text="Бот не запущен", 
                                       font=('Arial', 9),
                                       fg=self.colors['secondary'],
                                       bg=self.colors['card_bg'])
        self.bot_status_label.pack(pady=5)
    
    #СТАТИСТИКА БД В РЕАЛЬНОМ ВРЕМЕНИ
    def create_realtime_stats(self, parent):
        stats_frame = tk.Frame(parent, bg=self.colors['card_bg'])
        stats_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(stats_frame, text="Статистика БД", 
                font=('Arial', 12, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['card_bg']).pack(anchor=tk.W)
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
            frame = tk.Frame(stats_frame, bg=self.colors['card_bg'])
            frame.pack(fill=tk.X, pady=5)
            tk.Label(frame, text=text, font=('Arial', 9),
                   fg=self.colors['text_secondary'], bg=self.colors['card_bg']).pack(side=tk.LEFT)
            value_label = tk.Label(frame, text="0", font=('Arial', 10, 'bold'),
                                fg=self.colors['accent'], bg=self.colors['card_bg'])
            value_label.pack(side=tk.RIGHT)
            
            self.stats_labels[key] = value_label
        self.update_db_stats()
    def update_db_stats(self):
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
    
    #ЦЕНТРАЛЬНАЯ ПАНЕЛЬ
    def create_image_viewer(self, parent):
        viewer_frame = ttk.Frame(parent, style='Card.TFrame')
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.notebook = ttk.Notebook(viewer_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        original_tab = ttk.Frame(self.notebook)
        self.notebook.add(original_tab, text="Исходное")
        
        result_tab = ttk.Frame(self.notebook)
        self.notebook.add(result_tab, text="Результат")
        
        self.original_canvas = tk.Canvas(original_tab, bg=self.colors['card_bg'], 
                                       highlightthickness=1, highlightbackground=self.colors['primary'])
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.result_canvas = tk.Canvas(result_tab, bg=self.colors['card_bg'],
                                     highlightthickness=1, highlightbackground=self.colors['primary'])
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.original_canvas.bind("<Configure>", self.on_canvas_resize)
        self.result_canvas.bind("<Configure>", self.on_canvas_resize)
        
        self.original_label = tk.Label(original_tab, text="Загрузите изображение...",
                                     fg=self.colors['text_secondary'], bg=self.colors['card_bg'])
        self.original_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.result_label = tk.Label(result_tab, text="Результат появится здесь...",
                                   fg=self.colors['text_secondary'], bg=self.colors['card_bg'])
        self.result_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    #ОБРАБОТЧИК ИЗМЕНЕНИЯ РАЗМЕРА CANVAS
    def on_canvas_resize(self, event):
        if event.widget == self.original_canvas and hasattr(self, 'photo_original') and self.photo_original:
            self.redraw_original_image()
        elif event.widget == self.result_canvas and hasattr(self, 'photo_result') and self.photo_result:
            self.redraw_result_image()
    
    def redraw_original_image(self):
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            return
        
        try:
            image = Image.open(self.current_image_path)
            self._display_image_on_canvas(image, self.original_canvas, 'original')
        except Exception as e:
            print(f"Ошибка перерисовки исходного изображения: {e}")
    
    #ПЕРЕРИСОВКА ИЗОБРАЖЕНИЯ РЕЗУЛЬТАТА
    def redraw_result_image(self):
        if self.processed_image is None:
            return
        
        try:
            if isinstance(self.processed_image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
                self._display_image_on_canvas(image, self.result_canvas, 'result')
        except Exception as e:
            print(f"Ошибка перерисовки результата: {e}")
    def _display_image_on_canvas(self, image, canvas, img_type):
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
            
            print(f"Изображение отмасштабировано: {new_width}x{new_height} (масштаб: {scale:.3f})")
            
        except Exception as e:
            print(f"Ошибка отображения изображения: {e}")
            if img_type == 'original':
                self.original_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            else:
                self.result_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    
    #ОТОБРАЖЕНИЕ ИСХОДНОГО ИЗОБРАЖЕНИЯ
    def display_image_original(self, image_path):
        try:
            image = Image.open(image_path)
            self.original_image_size = image.size
            self._display_image_on_canvas(image, self.original_canvas, 'original')
            self.status_label.config(text="Изображение загружено")
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
    
    #ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА
    def display_image_result(self, image):
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self._display_image_on_canvas(image, self.result_canvas, 'result')
        except Exception as e:
            print(f"Ошибка отображения обработанного изображения: {e}")
            messagebox.showerror("Ошибка", f"Не удалось отобразить результат: {e}")
    
    #ПРАВАЯ ПАНЕЛЬ
    def create_analytics_panel(self, parent):

        analytics_frame = ttk.Frame(parent, style='Card.TFrame')
        analytics_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        tk.Label(analytics_frame, text="Детальная аналитика", 
                font=('Arial', 14, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['card_bg']).pack(pady=15)
        
        details_frame = tk.Frame(analytics_frame, bg=self.colors['card_bg'])
        details_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.detections_text = tk.Text(details_frame, height=20, width=35,
                                     bg='#1e2b3a', fg=self.colors['text_primary'],
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
    
    #ЗАГРУЗКА ДОСТУПНЫХ МОДЕЛЕЙ
    def load_available_models(self):
        model_extensions = ['*.pt', '*.onnx', '*.engine']
        found_models = []
        for ext in model_extensions:
            for path in Path('.').glob(ext):
                if path.exists():
                    model_name = path.name
                    self.models[model_name] = str(path)
                    found_models.append(model_name)
                    print(f"Найдена модель: {model_name}")
        search_paths = [
            "training_results/train/weights/",
            "training_results/train2/weights/",
            "models/",
            "optimized_models/"
        ]
        
        for search_path in search_paths:
            for ext in model_extensions:
                for path in Path(search_path).glob(ext):
                    if path.exists():
                        model_name = f"[{search_path}]{path.name}"
                        self.models[model_name] = str(path)
                        found_models.append(model_name)
                        print(f"Найдена модель: {model_name}")
        
        if self.models:
            self.current_model = list(self.models.keys())[0]
            self.load_model_thread(self.models[self.current_model])
            self.model_indicator.config(fg=self.colors['accent'])
        else:
            self.status_label.config(text="Модели не найдены")
            print("Модели не найдены. Поместите .pt, .onnx или .engine файлы в папку программы")
    
    #ЗАГРУЗКА МОДЕЛИ В ОТДЕЛЬНОМ ПОТОКЕ
    def load_model_thread(self, model_path):
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
                    self.root.after(0, lambda: self.model_indicator.config(fg=self.colors['accent'], text="OBB"))
                
            except Exception as e:
                error_msg = str(e)
                print(f"Ошибка загрузки модели: {error_msg}")
                self.root.after(0, lambda: self.status_label.config(text=f"Ошибка загрузки: {error_msg[:50]}"))
                self.root.after(0, lambda: self.model_indicator.config(fg='#ff4444'))
        
        threading.Thread(target=load, daemon=True).start()
    
    def load_model(self):
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
    
    def load_image(self):
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
    
    def start_detection(self):
        if not self.model:
            messagebox.showwarning("Внимание", "Сначала загрузите модель!")
            return
        
        if not self.current_image_path:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
            return
        
        self.status_label.config(text="Выполняется детекция...")
        
        threading.Thread(target=self.run_detection, daemon=True).start()
    def run_detection(self):
        """
        Выполняет детекцию на загруженном изображении:
        1. Запускает инференс модели
        2. Обрабатывает результаты
        3. Отрисовывает рамки на изображении

        """
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
                
                #ОБРАБОТКА OBB ДЕТЕКЦИЙ
                if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                    print(f"Найдены OBB детекции: {len(result.obb)}")
                    self.detections = result.obb
                    
                    for i, obb in enumerate(result.obb):
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
                    
                    self.processed_image = img
                    
                #ОБРАБОТКА ОБЫЧНЫХ ДЕТЕКЦИЙ
                elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    print(f"Найдены обычные детекции: {len(result.boxes)}")
                    
                    img = result.orig_img.copy()
                    self.detections = result.boxes
                    
                    for i, box in enumerate(result.boxes):
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
                    
                    self.processed_image = img
                else:
                    print("Детекции не найдены")
                    self.processed_image = img
                    self.detections = []
                session_data = {
                    'timestamp': datetime.now(),
                    'image_path': self.current_image_path,
                    'total_detections': len(self.detections) if self.detections else 0,
                    'processing_time': processing_time,
                    'confidence_threshold': self.confidence
                }
                
                session_id = self.db.save_session(session_data)
                
                if session_id and self.detections and len(self.detections) > 0:
                    inference_data = {
                        'timestamp': datetime.now(),
                        'image_path': self.current_image_path,
                        'image_width': img.shape[1],
                        'image_height': img.shape[0],
                        'inference_time_ms': inference_time_ms,
                        'detections_count': len(self.detections),
                        'avg_confidence': float(np.mean([det.conf.cpu().numpy() for det in self.detections])),
                        'model_name': self.current_model,
                        'model_type': self.model_type,
                        'confidence_threshold': self.confidence
                    }
                    self.db.save_ml_inference_log(inference_data)
                    
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
    
    #СОХРАНЕНИЕ ДЕТЕКЦИЙ В БД
    def save_detections_to_db(self, session_id):
        if not self.detections:
            return
        
        print(f"Сохранение {len(self.detections)} обнаружений в БД...")
        
        for i, det in enumerate(self.detections):
            try:
                if hasattr(det, 'conf'):
                    if hasattr(det.conf, '__len__') and len(det.conf) > 0:
                        confidence = float(det.conf[0].item())
                    else:
                        confidence = float(det.conf)
                else:
                    confidence = 0.5
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
                estimated_width_mm = float(width_px * self.calibration_factor_x)
                estimated_height_mm = float(height_px * self.calibration_factor_y)
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
    
    #ОБНОВЛЕНИЕ UI ПОСЛЕ ДЕТЕКЦИИ
    def update_ui_after_detection(self, processing_time, session_id, inference_time_ms):
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
    
    #ОБНОВЛЕНИЕ ТЕКСТОВОЙ ИНФОРМАЦИИ О ДЕТЕКЦИИ
    def update_detections_info(self, processing_time, session_id, inference_time_ms):
        self.detections_text.config(state=tk.NORMAL)
        self.detections_text.delete(1.0, tk.END)
        
        model_type = ""
        if self.current_model:
            ext = Path(self.current_model).suffix.lower()
            if ext == '.engine':
                model_type = "TensorRT (максимальная скорость)"
            elif ext == '.onnx':
                model_type = "ONNX (оптимизированная)"
            else:
                model_type = "YOLO (стандартная)"
        
        text = f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        text += f"         РЕЗУЛЬТАТЫ ДЕТЕКЦИИ          \n"
        text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        text += f"Тип модели: {model_type}\n"
        text += f"Общее время: {processing_time:.2f}с\n"
        text += f"Время ML: {inference_time_ms:.1f} мс\n"
        text += f"Обнаружено: {len(self.detections) if self.detections else 0}\n"
        text += f"Порог: {self.confidence:.2f}\n"
        text += f"Калибровка X: 1px = {self.calibration_factor_x:.4f}мм\n"
        text += f"Калибровка Y: 1px = {self.calibration_factor_y:.4f}мм\n"
        text += f"Файл: {os.path.basename(self.current_image_path)}\n"
        text += f"Сессия: {session_id}\n\n"
        if self.detections:
            text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            text += f"         ДЕТАЛИ ОБНАРУЖЕНИЙ          \n"
            text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            try:
                measurements_df = self.db.get_all_measurements()
                if not measurements_df.empty:
                    session_measurements = measurements_df[measurements_df['session_id'] == session_id]
                    
                    for i, (_, row) in enumerate(session_measurements.iterrows()):
                        text += f"\n┌── РАК #{i+1} ──────────────────────\n"
                        text += f"│ ID: {row['unique_id'] if 'unique_id' in row and not pd.isna(row['unique_id']) else 'Новый'}\n"
                        text += f"│ Уверенность: {row['confidence']:.3f}\n"
                        text += f"│ Размер: {row['width_mm']:.1f}x{row['height_mm']:.1f} мм\n"
                        if 'angle' in row and not pd.isna(row['angle']) and abs(row['angle']) > 0.01:
                            text += f"│ Угол: {row['angle']:.1f}°\n"
                        text += f"│ Размер в px: {row['width_px']:.0f}x{row['height_px']:.0f}\n"
                        text += f"│ Время: {row['timestamp'][:19] if isinstance(row['timestamp'], str) else str(row['timestamp'])[:19]}\n"
                        text += f"└──────────────────────────────────\n"
                else:
                    text += "\nДанные измерений временно недоступны\n"
                    
            except Exception as e:
                text += f"\nОшибка загрузки данных: {str(e)[:50]}\n"
        else:
            text += f"\nРАКИ НЕ ОБНАРУЖЕНЫ\n\n"
            text += f"Советы:\n"
            text += f"• Уменьшите порог уверенности (сейчас {self.confidence:.2f})\n"
            text += f"• Убедитесь, что раки видны на фото\n"
            text += f"• Попробуйте другое изображение\n"
            text += f"• Проверьте загруженную модель"
        
        text += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        self.detections_text.insert(1.0, text)
        self.detections_text.config(state=tk.DISABLED)
    
    def show_detection_result(self):
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
    
    #ПОКАЗ АНАЛИЗА РОСТА
    def show_growth_analysis(self):
        self.create_simple_growth_analysis()
    
    #СОЗДАНИЕ АНАЛИЗА РОСТА
    def create_simple_growth_analysis(self):
        try:
            daily_stats = self.db.get_daily_statistics()
            
            if daily_stats.empty:
                messagebox.showwarning("Внимание", "Нет данных в базе данных!\nСначала выполните детекцию на нескольких изображениях.")
                return
            
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Анализ роста раков по дням")
            analysis_window.geometry("1400x900")
            analysis_window.configure(bg=self.colors['dark_bg'])
            
            title_label = tk.Label(analysis_window, 
                                  text="АНАЛИЗ РОСТА РАКОВ ПО ДНЯМ",
                                  font=('Arial', 18, 'bold'),
                                  fg=self.colors['primary'],
                                  bg=self.colors['dark_bg'])
            title_label.pack(pady=20)
            
            info_text = f"Анализ основан на данных за {len(daily_stats)} дней\n"
            info_text += f"Период: {daily_stats['date'].min().strftime('%d.%m.%Y')} - {daily_stats['date'].max().strftime('%d.%m.%Y')}"
            
            info_label = tk.Label(analysis_window, text=info_text,
                                 font=('Arial', 12),
                                 fg=self.colors['text_primary'],
                                 bg=self.colors['dark_bg'])
            info_label.pack(pady=10)
            
            self.create_simple_growth_plots(analysis_window, daily_stats)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка создания анализа: {e}")
    
    #ПОСТРОЕНИЕ ГРАФИКОВ РОСТ
    def create_simple_growth_plots(self, parent, daily_stats):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('АНАЛИЗ РОСТА ПОПУЛЯЦИИ РАКОВ', 
                        fontsize=20, color='white', fontweight='bold', y=0.98)
            daily_stats_sorted = daily_stats.sort_values('date')
            
            #ГРАФИК 1: Средний размер по дням
            ax1 = axes[0, 0]
            if len(daily_stats_sorted) >= 1:
                line, = ax1.plot(daily_stats_sorted['date'], daily_stats_sorted['avg_width'], 
                                marker='o', markersize=8, color=self.colors['chart_1'], 
                                linewidth=3, label='Средний размер', zorder=5)
                if len(daily_stats_sorted) >= 3:
                    try:
                        from scipy.interpolate import make_interp_spline
                        
                        x_num = mdates.date2num(daily_stats_sorted['date'])
                        y_vals = daily_stats_sorted['avg_width'].values
                        
                        x_smooth = np.linspace(x_num.min(), x_num.max(), 300)
                        spline = make_interp_spline(x_num, y_vals, k=3)
                        y_smooth = spline(x_smooth)
                        
                        dates_smooth = mdates.num2date(x_smooth)
                        
                        ax1.plot(dates_smooth, y_smooth, '-', 
                                color=self.colors['chart_2'], linewidth=2, alpha=0.7,
                                label='Тренд роста', zorder=4)
                    except:
                        pass 
                
                ax1.fill_between(daily_stats_sorted['date'], 
                               daily_stats_sorted['min_width'], 
                               daily_stats_sorted['max_width'],
                               alpha=0.2, color=self.colors['chart_1'],
                               label='Min-Max диапазон', zorder=1)
                
                # Настройка внешнего вида
                ax1.set_title('Средний размер по дням', 
                             color='white', fontsize=14, fontweight='bold', pad=12)
                ax1.set_xlabel('Дата', color='white', fontsize=11)
                ax1.set_ylabel('Ширина (мм)', color='white', fontsize=11)
                ax1.grid(True, alpha=0.2, linestyle='--')
                
                # Адаптивный формат дат в зависимости от количества дней
                if len(daily_stats_sorted) <= 7:
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                    ax1.xaxis.set_major_locator(mdates.DayLocator())
                elif len(daily_stats_sorted) <= 14:
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                else:
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                    interval = max(3, len(daily_stats_sorted) // 8)
                    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
                
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                
                ax1.legend(facecolor=self.colors['card_bg'], edgecolor='white',
                          fontsize=9, loc='best', framealpha=0.9)
            
                if len(daily_stats_sorted) <= 8:
                    for i, row in daily_stats_sorted.iterrows():
                        ax1.annotate(f"{row['avg_width']:.1f}мм", 
                                   xy=(row['date'], row['avg_width']),
                                   xytext=(0, 8), textcoords='offset points',
                                   ha='center', fontsize=8, color='white', fontweight='bold')
            
            ax2 = axes[0, 1]
            all_measurements = self.db.get_all_measurements()
            
            if not all_measurements.empty and 'width_mm' in all_measurements.columns:
                sizes = pd.to_numeric(all_measurements['width_mm'], errors='coerce')
                sizes = sizes.dropna()
                
                if len(sizes) > 0:
                    n_bins = min(12, len(sizes))
                    if n_bins > 0:
                        ax2.hist(sizes, bins=n_bins, alpha=0.7, color=self.colors['chart_2'], 
                                edgecolor='white', density=True)
                        try:
                            from scipy.stats import gaussian_kde
                            density = gaussian_kde(sizes)
                            xs = np.linspace(sizes.min(), sizes.max(), 200)
                            ax2.plot(xs, density(xs), color=self.colors['accent'], linewidth=2)
                        except:
                            ax2.axvline(sizes.mean(), color='yellow', linestyle='--', linewidth=2, alpha=0.7)
                    
                    # Настройка графика
                    ax2.set_title('Распределение размеров', 
                                 color='white', fontsize=14, fontweight='bold', pad=12)
                    ax2.set_xlabel('Ширина (мм)', color='white', fontsize=11)
                    ax2.set_ylabel('Частота', color='white', fontsize=11)
                    ax2.tick_params(colors='white')
                    ax2.grid(True, alpha=0.2)
                    
                    # Статистика на графике
                    stats_text = f"Всего измерений: {len(sizes)}\n"
                    stats_text += f"Среднее: {sizes.mean():.1f} мм\n"
                    stats_text += f"Медиана: {np.median(sizes):.1f} мм\n"
                    stats_text += f"Станд. откл.: {sizes.std():.1f} мм"
                    
                    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
                            fontsize=9, color=self.colors['text_primary'],
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            else:
                ax2.text(0.5, 0.5, 'Нет данных о размерах',
                        ha='center', va='center', color='white')
                ax2.set_title('Распределение размеров', color='white')
            
            #ГРАФИК 3: Количество особей по дням
            ax3 = axes[1, 0]
            if not daily_stats_sorted.empty:
                bars = ax3.bar(daily_stats_sorted['date'], daily_stats_sorted['unique_crayfish_count'],
                             color=self.colors['chart_3'], alpha=0.7, edgecolor='white', width=0.6)
                
                ax3.set_title('Количество особей по дням', 
                             color='white', fontsize=14, fontweight='bold', pad=12)
                ax3.set_xlabel('Дата', color='white', fontsize=11)
                ax3.set_ylabel('Количество особей', color='white', fontsize=11)
                ax3.grid(True, alpha=0.2, axis='y')
                
                # Адаптивный формат дат
                if len(daily_stats_sorted) <= 7:
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                    ax3.xaxis.set_major_locator(mdates.DayLocator())
                elif len(daily_stats_sorted) <= 14:
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                else:
                    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                    interval = max(3, len(daily_stats_sorted) // 8)
                    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
                
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
                
                # Аннотации значений
                if len(daily_stats_sorted) <= 10:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax3.annotate(f'{int(height)}',
                                       xy=(bar.get_x() + bar.get_width() / 2, height),
                                       xytext=(0, 3), textcoords='offset points',
                                       ha='center', va='bottom', fontsize=8, color='white')
                
                # Статистика на графике
                total_crayfish = daily_stats_sorted['unique_crayfish_count'].sum()
                avg_per_day = daily_stats_sorted['unique_crayfish_count'].mean()
                
                stats_text = f"Всего обнаружений: {total_crayfish}\n"
                stats_text += f"Среднее в день: {avg_per_day:.1f}\n"
                stats_text += f"Макс. в день: {daily_stats_sorted['unique_crayfish_count'].max()}"
                
                ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                        fontsize=9, color=self.colors['text_primary'],
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            else:
                ax3.text(0.5, 0.5, 'Нет данных по дням',
                        ha='center', va='center', color='white')
                ax3.set_title('Количество особей по дням', color='white')
            
            #ГРАФИК 4: Сводная статистика
            ax4 = axes[1, 1]
            ax4.axis('off') 
            
            if not daily_stats_sorted.empty:
                info_text = "СВОДНАЯ СТАТИСТИКА\n\n"
                info_text += f"Всего дней с данными: {len(daily_stats_sorted)}\n"
                info_text += f"Всего обнаружений: {daily_stats_sorted['unique_crayfish_count'].sum()}\n"
                info_text += f"Всего измерений: {daily_stats_sorted['measurement_count'].sum()}\n\n"
                
                if 'avg_width' in daily_stats_sorted.columns:
                    avg_size = daily_stats_sorted['avg_width'].mean()
                    min_size = daily_stats_sorted['min_width'].min()
                    max_size = daily_stats_sorted['max_width'].max()
                    
                    info_text += "СТАТИСТИКА РАЗМЕРОВ:\n"
                    info_text += f"• Средний размер: {avg_size:.1f} мм\n"
                    info_text += f"• Минимальный: {min_size:.1f} мм\n"
                    info_text += f"• Максимальный: {max_size:.1f} мм\n\n"
                
                if len(daily_stats_sorted) >= 3:
                    top_days = daily_stats_sorted.nlargest(3, 'unique_crayfish_count')
                    info_text += "САМЫЕ АКТИВНЫЕ ДНИ:\n"
                    for _, row in top_days.iterrows():
                        date_str = row['date'].strftime('%d.%m')
                        count = int(row['unique_crayfish_count'])
                        info_text += f"• {date_str}: {count} особей\n"
                
                ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
                        fontsize=10, color=self.colors['text_primary'],
                        verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=self.colors['card_bg'], 
                                alpha=0.9, edgecolor=self.colors['primary']))
            
            plt.subplots_adjust(hspace=0.3, wspace=0.25)
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            toolbar = NavigationToolbar2Tk(canvas, parent)
            toolbar.update()
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
            export_btn = tk.Button(parent, text="Экспортировать график", 
                                  font=('Arial', 11, 'bold'),
                                  bg=self.colors['primary'], fg='white',
                                  command=lambda: self.export_plot_as_image(fig))
            export_btn.pack(pady=10)
            
        except Exception as e:
            print(f"Ошибка построения графиков: {e}")
            import traceback
            traceback.print_exc()
            tk.Label(parent, text=f"Ошибка построения графиков: {str(e)[:100]}",
                    fg=self.colors['secondary'], bg=self.colors['dark_bg']).pack(pady=50)
    
    #ЭКСПОРТ ГРАФИКА В ФАЙЛ
    def export_plot_as_image(self, fig):
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
    
    #ЗАПУСК КАЛИБРОВКИ КВАДРАТОМ
    def start_square_calibration(self):
        if not self.current_image_path:
            messagebox.showwarning("Внимание", "Сначала загрузите изображение!")
            return
        
        self.calibration_window = tk.Toplevel(self.root)
        self.calibration_window.title("Калибровка квадратом/прямоугольником")
        self.calibration_window.geometry("700x700")
        self.calibration_window.configure(bg=self.colors['dark_bg'])
        
        tk.Label(self.calibration_window, 
                text="КАЛИБРОВКА КВАДРАТОМ/ПРЯМОУГОЛЬНИКОМ",
                font=('Arial', 16, 'bold'),
                fg=self.colors['primary'],
                bg=self.colors['dark_bg']).pack(pady=20)
        
        # Инструкция
        instructions = tk.Label(self.calibration_window,
                              text="1. Выберите объект известного размера на фото\n"
                                   "2. Кликните мышкой по 4 углам объекта (по часовой стрелке)\n"
                                   "3. Введите реальные размеры объекта в мм\n"
                                   "4. Программа рассчитает раздельные коэффициенты X и Y",
                              font=('Arial', 11),
                              fg=self.colors['text_primary'],
                              bg=self.colors['dark_bg'],
                              justify=tk.LEFT)
        instructions.pack(pady=10)
        canvas_frame = tk.Frame(self.calibration_window, bg=self.colors['dark_bg'])
        canvas_frame.pack(pady=10)
        
        self.calibration_canvas = tk.Canvas(canvas_frame, 
                                          bg=self.colors['card_bg'],
                                          highlightthickness=1,
                                          highlightbackground=self.colors['primary'],
                                          width=600, height=400)
        self.calibration_canvas.pack()
        size_frame = tk.Frame(self.calibration_window, bg=self.colors['dark_bg'])
        size_frame.pack(pady=15)
        
        tk.Label(size_frame, text="Ширина объекта (мм):", 
                font=('Arial', 11),
                fg=self.colors['text_primary'],
                bg=self.colors['dark_bg']).pack(side=tk.LEFT, padx=10)
        
        self.calibration_width_var = tk.StringVar(value="10.0")
        width_entry = tk.Entry(size_frame, textvariable=self.calibration_width_var,
                            font=('Arial', 11), width=10)
        width_entry.pack(side=tk.LEFT, padx=10)
        
        tk.Label(size_frame, text="Высота объекта (мм):", 
                font=('Arial', 11),
                fg=self.colors['text_primary'],
                bg=self.colors['dark_bg']).pack(side=tk.LEFT, padx=10)
        
        self.calibration_height_var = tk.StringVar(value="10.0")
        height_entry = tk.Entry(size_frame, textvariable=self.calibration_height_var,
                              font=('Arial', 11), width=10)
        height_entry.pack(side=tk.LEFT, padx=10)
        info_frame = tk.Frame(self.calibration_window, bg=self.colors['card_bg'])
        info_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.calibration_info_label = tk.Label(info_frame,
                                             text="Жду выбора 4 точек... (0/4)",
                                             font=('Arial', 10),
                                             fg=self.colors['accent'],
                                             bg=self.colors['card_bg'])
        self.calibration_info_label.pack(pady=5)
        button_frame = tk.Frame(self.calibration_window, bg=self.colors['dark_bg'])
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="Завершить калибровку", 
                 font=('Arial', 11, 'bold'),
                 bg=self.colors['accent'], fg='white',
                 command=self.finish_square_calibration).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Очистить точки", 
                 font=('Arial', 11),
                 bg=self.colors['warning'], fg='white',
                 command=self.clear_calibration_points).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Отмена", 
                 font=('Arial', 11),
                 bg=self.colors['secondary'], fg='white',
                 command=self.cancel_calibration).pack(side=tk.LEFT, padx=10)
        
        self.calibration_mode = True
        self.calibration_points = []
        self.calibration_shape = None
        self.calibration_indicator.config(fg=self.colors['accent'])
    
        self.load_square_calibration_image()
        
        self.calibration_canvas.bind("<Button-1>", self.on_square_calibration_click)
        self.calibration_photo_ref = self.calibration_photo
    
    #ЗАГРУЗКА ИЗОБРАЖЕНИЯ ДЛЯ КАЛИБРОВКИ
    def load_square_calibration_image(self):
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
        if not self.calibration_mode:
            return
        
        x, y = event.x, event.y
        
        if len(self.calibration_points) >= 4:
            self.calibration_info_label.config(
                text=f"Уже выбрано 4 точки. Нажмите 'Очистить точки' чтобы начать заново.",
                fg=self.colors['warning']
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
            fg=self.colors['accent']
        )
        
        if not hasattr(self, 'calibration_objects'):
            self.calibration_objects = []
        self.calibration_objects.extend([point_id, text_id])
        
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
            self.detect_calibration_shape()
            self.show_calibration_preview()
    
    #ОПРЕДЕЛЕНИЕ ФОРМЫ КАЛИБРОВОЧНОГО ОБЪЕКТА
    def detect_calibration_shape(self):
        if len(self.calibration_points) != 4:
            return
        distances = []
        for i in range(4):
            x1, y1 = self.calibration_points[i]
            x2, y2 = self.calibration_points[(i + 1) % 4]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)
        
        avg_distance = sum(distances) / 4
        max_diff = max(abs(d - avg_distance) for d in distances)
        
        if max_diff < avg_distance * 0.2:
            angles = self.calculate_angles()
            if all(80 < angle < 100 for angle in angles):
                self.calibration_shape = 'square'
                shape_text = "КВАДРАТ"
            else:
                self.calibration_shape = 'rhombus'
                shape_text = "РОМБ"
        else:
            self.calibration_shape = 'rectangle'
            shape_text = "ПРЯМОУГОЛЬНИК"
        
        self.calibration_info_label.config(
            text=f"Фигура определена как: {shape_text}",
            fg=self.colors['chart_3']
        )
    
    #РАСЧЕТ УГЛОВ КАЛИБРОВОЧНОГО ОБЪЕКТА
    def calculate_angles(self):
        angles = []
        for i in range(4):
            p1 = np.array(self.calibration_points[i])
            p2 = np.array(self.calibration_points[(i + 1) % 4])
            p3 = np.array(self.calibration_points[(i + 2) % 4])
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
            angles.append(angle)
        
        return angle
    def show_calibration_preview(self):
        if len(self.calibration_points) != 4:
            return
        horizontal_sides = [
            math.sqrt((self.calibration_points[1][0] - self.calibration_points[0][0])**2 +
                     (self.calibration_points[1][1] - self.calibration_points[0][1])**2),
            math.sqrt((self.calibration_points[3][0] - self.calibration_points[2][0])**2 +
                     (self.calibration_points[3][1] - self.calibration_points[2][1])**2)
        ]
        
        vertical_sides = [
            math.sqrt((self.calibration_points[2][0] - self.calibration_points[1][0])**2 +
                     (self.calibration_points[2][1] - self.calibration_points[1][1])**2),
            math.sqrt((self.calibration_points[0][0] - self.calibration_points[3][0])**2 +
                     (self.calibration_points[0][1] - self.calibration_points[3][1])**2)
        ]
        
        avg_width_px = sum(horizontal_sides) / 2
        avg_height_px = sum(vertical_sides) / 2
        
        actual_width_px = avg_width_px / self.calibration_scale_factor
        actual_height_px = avg_height_px / self.calibration_scale_factor
        
        try:
            width_mm = float(self.calibration_width_var.get())
            height_mm = float(self.calibration_height_var.get())
            factor_x = width_mm / actual_width_px
            factor_y = height_mm / actual_height_px
            
            info_text = f"Предварительные коэффициенты:\n"
            info_text += f"• По ширине: 1px = {factor_x:.4f} мм\n"
            info_text += f"• По высоте: 1px = {factor_y:.4f} мм\n"
            info_text += f"• Измерено: {actual_width_px:.1f}x{actual_height_px:.1f} px"
            
            self.calibration_info_label.config(text=info_text)
            
        except ValueError:
            self.calibration_info_label.config(
                text="Введите корректные размеры для расчета",
                fg=self.colors['warning']
            )
    
    #ОЧИСТКА ТОЧЕК КАЛИБРОВКИ
    def clear_calibration_points(self):
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
            fg=self.colors['accent']
        )
        
        self.calibration_canvas.delete("all")
        self.calibration_canvas.create_image(0, 0, anchor=tk.NW, image=self.calibration_photo)
    
    #ЗАВЕРШЕНИЕ КАЛИБРОВКИ
    def finish_square_calibration(self):
        if len(self.calibration_points) != 4:
            messagebox.showwarning("Внимание", "Выберите 4 точки для калибровки!")
            return
        
        try:
            known_width_mm = float(self.calibration_width_var.get())
            known_height_mm = float(self.calibration_height_var.get())
            
            if known_width_mm <= 0 or known_height_mm <= 0:
                messagebox.showwarning("Внимание", "Размеры должны быть больше 0!")
                return
            
            horizontal_sides = [
                math.sqrt((self.calibration_points[1][0] - self.calibration_points[0][0])**2 +
                         (self.calibration_points[1][1] - self.calibration_points[0][1])**2),
                math.sqrt((self.calibration_points[3][0] - self.calibration_points[2][0])**2 +
                         (self.calibration_points[3][1] - self.calibration_points[2][1])**2)
            ]
            
            vertical_sides = [
                math.sqrt((self.calibration_points[2][0] - self.calibration_points[1][0])**2 +
                         (self.calibration_points[2][1] - self.calibration_points[1][1])**2),
                math.sqrt((self.calibration_points[0][0] - self.calibration_points[3][0])**2 +
                         (self.calibration_points[0][1] - self.calibration_points[3][1])**2)
            ]
            
            avg_width_px = sum(horizontal_sides) / 2
            avg_height_px = sum(vertical_sides) / 2
            
            actual_width_px = avg_width_px / self.calibration_scale_factor
            actual_height_px = avg_height_px / self.calibration_scale_factor
            
            self.calibration_factor_x = known_width_mm / actual_width_px
            self.calibration_factor_y = known_height_mm / actual_height_px
            
            self.calibration_label_x.config(text=f"X: 1 пиксель = {self.calibration_factor_x:.4f} мм")
            self.calibration_label_y.config(text=f"Y: 1 пиксель = {self.calibration_factor_y:.4f} мм")
            self.calibration_indicator.config(fg=self.colors['accent'])
            
            self.calibration_window.destroy()
            self.calibration_mode = False
            
            success_text = f"Калибровка завершена!\n\n"
            success_text += f"Результаты:\n"
            success_text += f"• По ширине: 1px = {self.calibration_factor_x:.4f} мм\n"
            success_text += f"• По высоте: 1px = {self.calibration_factor_y:.4f} мм\n\n"
            success_text += f"Измеренные размеры:\n"
            success_text += f"• В пикселях: {actual_width_px:.1f} x {actual_height_px:.1f} px\n"
            success_text += f"• В миллиметрах: {known_width_mm:.1f} x {known_height_mm:.1f} мм\n\n"
            success_text += f"Форма: {self.calibration_shape.upper() if self.calibration_shape else 'НЕ ОПРЕДЕЛЕНА'}"
            
            messagebox.showinfo("Успех", success_text)
            
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числа для размеров!")
    
    # ОТМЕНА КАЛИБРОВКИ
    def cancel_calibration(self):
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_shape = None
        if hasattr(self, 'calibration_window'):
            self.calibration_window.destroy()
        if hasattr(self, 'calibration_objects'):
            del self.calibration_objects
        messagebox.showinfo("Информация", "Калибровка отменена")
    
    #СБРОС КАЛИБРОВКИ К СТАНДАРТНЫМ ЗНАЧЕНИЯМ
    def reset_calibration(self):
        self.calibration_factor_x = 0.15
        self.calibration_factor_y = 0.15
        self.calibration_label_x.config(text=f"X: 1 пиксель = {self.calibration_factor_x:.4f} мм")
        self.calibration_label_y.config(text=f"Y: 1 пиксель = {self.calibration_factor_y:.4f} мм")
        messagebox.showinfo("Информация", f"Калибровка сброшена к стандартным значениям\n"
                                        f"По ширине: 1px = {self.calibration_factor_x:.4f} мм\n"
                                        f"По высоте: 1px = {self.calibration_factor_y:.4f} мм")
    
    #СТАТИСТИКА БАЗЫ ДАННЫХ
    def show_database_stats(self):
        try:
            status = self.db.check_database_status()
            
            if not status:
                messagebox.showinfo("Статистика БД", "Не удалось получить статистику базы данных")
                return
            
            daily_stats = self.db.get_daily_statistics()
            available_dates = self.db.get_available_dates()
            
            # Формируем текст статистики
            stats_text = f"СТАТИСТИКА БАЗЫ ДАННЫХ\n\n"
            stats_text += f"Файл БД: {status['database_file']}\n"
            stats_text += f"Размер файла: {status['file_size'] / 1024:.1f} KB\n\n"
            stats_text += f"Всего особей: {status['crayfish_count']}\n"
            stats_text += f"Всего измерений: {status['measurements_count']}\n"
            stats_text += f"Всего сессий: {status['sessions_count']}\n"
            stats_text += f"ML запросов: {status['ml_logs_count']}\n\n"
            
            if not daily_stats.empty:
                stats_text += f"Дней с данными: {len(daily_stats)}\n"
                stats_text += f"Первая запись: {daily_stats['date'].min().strftime('%d.%m.%Y')}\n"
                stats_text += f"Последняя запись: {daily_stats['date'].max().strftime('%d.%m.%Y')}\n\n"
            
            if available_dates:
                stats_text += f"Доступные даты ({len(available_dates)}):\n"
                for i, date_str in enumerate(available_dates[:5]):
                    stats_text += f"  • {date_str}\n"
                if len(available_dates) > 5:
                    stats_text += f"  ... и еще {len(available_dates) - 5} дней\n"
            
            messagebox.showinfo("Статистика БД", stats_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка получения статистики: {e}")
    
    #ЭКСПОРТ ВСЕХ ДАННЫХ
    def export_all_data(self):
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
                if file_path.endswith('.xlsx'):
                    try:
                        import openpyxl
                        from openpyxl import Workbook
                    except ImportError:
                        messagebox.showwarning("Экспорт")
                        csv_path = file_path.replace('.xlsx', '.csv')
                        export_df.to_csv(csv_path, index=False, encoding='utf-8')
                        messagebox.showinfo("Успех", f"Данные экспортированы в CSV:\n{csv_path}")
                        return
                    
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name='Данные_раков', index=False)
                        
                    messagebox.showinfo("Успех", 
                                      f"Данные экспортированы в Excel:\n{file_path}\n\n"
                                      f"Экспортировано:\n"
                                      f"• Строк: {len(export_df)}\n"
                                      f"• Колонки: Номер рака, Ширина (мм), Высота (мм), Угол поворота, Дата, Время, Уверенность")
                    
                else:
                    # Экспорт в CSV
                    export_df.to_csv(file_path, index=False, encoding='utf-8')
                    
                    messagebox.showinfo("Успех", 
                                      f"Данные экспортированы в CSV:\n{file_path}\n\n"
                                      f"Экспортировано:\n"
                                      f"• Строк: {len(export_df)}\n"
                                      f"• Колонки: Номер рака, Ширина (мм), Высота (мм), Угол поворота, Дата, Время, Уверенность")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    #ВЫБОР ДАТЫ ДЛЯ ЭКСПОРТА
    def export_by_date(self):
        try:
            available_dates = self.db.get_available_dates()
            
            if not available_dates:
                messagebox.showwarning("Экспорт", "В базе данных нет данных для экспорта")
                return
            
            date_window = tk.Toplevel(self.root)
            date_window.title("Выбор даты для экспорта")
            date_window.geometry("400x300")
            date_window.configure(bg=self.colors['dark_bg'])
            
            tk.Label(date_window, 
                    text="ВЫБЕРИТЕ ДАТУ ДЛЯ ЭКСПОРТА",
                    font=('Arial', 14, 'bold'),
                    fg=self.colors['primary'],
                    bg=self.colors['dark_bg']).pack(pady=20)
            
            # Список доступных дат
            listbox_frame = tk.Frame(date_window, bg=self.colors['card_bg'])
            listbox_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
            
            listbox = tk.Listbox(listbox_frame, bg='#1e2b3a', fg='white',
                               font=('Arial', 11), selectbackground=self.colors['primary'])
            
            scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
            listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=listbox.yview)
            
            for date_str in available_dates:
                listbox.insert(tk.END, date_str)
            
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            button_frame = tk.Frame(date_window, bg=self.colors['dark_bg'])
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
                     bg=self.colors['accent'], fg='white',
                     command=export_selected_date).pack(side=tk.LEFT, padx=10)
            
            tk.Button(button_frame, text="Отмена", 
                     font=('Arial', 11),
                     bg=self.colors['secondary'], fg='white',
                     command=date_window.destroy).pack(side=tk.LEFT, padx=10)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выборе даты: {e}")
    
    #ВЫПОЛНЕНИЕ ЭКСПОРТА ПО ДАТЕ
    def perform_export_by_date(self, selected_date):
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
                if file_path.endswith('.xlsx'):
                    try:
                        import openpyxl
                        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                            export_df.to_excel(writer, sheet_name=f'Данные_{selected_date}', index=False)
                    except ImportError:
                        csv_path = file_path.replace('.xlsx', '.csv')
                        export_df.to_csv(csv_path, index=False, encoding='utf-8')
                        file_path = csv_path
                else:
                    export_df.to_csv(file_path, index=False, encoding='utf-8')
                
                messagebox.showinfo("Успех", 
                                  f"Данные за {selected_date} экспортированы в:\n{file_path}\n\n"
                                  f"Статистика экспорта:\n"
                                  f"• Строк: {len(export_df)}\n"
                                  f"• Уникальных особей: {export_df['Номер рака'].nunique() if 'Номер рака' in export_df.columns else 0}\n"
                                  f"• Средняя ширина: {export_df['Ширина (мм)'].mean():.1f} мм\n"
                                  f"• Средняя высота: {export_df['Высота (мм)'].mean():.1f} мм")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    #ОЧИСТКА БАЗЫ ДАННЫХ
    def clear_database_ui(self):
        """
        Очищает всю базу данных после подтверждения пользователя
        """
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
    
    #НАСТРОЙКА TELEGRAM БОТА
    def setup_telegram_bot(self):

        dialog = tk.Toplevel(self.root)
        dialog.title("Настройка Telegram Bot")
        dialog.geometry("500x400")
        dialog.configure(bg=self.colors['dark_bg'])
        
        tk.Label(dialog, text="НАСТРОЙКА TELEGRAM BOT", 
                font=('Arial', 16, 'bold'),
                fg=self.colors['primary'],
                bg=self.colors['dark_bg']).pack(pady=20)
        
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
        
        # Поле ввода токена
        token_frame = tk.Frame(dialog, bg=self.colors['dark_bg'])
        token_frame.pack(pady=10)
        
        tk.Label(token_frame, text="Токен бота:", 
                font=('Arial', 11),
                fg=self.colors['text_primary'],
                bg=self.colors['dark_bg']).pack(side=tk.LEFT)
        
        token_entry = tk.Entry(token_frame, textvariable=self.bot_token_var,
                             font=('Arial', 10), width=40)
        token_entry.pack(side=tk.LEFT, padx=10)
        
        # Кнопки
        btn_frame = tk.Frame(dialog, bg=self.colors['dark_bg'])
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
                 bg=self.colors['accent'], fg='white',
                 command=save_and_start).pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_frame, text="Отмена", 
                 font=('Arial', 11),
                 bg=self.colors['secondary'], fg='white',
                 command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    def start_telegram_bot(self):
        """
        Запускает Telegram бота в отдельном потоке
        """
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
            
            self.bot_status_label.config(text="Бот запущен", fg=self.colors['accent'])
            self.bot_indicator.config(fg=self.colors['accent'])
            self.status_label.config(text="Telegram Bot запущен")
            
            messagebox.showinfo("Успех", "Telegram Bot успешно запущен!\n\nБот доступен по командам:\n/detect - анализ фото\n/stats - статистика\n/charts - графики\n/export - экспорт данных\n/status - статус системы\n/info - информация о боте")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Ошибка запуска бота: {error_msg}")
            messagebox.showerror("Ошибка", f"Не удалось запустить бота: {error_msg}")
            self.bot_status_label.config(text="Ошибка запуска", fg=self.colors['secondary'])
    
    def stop_telegram_bot(self):
        if self.telegram_bot:
            self.telegram_bot.stop()
            self.telegram_bot = None
            self.bot_thread = None
            
            self.bot_status_label.config(text="Бот остановлен", fg=self.colors['secondary'])
            self.bot_indicator.config(fg='#ff4444')
            self.status_label.config(text="Telegram Bot остановлен")
            
            messagebox.showinfo("телеграмм бот остановлен")
        else:
            messagebox.showinfo("бот не работает")


#ТОЧКА ВХОДА В ПРИЛОЖЕНИЕ
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernCrayfishDetector(root)
    
    root.mainloop()
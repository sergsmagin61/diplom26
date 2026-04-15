"""
database.py - Модуль управления базой данных SQLite
Содержит класс DatabaseManager для работы с БД: создание таблиц, сохранение данных, экспорт
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
import os
import numpy as np


class DatabaseManager:
    def __init__(self, db_name="crayfish_data.db"):
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        self.ensure_tables_exist()
    
    def get_connection(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
        return self.connection, self.cursor
    
    def close_connection(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
    
    def ensure_tables_exist(self):
        conn, cursor = self.get_connection()
        try:
            # Таблица особей раков
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS crayfish (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    unique_id TEXT UNIQUE,
                    first_detected TIMESTAMP,
                    last_detected TIMESTAMP,
                    total_measurements INTEGER DEFAULT 0,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица измерений
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crayfish_id INTEGER,
                    timestamp TIMESTAMP,
                    image_path TEXT,
                    width_mm REAL,
                    height_mm REAL,
                    width_px REAL,
                    height_px REAL,
                    angle REAL DEFAULT 0,
                    confidence REAL,
                    bounding_box TEXT,
                    session_id INTEGER,
                    FOREIGN KEY (crayfish_id) REFERENCES crayfish (id)
                )
            ''')
            cursor.execute("PRAGMA table_info(measurements)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'angle' not in columns:
                print("Добавляем колонку angle в существующую таблицу measurements...")
                cursor.execute("ALTER TABLE measurements ADD COLUMN angle REAL DEFAULT 0")
                conn.commit()
                print("Колонка angle успешно добавлена")
            
            # Таблица сессий анализа
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    image_path TEXT,
                    total_detections INTEGER,
                    processing_time REAL,
                    confidence_threshold REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица логов ML инференса
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    image_path TEXT,
                    image_width INTEGER,
                    image_height INTEGER,
                    inference_time_ms REAL,
                    detections_count INTEGER,
                    avg_confidence REAL,
                    model_name TEXT,
                    confidence_threshold REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            print("Таблицы БД проверены/созданы")
        except Exception as e:
            print(f"Ошибка создания таблиц: {e}")
        finally:
            self.close_connection()
    
    def save_crayfish(self, unique_id, first_size=None):
        """
        Сохранение информации о раке в базу данных
        """
        conn, cursor = self.get_connection()
        try:
            if first_size is not None:
                if hasattr(first_size, 'item'):
                    first_size = float(first_size.item())
                else:
                    first_size = float(first_size)
            
            # Проверка существования рака
            cursor.execute("SELECT id FROM crayfish WHERE unique_id = ?", (unique_id,))
            result = cursor.fetchone()
            
            if result:
                crayfish_id = result[0]
                print(f"Найден существующий рак ID: {crayfish_id}")
            else:
                metadata = {'first_size': first_size} if first_size is not None else {}
                cursor.execute('''
                    INSERT INTO crayfish (unique_id, first_detected, last_detected, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (unique_id, datetime.now(), datetime.now(), json.dumps(metadata)))
                crayfish_id = cursor.lastrowid
                print(f"Создан новый рак ID: {crayfish_id}")
            
            # Обновление информации о последнем обнаружении
            cursor.execute('''
                UPDATE crayfish 
                SET last_detected = ?, total_measurements = total_measurements + 1
                WHERE id = ?
            ''', (datetime.now(), crayfish_id))
            conn.commit()
            return crayfish_id
        except Exception as e:
            print(f"Ошибка сохранения рака: {e}")
            conn.rollback()
            return None
        finally:
            self.close_connection()
    
    def save_measurement(self, crayfish_id, measurement_data, session_id):
        """
        Сохранение измерения рака в базу данных
        """
        conn, cursor = self.get_connection()
        try:
            # Преобразование данных в нужные типы
            width_mm = float(measurement_data['width_mm'])
            height_mm = float(measurement_data['height_mm'])
            width_px = float(measurement_data['width_px'])
            height_px = float(measurement_data['height_px'])
            angle = float(measurement_data.get('angle', 0))
            confidence = float(measurement_data['confidence'])
            bounding_box = measurement_data['bounding_box']
            
            if not isinstance(bounding_box, str):
                bounding_box = json.dumps(bounding_box)
            
            # Вставка записи измерения
            cursor.execute('''
                INSERT INTO measurements 
                (crayfish_id, timestamp, image_path, width_mm, height_mm, width_px, height_px, angle, confidence, bounding_box, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                crayfish_id,
                measurement_data['timestamp'],
                measurement_data['image_path'],
                width_mm,
                height_mm,
                width_px,
                height_px,
                angle,
                confidence,
                bounding_box,
                session_id
            ))
            
            measurement_id = cursor.lastrowid
            conn.commit()
            print(f"Измерение сохранено ID: {measurement_id} для рака {crayfish_id}, угол: {angle:.1f}°")
            return measurement_id
            
        except Exception as e:
            print(f"Ошибка сохранения измерения: {e}")
            conn.rollback()
            return None
        finally:
            self.close_connection()
    
    def save_session(self, session_data):
        """
        Сохранение информации о сессии анализа
        """
        conn, cursor = self.get_connection()
        
        try:
            timestamp = session_data['timestamp']
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            
            cursor.execute('''
                INSERT INTO sessions (timestamp, image_path, total_detections, processing_time, confidence_threshold)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                timestamp,
                session_data['image_path'],
                session_data['total_detections'],
                float(session_data['processing_time']),
                float(session_data['confidence_threshold'])
            ))
            
            session_id = cursor.lastrowid
            conn.commit()
            print(f"Сессия сохранена ID: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"Ошибка сохранения сессии: {e}")
            conn.rollback()
            return None
        finally:
            self.close_connection()
    
    def save_ml_inference_log(self, inference_data):
        """
        Сохранение лога ML инференса
        """
        conn, cursor = self.get_connection()
        
        try:
            cursor.execute('''
                INSERT INTO ml_inference_logs 
                (timestamp, image_path, image_width, image_height, inference_time_ms, 
                detections_count, avg_confidence, model_name, confidence_threshold)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                inference_data['timestamp'],
                inference_data.get('image_path', ''),
                inference_data.get('image_width', 0),
                inference_data.get('image_height', 0),
                float(inference_data['inference_time_ms']),
                int(inference_data['detections_count']),
                float(inference_data.get('avg_confidence', 0)),
                inference_data.get('model_name', 'unknown'),
                float(inference_data.get('confidence_threshold', 0.5))
            ))
            
            log_id = cursor.lastrowid
            conn.commit()
            print(f"Лог ML inference сохранен ID: {log_id}")
            return log_id
            
        except Exception as e:
            print(f"Ошибка сохранения лога ML inference: {e}")
            conn.rollback()
            return None
        finally:
            self.close_connection()
    
    def get_all_measurements(self):
        """
        Получение всех измерений из базы данных
        
        Returns:
            pandas.DataFrame: DataFrame со всеми измерениями
        """
        conn, cursor = self.get_connection()
        
        try:
            query = '''
                SELECT m.*, c.unique_id 
                FROM measurements m
                LEFT JOIN crayfish c ON m.crayfish_id = c.id
                ORDER BY m.timestamp
            '''
            df = pd.read_sql_query(query, conn)
            print(f"Загружено {len(df)} измерений из БД")
            
            # Преобразование числовых колонок
            numeric_columns = ['width_mm', 'height_mm', 'width_px', 'height_px', 'angle', 'confidence']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
        except Exception as e:
            print(f"Ошибка получения измерений: {e}")
            return pd.DataFrame()
        finally:
            self.close_connection()
    
    def get_crayfish_export_data(self):
        """
        Получение данных для экспорта всех раков
        """
        conn, cursor = self.get_connection()
        
        try:
            query = '''
                SELECT 
                    c.unique_id as 'Номер рака',
                    m.width_mm as 'Ширина (мм)',
                    m.height_mm as 'Высота (мм)',
                    m.angle as 'Угол поворота',
                    DATE(m.timestamp) as 'Дата',
                    TIME(m.timestamp) as 'Время',
                    m.confidence as 'Уверенность'
                FROM measurements m
                LEFT JOIN crayfish c ON m.crayfish_id = c.id
                ORDER BY m.timestamp DESC
            '''
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                numeric_columns = ['Ширина (мм)', 'Высота (мм)', 'Угол поворота', 'Уверенность']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Округление значений
                df['Ширина (мм)'] = df['Ширина (мм)'].round(2)
                df['Высота (мм)'] = df['Высота (мм)'].round(2)
                df['Угол поворота'] = df['Угол поворота'].round(1)
                df['Уверенность'] = df['Уверенность'].round(3)
            
            print(f"Загружено {len(df)} строк для экспорта")
            return df
        except Exception as e:
            print(f"Ошибка получения данных для экспорта: {e}")
            return pd.DataFrame()
        finally:
            self.close_connection()
    
    def get_crayfish_export_by_date(self, date_str):
        """
        Получение данных для экспорта за конкретную дату
        """
        conn, cursor = self.get_connection()
        
        try:
            query = '''
                SELECT 
                    c.unique_id as 'Номер рака',
                    m.width_mm as 'Ширина (мм)',
                    m.height_mm as 'Высота (мм)',
                    m.angle as 'Угол поворота',
                    DATE(m.timestamp) as 'Дата',
                    TIME(m.timestamp) as 'Время',
                    m.confidence as 'Уверенность'
                FROM measurements m
                LEFT JOIN crayfish c ON m.crayfish_id = c.id
                WHERE DATE(m.timestamp) = DATE(?)
                ORDER BY m.timestamp DESC
            '''
            df = pd.read_sql_query(query, conn, params=(date_str,))
            
            if not df.empty:
                numeric_columns = ['Ширина (мм)', 'Высота (мм)', 'Угол поворота', 'Уверенность']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['Ширина (мм)'] = df['Ширина (мм)'].round(2)
                df['Высота (мм)'] = df['Высота (мм)'].round(2)
                df['Угол поворота'] = df['Угол поворота'].round(1)
                df['Уверенность'] = df['Уверенность'].round(3)
            
            print(f"Загружено {len(df)} строк для экспорта за дату {date_str}")
            return df
        except Exception as e:
            print(f"Ошибка получения данных по дате для экспорта: {e}")
            return pd.DataFrame()
        finally:
            self.close_connection()
    
    def get_all_crayfish(self):
        """
        Получение информации о всех раках
        """
        conn, cursor = self.get_connection()
        
        try:
            query = '''
                SELECT c.*, 
                       (SELECT COUNT(*) FROM measurements WHERE crayfish_id = c.id) as measurement_count,
                       (SELECT MIN(timestamp) FROM measurements WHERE crayfish_id = c.id) as first_measurement,
                       (SELECT MAX(timestamp) FROM measurements WHERE crayfish_id = c.id) as last_measurement,
                       (SELECT AVG(width_mm) FROM measurements WHERE crayfish_id = c.id) as avg_width
                FROM crayfish c
                ORDER BY c.first_detected
            '''
            df = pd.read_sql_query(query, conn)
            print(f"Загружено {len(df)} раков из БД")
            return df
        except Exception as e:
            print(f"Ошибка получения раков: {e}")
            return pd.DataFrame()
        finally:
            self.close_connection()
    
    def get_available_dates(self):
        """
        Получение списка всех дат, за которые есть измерения
        """
        conn, cursor = self.get_connection()
        
        try:
            cursor.execute('''
                SELECT DISTINCT DATE(timestamp) as date
                FROM measurements
                WHERE timestamp IS NOT NULL
                ORDER BY date DESC
            ''')
            dates = [row[0] for row in cursor.fetchall()]
            return dates
        except Exception as e:
            print(f"Ошибка получения дат: {e}")
            return []
        finally:
            self.close_connection()
    
    def get_daily_statistics(self):
        """
        Получение дневной статистики измерений
        """
        conn, cursor = self.get_connection()
        
        try:
            query = '''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as measurement_count,
                    COUNT(DISTINCT crayfish_id) as unique_crayfish_count,
                    AVG(width_mm) as avg_width,
                    AVG(height_mm) as avg_height,
                    MIN(width_mm) as min_width,
                    MAX(width_mm) as max_width,
                    MIN(timestamp) as first_measurement_time,
                    MAX(timestamp) as last_measurement_time
                FROM measurements
                WHERE timestamp IS NOT NULL
                GROUP BY DATE(timestamp)
                ORDER BY date
            '''
            cursor.execute(query)
            rows = cursor.fetchall()
            
            columns = ['date', 'measurement_count', 'unique_crayfish_count', 
                     'avg_width', 'avg_height', 'min_width', 'max_width',
                     'first_measurement_time', 'last_measurement_time']
            
            daily_stats = pd.DataFrame(rows, columns=columns)
            
            if not daily_stats.empty:
                daily_stats['date'] = pd.to_datetime(daily_stats['date'])
                daily_stats['avg_width'] = pd.to_numeric(daily_stats['avg_width'], errors='coerce').round(2)
                daily_stats['avg_height'] = pd.to_numeric(daily_stats['avg_height'], errors='coerce').round(2)
                daily_stats['min_width'] = pd.to_numeric(daily_stats['min_width'], errors='coerce').round(2)
                daily_stats['max_width'] = pd.to_numeric(daily_stats['max_width'], errors='coerce').round(2)
            
            return daily_stats
            
        except Exception as e:
            print(f"Ошибка получения дневной статистики: {e}")
            return pd.DataFrame()
        finally:
            self.close_connection()
    
    def get_ml_inference_stats(self):
        """
        Получение статистики ML инференсов
        """
        conn, cursor = self.get_connection()
        
        try:
            query = '''
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_inferences,
                    AVG(inference_time_ms) as avg_inference_time,
                    MIN(inference_time_ms) as min_inference_time,
                    MAX(inference_time_ms) as max_inference_time,
                    AVG(detections_count) as avg_detections,
                    AVG(avg_confidence) as avg_confidence
                FROM ml_inference_logs
                WHERE timestamp IS NOT NULL
                GROUP BY DATE(timestamp)
                ORDER BY date
            '''
            cursor.execute(query)
            rows = cursor.fetchall()
            
            columns = ['date', 'total_inferences', 'avg_inference_time', 
                      'min_inference_time', 'max_inference_time', 
                      'avg_detections', 'avg_confidence']
            
            ml_stats = pd.DataFrame(rows, columns=columns)
            
            if not ml_stats.empty:
                ml_stats['date'] = pd.to_datetime(ml_stats['date'])
                for col in ['avg_inference_time', 'min_inference_time', 'max_inference_time',
                          'avg_detections', 'avg_confidence']:
                    if col in ml_stats.columns:
                        ml_stats[col] = pd.to_numeric(ml_stats[col], errors='coerce').round(2)
            
            return ml_stats
            
        except Exception as e:
            print(f"Ошибка получения статистики ML inference: {e}")
            return pd.DataFrame()
        finally:
            self.close_connection()
    
    def clear_database(self):
        """
        Полная очистка всех таблиц базы данных
        """
        conn, cursor = self.get_connection()
        
        try:
            # Удаление данных из всех таблиц
            cursor.execute("DELETE FROM measurements")
            cursor.execute("DELETE FROM crayfish")
            cursor.execute("DELETE FROM sessions")
            cursor.execute("DELETE FROM ml_inference_logs")
            
            # Сброс счетчиков автоинкремента
            cursor.execute("UPDATE SQLITE_SEQUENCE SET seq = 0 WHERE name = 'measurements'")
            cursor.execute("UPDATE SQLITE_SEQUENCE SET seq = 0 WHERE name = 'crayfish'")
            cursor.execute("UPDATE SQLITE_SEQUENCE SET seq = 0 WHERE name = 'sessions'")
            cursor.execute("UPDATE SQLITE_SEQUENCE SET seq = 0 WHERE name = 'ml_inference_logs'")
            
            conn.commit()
            print("База данных очищена")
            return True
        except Exception as e:
            print(f"Ошибка очистки БД: {e}")
            return False
        finally:
            self.close_connection()
    
    def check_database_status(self):
        """
        Проверка текущего статуса базы данных
        """
        conn, cursor = self.get_connection()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM crayfish")
            crayfish_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM measurements")
            measurements_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            sessions_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ml_inference_logs")
            ml_logs_count = cursor.fetchone()[0]
            
            status = {
                'crayfish_count': crayfish_count,
                'measurements_count': measurements_count,
                'sessions_count': sessions_count,
                'ml_logs_count': ml_logs_count,
                'database_file': os.path.abspath(self.db_name),
                'file_exists': os.path.exists(self.db_name),
                'file_size': os.path.getsize(self.db_name) if os.path.exists(self.db_name) else 0
            }
            
            return status
        except Exception as e:
            print(f"Ошибка проверки статуса БД: {e}")
            return {}
        finally:
            self.close_connection()
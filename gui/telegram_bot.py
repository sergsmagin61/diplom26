# telegram_bot.py - Модуль Telegram бота для удаленного управления

import os
import time
import cv2
import numpy as np
from datetime import datetime
import telebot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class TelegramBotManager:
    # Класс для управления Telegram ботом
    
    def __init__(self, detector_app, token=None):
        # Инициализация менеджера Telegram бота
        self.detector = detector_app
        self.bot = None
        self.bot_thread = None
        self.running = False
        self.last_update_id = None
        
        if token:
            self.initialize_bot(token)
    
    # Инициализация бота с указанным токеном
    def initialize_bot(self, token):
        try:
            self.bot = telebot.TeleBot(token, threaded=False)
            print(f"Telegram Bot инициализирован с токеном: {token[:10]}...")
            return True
        except Exception as e:
            print(f"Ошибка инициализации бота: {e}")
            return False
    
    # Настройка обработчиков команд бота
    def setup_handlers(self):
        
        # Обработчик команды /start и /help
        @self.bot.message_handler(commands=['start', 'help'])
        def send_welcome(message):
            welcome_text = """
Crayfish AI Studio Bot

Доступные команды:

/detect - Анализ фото раков
/stats - Статистика базы данных
/charts - Графики роста популяции
/export - Экспорт данных в CSV
/status - Статус системы
/info - Информация о боте

Для анализа просто отправьте фото с раками
            """
            try:
                self.bot.reply_to(message, welcome_text)
            except Exception as e:
                print(f"Ошибка отправки welcome: {e}")
        
        # Обработчик команды /detect
        @self.bot.message_handler(commands=['detect'])
        def handle_detect(message):
            try:
                self.bot.reply_to(message, "Отправьте фото раков для анализа")
            except Exception as e:
                print(f"Ошибка команды detect: {e}")
        
        # Обработчик получения фото
        @self.bot.message_handler(content_types=['photo'])
        def handle_photo(message):
            try:
                self.process_photo(message)
            except Exception as e:
                print(f"Ошибка обработки фото: {e}")
                try:
                    self.bot.reply_to(message, f"Ошибка обработки фото: {str(e)[:100]}")
                except:
                    pass
        
        # Обработчик команды /stats
        @self.bot.message_handler(commands=['stats'])
        def handle_stats(message):
            try:
                self.send_statistics(message)
            except Exception as e:
                print(f"Ошибка команды stats: {e}")
        
        # Обработчик команды /charts
        @self.bot.message_handler(commands=['charts'])
        def handle_charts(message):
            try:
                self.send_charts(message)
            except Exception as e:
                print(f"Ошибка команды charts: {e}")
        
        # Обработчик команды /export
        @self.bot.message_handler(commands=['export'])
        def handle_export(message):
            try:
                self.send_export(message)
            except Exception as e:
                print(f"Ошибка команды export: {e}")
        
        # Обработчик команды /status
        @self.bot.message_handler(commands=['status'])
        def handle_status(message):
            try:
                self.send_status(message)
            except Exception as e:
                print(f"Ошибка команды status: {e}")
        
        # Обработчик команды /info
        @self.bot.message_handler(commands=['info'])
        def handle_info(message):
            try:
                info_text = "Информация о боте\n\n"
                info_text += "Crayfish AI Studio Bot\n"
                info_text += "Версия: 2.0\n"
                info_text += "Функции: Детекция раков, анализ размеров, статистика\n\n"
                info_text += "Используйте /help для списка команд"
                self.bot.reply_to(message, info_text)
            except Exception as e:
                print(f"Ошибка команды info: {e}")
    
    # Обработка фото: детекция и отправка результата
    def process_photo(self, message):
        try:
            self.bot.reply_to(message, "Загружаю и анализирую изображение...")
            
            # Скачивание фото
            file_info = self.bot.get_file(message.photo[-1].file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            
            # Создание временной папки
            os.makedirs("temp", exist_ok=True)
            temp_path = f"temp/tg_photo_{message.chat.id}_{int(time.time())}.jpg"
            
            # Сохранение временного файла
            with open(temp_path, 'wb') as f:
                f.write(downloaded_file)
            
            # Проверка размера файла (максимум 10MB)
            file_size = os.path.getsize(temp_path)
            if file_size > 10 * 1024 * 1024:
                self.bot.reply_to(message, "Файл слишком большой (максимум 10MB)")
                os.remove(temp_path)
                return
            
            # Проверка загрузки модели
            if self.detector.model is None:
                self.bot.reply_to(message, "Модель не загружена. Пожалуйста, подождите...")
                os.remove(temp_path)
                return
            
            # Запуск детекции
            start_time = time.perf_counter()
            
            try:
                results = self.detector.model.predict(
                    source=temp_path,
                    conf=self.detector.confidence,
                    save=False
                )
            except Exception as e:
                self.bot.reply_to(message, f"Ошибка детекции: {str(e)[:100]}")
                os.remove(temp_path)
                return
            
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            
            # Обработка результатов детекции
            if results and len(results) > 0:
                try:
                    img = results[0].orig_img.copy()
                    detection_count = 0
                    
                    # Обработка OBB детекций (повернутые рамки)
                    if hasattr(results[0], 'obb') and results[0].obb is not None:
                        detection_count = len(results[0].obb)
                        print(f"Найдены OBB детекции: {detection_count}")
                        
                        # Отрисовка каждого обнаруженного рака
                        for i, obb in enumerate(results[0].obb):
                            try:
                                # Отрисовка через точки углов
                                if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None:
                                    corners = obb.xyxyxyxy[0].cpu().numpy().reshape((-1, 1, 2)).astype(np.int32)
                                    cv2.polylines(img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                                    
                                    cx = int(np.mean(corners[:, 0, 0]))
                                    cy = int(np.mean(corners[:, 0, 1]))
                                    
                                    if hasattr(obb, 'conf'):
                                        conf = obb.conf[0].item() if hasattr(obb.conf, '__len__') else obb.conf
                                        
                                        angle_text = ""
                                        if hasattr(obb, 'xywhr'):
                                            angle = obb.xywhr[0][4].item() * 180 / np.pi
                                            angle_text = f" {angle:.1f}°"
                                        
                                        text = f"#{i+1}: {conf:.2f}{angle_text}"
                                        cv2.putText(img, text, (cx - 50, cy - 20),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                # Отрисовка через xywhr
                                elif hasattr(obb, 'xywhr') and obb.xywhr is not None:
                                    xywhr = obb.xywhr[0].cpu().numpy()
                                    cx, cy, w, h, angle = xywhr
                                    
                                    rect = ((int(cx), int(cy)), (int(w), int(h)), angle * 180 / np.pi)
                                    box = cv2.boxPoints(rect)
                                    box = np.int32(box)
                                    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                                    
                                    if hasattr(obb, 'conf'):
                                        conf = obb.conf[0].item() if hasattr(obb.conf, '__len__') else obb.conf
                                        text = f"#{i+1}: {conf:.2f} {angle:.1f}°"
                                        cv2.putText(img, text, (int(cx - 50), int(cy - 20)),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            except Exception as e:
                                print(f"Ошибка отрисовки OBB {i}: {e}")
                                continue
                        
                        processed_img = img
                        
                    else:
                        # Обработка обычных box детекций
                        processed_img = results[0].plot()
                        detection_count = len(results[0].boxes) if results[0].boxes else 0
                    
                    # Сохранение изображения с результатами
                    result_path = f"temp/tg_result_{message.chat.id}_{int(time.time())}.jpg"
                    cv2.imwrite(result_path, processed_img)
                    
                    # Сохранение лога инференса в БД
                    inference_data = {
                        'timestamp': datetime.now(),
                        'image_path': temp_path,
                        'image_width': processed_img.shape[1],
                        'image_height': processed_img.shape[0],
                        'inference_time_ms': inference_time_ms,
                        'detections_count': detection_count,
                        'avg_confidence': float(np.mean([box.conf.cpu().numpy() for box in (results[0].obb if hasattr(results[0], 'obb') and results[0].obb else results[0].boxes)])) if detection_count > 0 else 0,
                        'model_name': self.detector.current_model if hasattr(self.detector, 'current_model') else 'unknown',
                        'confidence_threshold': self.detector.confidence
                    }
                    self.detector.db.save_ml_inference_log(inference_data)
                    
                    # Отправка результата пользователю
                    with open(result_path, 'rb') as photo:
                        caption = f"Обнаружено раков: {detection_count}\n"
                        caption += f"Время анализа: {inference_time_ms:.1f} мс\n"
                        caption += f"Порог уверенности: {self.detector.confidence:.2f}"
                        
                        try:
                            self.bot.send_photo(
                                message.chat.id,
                                photo,
                                caption=caption
                            )
                        except Exception as e:
                            photo.seek(0)
                            self.bot.send_photo(message.chat.id, photo)
                            self.bot.reply_to(message, caption)
                    
                    # Очистка временных файлов
                    try:
                        os.remove(temp_path)
                        os.remove(result_path)
                    except:
                        pass
                    
                except Exception as e:
                    self.bot.reply_to(message, f"Ошибка обработки результатов: {str(e)[:100]}")
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            else:
                self.bot.reply_to(message, "Раки не обнаружены на изображении")
                os.remove(temp_path)
                
        except Exception as e:
            error_msg = f"Ошибка обработки фото: {str(e)[:100]}"
            print(f"Telegram Bot Error: {e}")
            try:
                self.bot.reply_to(message, error_msg)
            except:
                pass
    
    # Отправка статистики базы данных
    def send_statistics(self, message):
        try:
            status = self.detector.db.check_database_status()
            
            stats_text = "Статистика базы данных\n\n"
            stats_text += f"Всего особей: {status.get('crayfish_count', 0)}\n"
            stats_text += f"Всего измерений: {status.get('measurements_count', 0)}\n"
            stats_text += f"Сессий анализа: {status.get('sessions_count', 0)}\n"
            stats_text += f"ML запросов: {status.get('ml_logs_count', 0)}\n\n"
            stats_text += f"Размер БД: {status.get('file_size', 0) / 1024:.1f} KB"
            
            self.bot.reply_to(message, stats_text)
            
        except Exception as e:
            self.bot.reply_to(message, f"Ошибка получения статистики: {e}")
    
    # Генерация и отправка графиков
    def send_charts(self, message):
        try:
            self.bot.reply_to(message, "Генерирую графики...")
            
            chart_paths = self.generate_charts()
            
            if chart_paths:
                for chart_path in chart_paths:
                    with open(chart_path, 'rb') as chart:
                        self.bot.send_photo(message.chat.id, chart)
                    try:
                        os.remove(chart_path)
                    except:
                        pass
            else:
                self.bot.reply_to(message, "Не удалось сгенерировать графики")
                
        except Exception as e:
            self.bot.reply_to(message, f"Ошибка создания графиков: {e}")
    
    # Генерация графиков на основе данных из БД
    def generate_charts(self):
        chart_paths = []
        
        try:
            os.makedirs("temp", exist_ok=True)
            
            # График роста размеров по дням
            daily_stats = self.detector.db.get_daily_statistics()
            if not daily_stats.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(daily_stats['date'], daily_stats['avg_width'], 'o-', linewidth=2, color='#00d4ff')
                ax.set_title('Средний размер раков по дням', fontsize=14)
                ax.set_xlabel('Дата')
                ax.set_ylabel('Ширина (мм)')
                ax.grid(True, alpha=0.3)
                
                path1 = 'temp/chart_growth.png'
                plt.tight_layout()
                plt.savefig(path1, dpi=100, facecolor='white')
                plt.close()
                chart_paths.append(path1)
            
            # График времени инференса по дням
            ml_stats = self.detector.db.get_ml_inference_stats()
            if not ml_stats.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(ml_stats['date'], ml_stats['avg_inference_time'], 'o-', linewidth=2, color='#ff6b6b')
                ax.set_title('Среднее время inference по дням', fontsize=14)
                ax.set_xlabel('Дата')
                ax.set_ylabel('Время (мс)')
                ax.grid(True, alpha=0.3)
                
                path2 = 'temp/chart_inference.png'
                plt.tight_layout()
                plt.savefig(path2, dpi=100, facecolor='white')
                plt.close()
                chart_paths.append(path2)
                
        except Exception as e:
            print(f"Ошибка генерации графиков: {e}")
            
        return chart_paths
    
    # Экспорт данных в CSV и отправка файла
    def send_export(self, message):
        try:
            self.bot.reply_to(message, "Подготавливаю данные для экспорта...")
            
            export_df = self.detector.db.get_crayfish_export_data()
            
            if not export_df.empty:
                os.makedirs("temp", exist_ok=True)
                csv_path = f"temp/export_{message.chat.id}_{int(time.time())}.csv"
                export_df.to_csv(csv_path, index=False, encoding='utf-8')
                
                with open(csv_path, 'rb') as file:
                    self.bot.send_document(
                        message.chat.id,
                        file,
                        caption="Экспорт данных в CSV"
                    )
                
                try:
                    os.remove(csv_path)
                except:
                    pass
            else:
                self.bot.reply_to(message, "В базе данных нет данных для экспорта")
                
        except Exception as e:
            self.bot.reply_to(message, f"Ошибка экспорта: {e}")
    
    # Отправка статуса системы
    def send_status(self, message):
        try:
            status_text = "Статус системы\n\n"
            
            if hasattr(self.detector, 'current_model') and self.detector.current_model:
                status_text += f"Модель: {self.detector.current_model}\n"
            else:
                status_text += "Модель: Не загружена\n"
            
            status_text += f"Порог уверенности: {self.detector.confidence:.2f}\n"
            status_text += f"Калибровка: 1px = {self.detector.calibration_factor_x:.4f} мм (X), {self.detector.calibration_factor_y:.4f} мм (Y)\n"
            
            if hasattr(self.detector, 'current_image_path') and self.detector.current_image_path:
                status_text += f"Текущее изображение: {os.path.basename(self.detector.current_image_path)}\n"
            else:
                status_text += "Текущее изображение: Нет\n"
            
            if hasattr(self.detector, 'detections') and self.detector.detections:
                status_text += f"Детекций в памяти: {len(self.detector.detections)}"
            else:
                status_text += "Детекций в памяти: 0"
            
            self.bot.reply_to(message, status_text)
            
        except Exception as e:
            self.bot.reply_to(message, f"Ошибка получения статуса: {e}")
    
    # Запуск процесса polling для получения обновлений
    def start_polling(self):
        try:
            self.running = True
            self.setup_handlers()
            print("Telegram Bot запускается...")
            
            self.bot.polling(
                none_stop=True,
                interval=1,
                timeout=30,
                long_polling_timeout=30
            )
            
        except Exception as e:
            print(f"Ошибка работы бота: {e}")
            self.running = False
    
    # Остановка Telegram бота
    def stop(self):
        self.running = False
        if self.bot:
            try:
                self.bot.stop_polling()
            except:
                pass
        print("Telegram Bot остановлен")
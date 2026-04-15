"""
utils.py - Вспомогательные функции
Содержит утилиты для калибровки, графиков, обработки изображений
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib import rcParams
from config import COLORS, CHART_FIGURE_SIZE, CHART_DPI


def setup_matplotlib_style():
    """Настройка стиля matplotlib для темной темы"""
    try:
        plt.style.use('dark_background')
    except:
        pass
    
    rcParams['axes.facecolor'] = COLORS['card_bg']
    rcParams['axes.edgecolor'] = COLORS['primary']
    rcParams['axes.labelcolor'] = 'white'
    rcParams['text.color'] = 'white'
    rcParams['xtick.color'] = 'white'
    rcParams['ytick.color'] = 'white'
    rcParams['grid.color'] = '#2a3b4c'
    rcParams['figure.facecolor'] = COLORS['dark_bg']
    rcParams['figure.edgecolor'] = COLORS['dark_bg']


def calculate_distance(point1, point2):
    """Расчет расстояния между двумя точками"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_angles_from_points(points):
    """Расчет углов четырехугольника по точкам"""
    angles = []
    for i in range(4):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % 4])
        p3 = np.array(points[(i + 2) % 4])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))
        angles.append(angle)
    
    return angles


def detect_shape_from_points(points):
    """Определение формы калибровочного объекта по точкам"""
    if len(points) != 4:
        return None
    
    # Расчет длин сторон
    distances = []
    for i in range(4):
        dist = calculate_distance(points[i], points[(i + 1) % 4])
        distances.append(dist)
    
    avg_distance = sum(distances) / 4
    max_diff = max(abs(d - avg_distance) for d in distances)
    
    if max_diff < avg_distance * 0.2:
        angles = calculate_angles_from_points(points)
        if all(80 < angle < 100 for angle in angles):
            return 'square'  # Квадрат
        else:
            return 'rhombus'  # Ромб
    else:
        return 'rectangle'  # Прямоугольник


def calculate_calibration_factors(points, scale_factor, known_width_mm, known_height_mm):
    """Расчет калибровочных коэффициентов по точкам"""
    # Горизонтальные стороны
    horizontal_sides = [
        calculate_distance(points[1], points[0]),
        calculate_distance(points[3], points[2])
    ]
    
    # Вертикальные стороны
    vertical_sides = [
        calculate_distance(points[2], points[1]),
        calculate_distance(points[0], points[3])
    ]
    
    avg_width_px = sum(horizontal_sides) / 2
    avg_height_px = sum(vertical_sides) / 2
    
    actual_width_px = avg_width_px / scale_factor
    actual_height_px = avg_height_px / scale_factor
    
    factor_x = known_width_mm / actual_width_px
    factor_y = known_height_mm / actual_height_px
    
    return factor_x, factor_y, actual_width_px, actual_height_px


def create_growth_plots(daily_stats, db, colors=COLORS):
    """Создание графиков анализа роста"""
    fig, axes = plt.subplots(2, 2, figsize=CHART_FIGURE_SIZE)
    fig.suptitle('АНАЛИЗ РОСТА ПОПУЛЯЦИИ РАКОВ', 
                fontsize=20, color='white', fontweight='bold', y=0.98)
    
    daily_stats_sorted = daily_stats.sort_values('date')
    
    # График 1: Средний размер по дням
    create_avg_size_chart(axes[0, 0], daily_stats_sorted, colors)
    
    # График 2: Распределение размеров
    create_size_distribution_chart(axes[0, 1], db, colors)
    
    # График 3: Количество особей по дням
    create_crayfish_count_chart(axes[1, 0], daily_stats_sorted, colors)
    
    # График 4: Сводная статистика
    create_summary_stats_panel(axes[1, 1], daily_stats_sorted, colors)
    
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    return fig


def create_avg_size_chart(ax, daily_stats_sorted, colors):
    """Создание графика среднего размера по дням"""
    if len(daily_stats_sorted) >= 1:
        ax.plot(daily_stats_sorted['date'], daily_stats_sorted['avg_width'], 
               marker='o', markersize=8, color=colors['chart_1'], 
               linewidth=3, label='Средний размер', zorder=5)
        
        # Добавление тренда при достаточном количестве данных
        if len(daily_stats_sorted) >= 3:
            try:
                from scipy.interpolate import make_interp_spline
                x_num = mdates.date2num(daily_stats_sorted['date'])
                y_vals = daily_stats_sorted['avg_width'].values
                
                x_smooth = np.linspace(x_num.min(), x_num.max(), 300)
                spline = make_interp_spline(x_num, y_vals, k=3)
                y_smooth = spline(x_smooth)
                dates_smooth = mdates.num2date(x_smooth)
                
                ax.plot(dates_smooth, y_smooth, '-', 
                       color=colors['chart_2'], linewidth=2, alpha=0.7,
                       label='Тренд роста', zorder=4)
            except:
                pass
        
        # Заливка диапазона min-max
        ax.fill_between(daily_stats_sorted['date'], 
                       daily_stats_sorted['min_width'], 
                       daily_stats_sorted['max_width'],
                       alpha=0.2, color=colors['chart_1'],
                       label='Min-Max диапазон', zorder=1)
        
        # Настройка внешнего вида
        ax.set_title('Средний размер по дням', 
                    color='white', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Дата', color='white', fontsize=11)
        ax.set_ylabel('Ширина (мм)', color='white', fontsize=11)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # Адаптивный формат дат
        configure_date_axis(ax, len(daily_stats_sorted))
        
        ax.legend(facecolor=colors['card_bg'], edgecolor='white',
                 fontsize=9, loc='best', framealpha=0.9)
        
        # Аннотации значений для небольших наборов данных
        if len(daily_stats_sorted) <= 8:
            for _, row in daily_stats_sorted.iterrows():
                ax.annotate(f"{row['avg_width']:.1f}мм", 
                           xy=(row['date'], row['avg_width']),
                           xytext=(0, 8), textcoords='offset points',
                           ha='center', fontsize=8, color='white', fontweight='bold')


def create_size_distribution_chart(ax, db, colors):
    """Создание гистограммы распределения размеров"""
    all_measurements = db.get_all_measurements()
    
    if not all_measurements.empty and 'width_mm' in all_measurements.columns:
        sizes = pd.to_numeric(all_measurements['width_mm'], errors='coerce')
        sizes = sizes.dropna()
        
        if len(sizes) > 0:
            n_bins = min(12, len(sizes))
            if n_bins > 0:
                ax.hist(sizes, bins=n_bins, alpha=0.7, color=colors['chart_2'], 
                       edgecolor='white', density=True)
                
                # Добавление кривой плотности
                try:
                    from scipy.stats import gaussian_kde
                    density = gaussian_kde(sizes)
                    xs = np.linspace(sizes.min(), sizes.max(), 200)
                    ax.plot(xs, density(xs), color=colors['accent'], linewidth=2)
                except:
                    ax.axvline(sizes.mean(), color='yellow', linestyle='--', 
                              linewidth=2, alpha=0.7)
            
            # Настройка графика
            ax.set_title('Распределение размеров', 
                        color='white', fontsize=14, fontweight='bold', pad=12)
            ax.set_xlabel('Ширина (мм)', color='white', fontsize=11)
            ax.set_ylabel('Частота', color='white', fontsize=11)
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2)
            
            # Статистика на графике
            stats_text = f"Всего измерений: {len(sizes)}\n"
            stats_text += f"Среднее: {sizes.mean():.1f} мм\n"
            stats_text += f"Медиана: {np.median(sizes):.1f} мм\n"
            stats_text += f"Станд. откл.: {sizes.std():.1f} мм"
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, color=colors['text_primary'],
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'Нет данных о размерах',
               ha='center', va='center', color='white')
        ax.set_title('Распределение размеров', color='white')


def create_crayfish_count_chart(ax, daily_stats_sorted, colors):
    """Создание графика количества особей по дням"""
    if not daily_stats_sorted.empty:
        bars = ax.bar(daily_stats_sorted['date'], daily_stats_sorted['unique_crayfish_count'],
                     color=colors['chart_3'], alpha=0.7, edgecolor='white', width=0.6)
        
        ax.set_title('Количество особей по дням', 
                    color='white', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Дата', color='white', fontsize=11)
        ax.set_ylabel('Количество особей', color='white', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
        
        # Адаптивный формат дат
        configure_date_axis(ax, len(daily_stats_sorted))
        
        # Аннотации значений
        if len(daily_stats_sorted) <= 10:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords='offset points',
                               ha='center', va='bottom', fontsize=8, color='white')
        
        # Статистика на графике
        total_crayfish = daily_stats_sorted['unique_crayfish_count'].sum()
        avg_per_day = daily_stats_sorted['unique_crayfish_count'].mean()
        
        stats_text = f"Всего обнаружений: {total_crayfish}\n"
        stats_text += f"Среднее в день: {avg_per_day:.1f}\n"
        stats_text += f"Макс. в день: {daily_stats_sorted['unique_crayfish_count'].max()}"
        
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, color=colors['text_primary'],
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    else:
        ax.text(0.5, 0.5, 'Нет данных по дням',
               ha='center', va='center', color='white')
        ax.set_title('Количество особей по дням', color='white')


def create_summary_stats_panel(ax, daily_stats_sorted, colors):
    """Создание панели со сводной статистикой"""
    ax.axis('off')
    
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
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, color=colors['text_primary'],
               verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=colors['card_bg'], 
                        alpha=0.9, edgecolor=colors['primary']))


def configure_date_axis(ax, days_count):
    """Настройка формата отображения дат на оси X"""
    if days_count <= 7:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    elif days_count <= 14:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
        interval = max(3, days_count // 8)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)


def format_detection_results_text(detections, confidence, calibration_factor_x, 
                                  calibration_factor_y, current_image_path, 
                                  session_id, processing_time, inference_time_ms, 
                                  current_model, db):
    """Форматирование текста с результатами детекции"""
    from pathlib import Path
    
    model_type = ""
    if current_model:
        ext = Path(current_model).suffix.lower()
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
    text += f"Обнаружено: {len(detections) if detections else 0}\n"
    text += f"Порог: {confidence:.2f}\n"
    text += f"Калибровка X: 1px = {calibration_factor_x:.4f}мм\n"
    text += f"Калибровка Y: 1px = {calibration_factor_y:.4f}мм\n"
    text += f"Файл: {Path(current_image_path).name}\n"
    text += f"Сессия: {session_id}\n\n"
    
    if detections and len(detections) > 0:
        text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        text += f"         ДЕТАЛИ ОБНАРУЖЕНИЙ          \n"
        text += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        try:
            measurements_df = db.get_all_measurements()
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
        text += f"• Уменьшите порог уверенности (сейчас {confidence:.2f})\n"
        text += f"• Убедитесь, что раки видны на фото\n"
        text += f"• Попробуйте другое изображение\n"
        text += f"• Проверьте загруженную модель"
    
    text += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    return text


def create_database_stats_text(status, daily_stats, available_dates):
    """Создание текста со статистикой базы данных"""
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
    
    return stats_text
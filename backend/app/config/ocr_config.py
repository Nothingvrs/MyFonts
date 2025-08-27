"""
Конфигурация PaddleOCR для настройки параметров детекции текста
"""
import os
from typing import Optional

# Основные настройки PaddleOCR
PADDLEOCR_CONFIG = {
    # Язык и базовые настройки
    'lang': 'ru',                    # Русский язык (включает кириллицу)
    'use_angle_cls': True,           # Определение ориентации текста
    
    # Настройки детекции (detection)
    'det_db_thresh': 0.05,           # Порог детекции текста (0.05 = очень агрессивно)
    'det_db_box_thresh': 0.1,        # Порог для bounding box (0.1 = очень агрессивно)
    'det_db_unclip_ratio': 4.0,      # Расширение области детекции (4.0 = максимальное)
    'det_limit_side_len': 960,       # Ограничение размера стороны для детекции
    'det_limit_type': 'min',         # Ограничение по минимальной стороне
    
    # Настройки распознавания (recognition)
    'rec_batch_num': 32,             # Размер батча для распознавания
    'max_text_length': 300,          # Максимальная длина текста в строке
    'rec_algorithm': 'SVTR_LCNet',   # Алгоритм распознавания
    'rec_char_dict_path': None,      # Используем встроенный словарь
}

# Настройки качества текста
TEXT_QUALITY_CONFIG = {
    'min_confidence': 0.25,          # Минимальная уверенность для отдельной области
    'min_text_length': 3,            # Минимальная длина очищенного текста
    'min_avg_confidence': 0.20,      # Минимальная средняя уверенность по всем регионам
    'min_regions_count': 1,          # Минимум валидных регионов
    'min_letters_count': 3,          # Минимум буквенных символов в тексте
}

# Настройки детекции множественных шрифтов
MULTIPLE_FONTS_CONFIG = {
    'size_variation_threshold': 0.4,     # Порог вариации размеров (40% для более точной детекции)
    'area_ratio_threshold': 2.5,         # Порог соотношения площадей (2.5 для лучшей детекции)
    'min_regions_count': 4,              # Минимальное количество областей для множественных шрифтов
    'height_ratio_threshold': 2.0,       # Порог соотношения высот текста
    'min_size_groups': 2,                # Минимальное количество групп размеров
}

# Пресеты чувствительности детекции множественных шрифтов
# Можно выбрать через переменную окружения MULTIPLE_FONTS_SENSITIVITY: strict | balanced | relaxed
MULTIPLE_FONTS_SENSITIVITY_PRESETS = {
    'strict': {
        # Требуем более явные различия и больше данных
        'min_regions_count': 6,
        'min_regions_per_cluster': 4,
        'min_lines_for_multi': 2,
        'require_metric_count': 2,  # минимум 2 метрики должны различаться
        'size_variation_threshold': 0.6,
        'height_ratio_threshold': 2.2,
        'area_ratio_threshold': 3.5,
        'in_band_frac': 0.80,  # ранний признак одного шрифта
        'density_diff_threshold': 0.14,
        'saturation_diff_threshold': 24.0,
        'brightness_diff_threshold': 14.0,
    },
    'balanced': {
        # Значения по умолчанию (сбалансированные)
        'min_regions_count': 5,
        'min_regions_per_cluster': 3,
        'min_lines_for_multi': 2,
        'require_metric_count': 2,
        'size_variation_threshold': 0.5,
        'height_ratio_threshold': 2.0,
        'area_ratio_threshold': 3.0,
        'in_band_frac': 0.75,
        'density_diff_threshold': 0.12,
        'saturation_diff_threshold': 20.0,
        'brightness_diff_threshold': 12.0,
    },
    'relaxed': {
        # Более чувствительно, допускает меньше данных и слабее различия
        'min_regions_count': 4,
        'min_regions_per_cluster': 2,
        'min_lines_for_multi': 2,
        'require_metric_count': 1,
        'size_variation_threshold': 0.4,
        'height_ratio_threshold': 1.8,
        'area_ratio_threshold': 2.5,
        'in_band_frac': 0.70,
        'density_diff_threshold': 0.10,
        'saturation_diff_threshold': 16.0,
        'brightness_diff_threshold': 10.0,
    },
}

# Настройки предобработки изображений
IMAGE_PREPROCESSING_CONFIG = {
    'resize_threshold': 600,             # Порог для увеличения изображения
    'max_resize_scale': 4,               # Максимальный масштаб увеличения
    'contrast_alpha': 3.0,               # Коэффициент контраста
    'contrast_beta': 50,                 # Смещение яркости
    'clahe_clip_limit': 5.0,             # CLAHE clip limit
    'clahe_tile_size': (8, 8),           # CLAHE tile size
    'adaptive_block_size': 15,           # Размер блока для адаптивной бинаризации
    'adaptive_c': 5,                     # Константа для адаптивной бинаризации
    'morphology_kernel_size': (2, 2),    # Размер ядра для морфологии
    'brightness_alpha': 2.0,             # Коэффициент яркости
    'brightness_beta': 80,               # Смещение яркости
}

# Настройки для разных типов изображений
IMAGE_TYPE_CONFIGS = {
    'advertisement': {
        'det_db_thresh': 0.0001,         # АБСОЛЮТНЫЙ МИНИМУМ для рекламы
        'det_db_box_thresh': 0.001,      # АБСОЛЮТНЫЙ МИНИМУМ
        'det_db_unclip_ratio': 15.0,     # МАКСИМАЛЬНОЕ расширение
        'min_confidence': 0.0001,        # АБСОЛЮТНЫЙ МИНИМУМ порог
        'min_avg_confidence': 0.0001,    # АБСОЛЮТНЫЙ МИНИМУМ порог
    },
    'document': {
        'det_db_thresh': 0.1,            # Умеренно агрессивно для документов
        'det_db_box_thresh': 0.15,       # Умеренно агрессивно
        'det_db_unclip_ratio': 3.0,      # Умеренное расширение
        'min_confidence': 0.2,           # Средний порог
        'min_avg_confidence': 0.15,      # Средний порог
    },
    'book': {
        'det_db_thresh': 0.08,           # Агрессивно для книг
        'det_db_box_thresh': 0.12,       # Агрессивно
        'det_db_unclip_ratio': 3.5,      # Большое расширение
        'min_confidence': 0.15,          # Низкий порог
        'min_avg_confidence': 0.12,      # Низкий порог
    }
}

def get_ocr_config(image_type='default'):
    """
    Получение конфигурации PaddleOCR для определенного типа изображения
    
    Args:
        image_type (str): Тип изображения ('advertisement', 'document', 'book', 'default')
    
    Returns:
        dict: Конфигурация PaddleOCR
    """
    config = PADDLEOCR_CONFIG.copy()
    
    if image_type in IMAGE_TYPE_CONFIGS:
        # Обновляем настройки для конкретного типа изображения
        for key, value in IMAGE_TYPE_CONFIGS[image_type].items():
            if key in config:
                config[key] = value
    
    return config

def get_text_quality_config():
    """Получение конфигурации качества текста"""
    return TEXT_QUALITY_CONFIG.copy()

def get_multiple_fonts_config(mode: Optional[str] = None):
    """Получение конфигурации детекции множественных шрифтов

    Приоритет выбора чувствительности:
    1) Аргумент mode (strict|balanced|relaxed)
    2) Переменная окружения MULTIPLE_FONTS_SENSITIVITY
    3) balanced (по умолчанию)
    """
    base = MULTIPLE_FONTS_CONFIG.copy()
    env_mode = (os.getenv('MULTIPLE_FONTS_SENSITIVITY') or '').strip().lower()
    eff_mode = (mode or env_mode or 'strict').lower()
    if eff_mode not in MULTIPLE_FONTS_SENSITIVITY_PRESETS:
        eff_mode = 'balanced'
    preset = MULTIPLE_FONTS_SENSITIVITY_PRESETS[eff_mode]
    # Сливаем пресет поверх базовой конфигурации
    merged = base | preset  # Python 3.9+: объединение словарей
    return merged

def get_preprocessing_config():
    """Получение конфигурации предобработки изображений"""
    return IMAGE_PREPROCESSING_CONFIG.copy()

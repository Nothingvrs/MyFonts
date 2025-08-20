"""
Конфигурация PaddleOCR для настройки параметров детекции текста
"""

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
    'min_confidence': 0.05,          # КРИТИЧЕСКИ низкая уверенность для валидной области
    'min_text_length': 1,            # Минимальная длина текста (1 символ)
    'min_avg_confidence': 0.03,      # КРИТИЧЕСКИ низкая средняя уверенность
}

# Настройки детекции множественных шрифтов
MULTIPLE_FONTS_CONFIG = {
    'size_variation_threshold': 0.25,    # Порог вариации размеров (25% = менее чувствительно)
    'area_ratio_threshold': 3.0,         # Порог соотношения площадей (3.0 = менее чувствительно)
    'min_regions_count': 6,              # Минимальное количество областей для множественных шрифтов
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
    'det_db_thresh': 0.01,           # КРИТИЧЕСКИ низкий для рекламы
    'det_db_box_thresh': 0.05,       # КРИТИЧЕСКИ низкий
    'det_db_unclip_ratio': 5.0,      # МАКСИМАЛЬНОЕ расширение
    'min_confidence': 0.05,          # КРИТИЧЕСКИ низкий порог
    'min_avg_confidence': 0.03,      # КРИТИЧЕСКИ низкий порог
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

def get_multiple_fonts_config():
    """Получение конфигурации детекции множественных шрифтов"""
    return MULTIPLE_FONTS_CONFIG.copy()

def get_preprocessing_config():
    """Получение конфигурации предобработки изображений"""
    return IMAGE_PREPROCESSING_CONFIG.copy()

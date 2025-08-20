"""
Сервис для работы с PaddleOCR - профессиональная детекция и распознавание текста
"""

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..config.ocr_config import get_ocr_config, get_text_quality_config, get_multiple_fonts_config

# Сначала определяем logger
logger = logging.getLogger(__name__)

# Проверяем доступность основных зависимостей
try:
    import numpy as np
    logger.info("✅ NumPy доступен")
except ImportError as e:
    logger.error(f"❌ NumPy не доступен: {str(e)}")
    raise

try:
    import cv2
    logger.info("✅ OpenCV доступен")
except ImportError as e:
    logger.error(f"❌ OpenCV не доступен: {str(e)}")
    raise

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    
    # Проверяем версию PaddleOCR
    try:
        import paddleocr
        version = getattr(paddleocr, '__version__', 'unknown')
        logger.info(f"📦 PaddleOCR версия: {version}")
    except Exception as version_error:
        logger.info(f"📦 PaddleOCR версия: неизвестна (ошибка: {str(version_error)})")
        
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None
    logger.error(f"❌ Ошибка импорта PaddleOCR: {str(e)}")
    logger.error("💡 Установите: pip install paddlepaddle paddleocr")

if PADDLEOCR_AVAILABLE:
    logger.info("✅ PaddleOCR успешно импортирован")
else:
    logger.error("❌ PaddleOCR не импортирован")


class PaddleOCRService:
    """Сервис для профессиональной детекции и анализа текста с помощью PaddleOCR"""
    
    def __init__(self):
        self.ocr = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Безопасная инициализация
        try:
            self._initialize_ocr()
        except Exception as init_error:
            logger.error(f"❌ Ошибка инициализации в конструкторе: {str(init_error)}")
            logger.error(f"💡 Тип ошибки: {type(init_error).__name__}")
            # Не падаем, просто оставляем self.ocr = None
    
    def _initialize_ocr(self):
        """Инициализация PaddleOCR с максимально агрессивными настройками"""
        try:
            if not PADDLEOCR_AVAILABLE:
                logger.error("❌ PaddleOCR не установлен!")
                logger.error("💡 Установите: pip install paddlepaddle paddleocr")
                return
            
            logger.info("🚀 Начинаем инициализацию PaddleOCR...")
            
            # Получаем максимально агрессивную конфигурацию для рекламы
            try:
                ocr_config = get_ocr_config('advertisement')
                logger.info("✅ Конфигурация OCR загружена успешно")
                
                # Упрощаем конфигурацию - убираем потенциально проблемные параметры
                safe_config = {
                    'lang': ocr_config.get('lang', 'ru'),
                    'use_textline_orientation': True,  # Новый параметр вместо use_angle_cls
                    'det_db_thresh': 0.001,  # ЕЩЕ БОЛЕЕ АГРЕССИВНО
                    'det_db_box_thresh': 0.01,  # ЕЩЕ БОЛЕЕ АГРЕССИВНО
                    'det_db_unclip_ratio': 10.0,  # МАКСИМАЛЬНОЕ расширение
                    'det_limit_side_len': 4096,  # Увеличиваем лимит размера
                    'det_limit_type': 'max',  # По максимальной стороне
                }
                ocr_config = safe_config
                logger.info("🔄 Используем СУПЕР агрессивную конфигурацию")
                
            except Exception as config_error:
                logger.error(f"❌ Ошибка загрузки конфигурации OCR: {str(config_error)}")
                logger.info("🔄 Используем СУПЕР агрессивную конфигурацию по умолчанию...")
                ocr_config = {
                    'lang': 'ru',
                    'use_textline_orientation': True,  # Новый параметр вместо use_angle_cls
                    'det_db_thresh': 0.001,  # СУПЕР агрессивно
                    'det_db_box_thresh': 0.01,  # СУПЕР агрессивно
                    'det_db_unclip_ratio': 10.0,  # МАКСИМАЛЬНОЕ расширение
                    'det_limit_side_len': 4096,  # Большой лимит
                    'det_limit_type': 'max',  # По максимальной стороне
                }
            
            logger.info("🚀 Инициализация PaddleOCR с максимально агрессивными настройками...")
            logger.info(f"  - det_db_thresh: {ocr_config['det_db_thresh']} (КРИТИЧЕСКИ низкий)")
            logger.info(f"  - det_db_box_thresh: {ocr_config['det_db_box_thresh']} (КРИТИЧЕСКИ низкий)")
            logger.info(f"  - det_db_unclip_ratio: {ocr_config['det_db_unclip_ratio']} (МАКСИМАЛЬНОЕ расширение)")
            
            # Создаем PaddleOCR с агрессивными настройками
            logger.info("🔄 Создание PaddleOCR с агрессивными настройками...")
            logger.info(f"📋 Конфигурация: {ocr_config}")
            
            try:
                self.ocr = PaddleOCR(**ocr_config)
                logger.info("✅ PaddleOCR объект создан с агрессивными настройками")
                logger.info("✅ Агрессивные настройки применены")
                
            except Exception as create_error:
                logger.error(f"❌ Ошибка создания PaddleOCR объекта: {str(create_error)}")
                logger.error(f"💡 Тип ошибки: {type(create_error).__name__}")
                logger.error(f"🔍 Детали: {repr(create_error)}")
                
                # Пробуем создать с минимальной конфигурацией
                logger.info("🔄 Пробуем создать с минимальной конфигурацией...")
                try:
                    minimal_config = {
                        'lang': 'ru',
                        'use_textline_orientation': False,  # Отключаем классификацию углов
                    }
                    logger.info(f"📋 Минимальная конфигурация: {minimal_config}")
                    self.ocr = PaddleOCR(**minimal_config)
                    logger.info("✅ PaddleOCR создан с минимальной конфигурацией")
                except Exception as minimal_error:
                    logger.error(f"❌ Ошибка создания с минимальной конфигурацией: {str(minimal_error)}")
                    
                    # Последняя попытка - только базовые параметры
                    logger.info("🔄 Последняя попытка - только базовые параметры...")
                    try:
                        basic_config = {'lang': 'ru', 'use_textline_orientation': False}
                        logger.info(f"📋 Базовая конфигурация: {basic_config}")
                        self.ocr = PaddleOCR(**basic_config)
                        logger.info("✅ PaddleOCR создан с базовой конфигурацией")
                    except Exception as basic_error:
                        logger.error(f"❌ Критическая ошибка - PaddleOCR не может быть создан: {str(basic_error)}")
                        self.ocr = None
                        return
            
            # Проверяем что объект создался
            if self.ocr is None:
                logger.error("❌ PaddleOCR объект не создался!")
                return
            
            logger.info("✅ PaddleOCR объект создан успешно")
            
            # Делаем тестовый вызов для проверки работоспособности
            logger.info("🧪 Тестируем PaddleOCR...")
            try:
                # Создаем тестовое изображение с текстом
                test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
                # Добавляем простой черный текст
                cv2.putText(test_image, "TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                logger.info("🖼️ Создано тестовое изображение 200x400 с текстом 'TEST'")
                
                # Проверяем что у объекта есть метод predict
                if not hasattr(self.ocr, 'predict') or not callable(getattr(self.ocr, 'predict', None)):
                    logger.error("❌ У объекта PaddleOCR нет метода predict")
                    self.ocr = None
                    return
                
                logger.info("✅ Метод predict найден, делаем тестовый вызов...")
                test_result = self.ocr.predict(test_image)
                logger.info(f"✅ PaddleOCR тест прошел успешно, результат: {type(test_result)}")
                
                # Проверяем что тест действительно нашел текст
                if test_result and len(test_result) > 0 and test_result[0]:
                    logger.info(f"✅ Тест найден текст: {len(test_result[0])} областей")
                    for i, detection in enumerate(test_result[0]):
                        if len(detection) >= 2 and len(detection[1]) >= 2:
                            text = detection[1][0]
                            conf = detection[1][1]
                            logger.info(f"  - Область {i+1}: '{text}' (уверенность: {conf:.3f})")
                else:
                    logger.warning("⚠️ Тест не нашел текст - возможно проблема с настройками")
                
            except Exception as test_error:
                logger.error(f"❌ PaddleOCR тест не прошел: {str(test_error)}")
                logger.error(f"💡 Тип ошибки теста: {type(test_error).__name__}")
                logger.error(f"🔍 Детали теста: {repr(test_error)}")
                self.ocr = None
                return
            
            logger.info("🎉 PaddleOCR полностью инициализирован и готов к работе!")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка инициализации PaddleOCR: {str(e)}")
            logger.error(f"💡 Тип ошибки: {type(e).__name__}")
            self.ocr = None
    
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Алиас для обратной совместимости"""
        return await self.detect_and_analyze_text(image)
    
    async def detect_and_analyze_text(self, image: np.ndarray) -> Dict[str, Any]:
        """Детекция и анализ текста на изображении"""
        if not self.ocr:
            logger.error("❌ PaddleOCR не инициализирован")
            return {
                'has_text': False,
                'text_regions': [],
                'multiple_fonts': False,
                'confidence': 0.0,
                'text_content': '',
                'error': 'PaddleOCR не инициализирован'
            }
        
        try:
            # Запускаем OCR в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_ocr_sync,
                image
            )
            return result
            
        except Exception as e:
            logger.error(f"Техническая ошибка PaddleOCR: {str(e)}")
            return {
                'has_text': False,
                'text_regions': [],
                'multiple_fonts': False,
                'confidence': 0.0,
                'text_content': '',
                'error': f"Техническая ошибка OCR: {str(e)}"
            }
    
    def _create_image_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """Создание множественных вариантов изображения для агрессивного поиска текста"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            variants = []
            
            # 1. Оригинальное изображение
            variants.append(image.copy())
            
            # 2. Увеличенное изображение (для мелкого текста)
            try:
                h, w = gray.shape
                if min(h, w) < 800:  # Увеличиваем порог
                    scale = max(3, 800 // min(h, w))  # Более агрессивное увеличение
                    if len(image.shape) == 3:
                        resized = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                    else:
                        resized_gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                        resized = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)
                    variants.append(resized)
                
                # Очень большое увеличение для мелкого текста
                if min(h, w) < 400:  # Увеличиваем порог
                    scale = max(5, 1000 // min(h, w))  # Еще более агрессивное увеличение
                    if len(image.shape) == 3:
                        resized_large = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                    else:
                        resized_gray_large = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                        resized_large = cv2.cvtColor(resized_gray_large, cv2.COLOR_GRAY2RGB)
                    variants.append(resized_large)
            except:
                pass
            
            # 3. Высокий контраст
            try:
                enhanced = cv2.convertScaleAbs(gray, alpha=3.0, beta=50)
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                variants.append(enhanced_rgb)
            except:
                pass
            
            # 4. CLAHE (адаптивная эквализация)
            try:
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
                clahe_image = clahe.apply(gray)
                clahe_rgb = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
                variants.append(clahe_rgb)
            except:
                pass
            
            # 5. Адаптивная бинаризация
            try:
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
                adaptive_rgb = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)
                variants.append(adaptive_rgb)
            except:
                pass
            
            # 6. Инверсия (для белого текста на темном фоне)
            try:
                inverted = cv2.bitwise_not(gray)
                inverted_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
                variants.append(inverted_rgb)
            except:
                pass
            
            # 7. Морфологическая очистка
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                morphed_rgb = cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB)
                variants.append(morphed_rgb)
            except:
                pass
            
            # 8. Легкое размытие
            try:
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
                variants.append(blurred_rgb)
            except:
                pass
            
            # 9. Увеличенная яркость
            try:
                brightened = cv2.convertScaleAbs(gray, alpha=2.0, beta=80)
                brightened_rgb = cv2.cvtColor(brightened, cv2.COLOR_GRAY2RGB)
                variants.append(brightened_rgb)
            except:
                pass
            
            # 10. Двойная инверсия
            try:
                double_inverted = cv2.bitwise_not(cv2.bitwise_not(gray))
                double_inverted_rgb = cv2.cvtColor(double_inverted, cv2.COLOR_GRAY2RGB)
                variants.append(double_inverted_rgb)
            except:
                pass
            
            # 11. Локальная эквализация
            try:
                clahe_local = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4))
                local_eq = clahe_local.apply(gray)
                local_eq_rgb = cv2.cvtColor(local_eq, cv2.COLOR_GRAY2RGB)
                variants.append(local_eq_rgb)
            except:
                pass
            
            # 12. Комбинированная обработка
            try:
                combined = cv2.convertScaleAbs(gray, alpha=2.5, beta=60)
                combined = cv2.GaussianBlur(combined, (3, 3), 0)
                combined = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
                variants.append(combined_rgb)
            except:
                pass
            
            # 13. Экстремальный контраст
            try:
                extreme_contrast = cv2.convertScaleAbs(gray, alpha=5.0, beta=100)
                extreme_rgb = cv2.cvtColor(extreme_contrast, cv2.COLOR_GRAY2RGB)
                variants.append(extreme_rgb)
            except:
                pass
            
            # 14. Морфологическая операция открытия (убирает шум)
            try:
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open)
                opened_rgb = cv2.cvtColor(opened, cv2.COLOR_GRAY2RGB)
                variants.append(opened_rgb)
            except:
                pass
            
            # 15. Морфологическая операция закрытия (заполняет пробелы)
            try:
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
                closed_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
                variants.append(closed_rgb)
            except:
                pass
            
            logger.info(f"✅ Создано {len(variants)} вариантов для OCR")
            return variants
            
        except Exception as e:
            logger.error(f"Ошибка создания вариантов: {str(e)}")
            return [image.copy()]
    
    def _run_ocr_sync(self, image: np.ndarray) -> Dict[str, Any]:
        """Синхронный запуск OCR с максимально агрессивными настройками"""
        try:
            logger.info("🔍 Запуск PaddleOCR анализа...")
            logger.info(f"🖼️ Размер изображения: {image.shape}")
            
            # Создаем варианты изображения
            image_variants = self._create_image_variants(image)
            logger.info(f"🔄 Создано {len(image_variants)} вариантов изображения")
            
            ocr_result = None
            best_result = None
            best_confidence = 0.0
            
            # Пробуем каждый вариант изображения
            for i, variant in enumerate(image_variants):
                try:
                    logger.info(f"🔍 Попытка OCR #{i+1}/{len(image_variants)}")
                    logger.info(f"  - Размер варианта: {variant.shape}")
                    logger.info(f"  - Тип данных: {variant.dtype}")
                    logger.info(f"  - Диапазон значений: [{variant.min()}, {variant.max()}]")
                    
                    # Вызываем PaddleOCR
                    variant_result = self.ocr.predict(variant)
                    
                    logger.info(f"🔍 Вариант #{i+1}: результат PaddleOCR: {type(variant_result)}")
                    if variant_result:
                        logger.info(f"  - Длина результата: {len(variant_result)}")
                        if len(variant_result) > 0:
                            logger.info(f"  - Первая страница: {len(variant_result[0]) if variant_result[0] else 0} элементов")
                            # Детальный анализ результата
                            if variant_result[0]:
                                logger.info(f"  - Детали первой страницы:")
                                for j, detection in enumerate(variant_result[0]):
                                    if len(detection) >= 2:
                                        bbox = detection[0]
                                        text_info = detection[1]
                                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                            text = str(text_info[0])
                                            conf = float(text_info[1])
                                            logger.info(f"    * Область {j+1}: текст='{text}', уверенность={conf:.3f}, bbox={bbox}")
                                        else:
                                            logger.info(f"    * Область {j+1}: неожиданный формат {text_info}")
                            else:
                                logger.info(f"  - Первая страница пустая")
                    else:
                        logger.info(f"  - PaddleOCR вернул None")
                    
                    if variant_result and len(variant_result) > 0 and variant_result[0]:
                        # Вычисляем среднюю уверенность
                        confidences = []
                        for detection in variant_result[0]:
                            if len(detection) >= 2 and len(detection[1]) >= 2:
                                conf = detection[1][1] if isinstance(detection[1][1], (int, float)) else 0.0
                                confidences.append(conf)
                        
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        logger.info(f"✅ Вариант #{i+1}: найдено {len(variant_result[0])} регионов, уверенность: {avg_confidence:.2f}")
                        
                        # Сохраняем лучший результат
                        if avg_confidence > best_confidence:
                            best_result = variant_result
                            best_confidence = avg_confidence
                            logger.info(f"🏆 Новый лучший результат: уверенность {avg_confidence:.2f}")
                    else:
                        logger.info(f"❌ Вариант #{i+1}: текст не найден")
                        
                except Exception as variant_error:
                    logger.error(f"💥 Ошибка OCR варианта #{i+1}: {str(variant_error)}")
                    continue
            
            # Используем лучший результат
            if best_result:
                ocr_result = best_result
                logger.info(f"✅ Используем лучший результат с уверенностью {best_confidence:.2f}")
            else:
                logger.info(f"ℹ️ OCR не нашел текст ни в одном варианте изображения")
                return {
                    'has_text': False,
                    'text_regions': [],
                    'multiple_fonts': False,
                    'confidence': 0.0,
                    'text_content': '',
                    'error': "OCR не нашел текст на изображении"
                }
            
            # Обрабатываем результат
            page_result = ocr_result[0] if len(ocr_result) > 0 else None
            
            if not page_result:
                logger.info("ℹ️ PaddleOCR не обнаружил текст на странице")
                return {
                    'has_text': False,
                    'text_regions': [],
                    'multiple_fonts': False,
                    'confidence': 0.0,
                    'text_content': '',
                    'error': "OCR не обнаружил текст на странице"
                }
            
            # Обрабатываем строки
            text_regions = []
            all_text = []
            confidences = []
            
            for line in page_result:
                if line and len(line) >= 2:
                    bbox = line[0]  # Координаты области
                    text_info = line[1]  # (текст, уверенность)
                    
                    # Извлекаем текст и уверенность
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0])
                        confidence = float(text_info[1])
                    elif isinstance(text_info, str):
                        text = text_info
                        confidence = 1.0
                    else:
                        continue
                    
                    # Анализируем область текста
                    region_info = self._analyze_text_region(image, bbox, text, confidence)
                    text_regions.append(region_info)
                    
                    all_text.append(text)
                    confidences.append(confidence)
            
            # Статистика и проверка качества
            avg_confidence = np.mean(confidences) if confidences else 0.0
            text_content = ' '.join(all_text)
            
            # Получаем конфигурацию качества текста
            quality_config = get_text_quality_config()
            
            # Проверяем наличие текста с максимально низкими порогами
            valid_regions = [r for r in text_regions if 
                           r.get('confidence', 0) > quality_config['min_confidence'] and
                           not r.get('is_invalid', False)]
            clean_text = ''.join(c for c in text_content if c.isalnum() or c.isspace()).strip()
            
            invalid_regions = [r for r in text_regions if r.get('is_invalid', False)]
            
            has_text = (len(valid_regions) > 0 and 
                       len(clean_text) >= quality_config['min_text_length'] and
                       avg_confidence > quality_config['min_avg_confidence'] and
                       len(invalid_regions) < len(text_regions))
            
            logger.info(f"🔍 Проверка качества: всего областей={len(text_regions)}, валидных={len(valid_regions)}")
            logger.info(f"📝 Чистый текст: '{clean_text[:50]}' (длина: {len(clean_text)})")
            logger.info(f"📊 Средняя уверенность: {avg_confidence:.2f}")
            logger.info(f"✅ Результат проверки: has_text={has_text}")
            
            # Определяем множественные шрифты
            multiple_fonts = self._detect_multiple_fonts_from_regions(text_regions)
            
            # Формируем результат
            result = {
                'has_text': has_text,
                'text_regions': text_regions,
                'multiple_fonts': multiple_fonts,
                'confidence': avg_confidence,
                'text_content': text_content,
                'regions_count': len(text_regions),
                'ocr_boxes': text_regions,
                'error': None if has_text else "OCR нашел текст, но он не прошел проверку качества"
            }
            
            logger.info(f"✅ PaddleOCR результат: has_text={has_text}, текст='{text_content[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Техническая ошибка в _run_ocr_sync: {str(e)}")
            raise
    
    def _analyze_text_region(self, image: np.ndarray, bbox: List, text: str, confidence: float) -> Dict[str, Any]:
        """Анализ отдельной области текста"""
        try:
            # Безопасное извлечение координат
            try:
                if isinstance(bbox[0], (list, tuple)):
                    points = np.array(bbox, dtype=np.float32)
                    x_min = int(np.min(points[:, 0]))
                    y_min = int(np.min(points[:, 1]))
                    x_max = int(np.max(points[:, 0]))
                    y_max = int(np.max(points[:, 1]))
                else:
                    coords = [float(coord) for coord in bbox[:4]]
                    x_min, y_min, x_max, y_max = map(int, coords)
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Ошибка парсинга bbox {bbox}: {str(e)}")
                return {
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence,
                    'width': 10,
                    'height': 10,
                    'area': 100,
                    'font_size_estimate': 12,
                    'is_invalid': True
                }
            
            # Извлекаем область изображения
            region = image[y_min:y_max, x_min:x_max]
            
            if region.size == 0:
                return {
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence,
                    'width': 0,
                    'height': 0,
                    'area': 0,
                    'font_size_estimate': 0
                }
            
            # Базовые характеристики области
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            font_size_estimate = height * 0.7
            
            return {
                'bbox': bbox,
                'text': text,
                'confidence': confidence,
                'width': width,
                'height': height,
                'area': area,
                'font_size_estimate': font_size_estimate,
                'region': region
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа области текста: {str(e)}")
            return {
                'bbox': bbox,
                'text': text,
                'confidence': confidence,
                'width': 0,
                'height': 0,
                'area': 0,
                'font_size_estimate': 0
            }
    
    def _detect_multiple_fonts_from_regions(self, text_regions: List[Dict]) -> bool:
        """Определение множественных шрифтов на основе анализа областей"""
        try:
            if len(text_regions) < 2:
                return False
            
            # Собираем размеры шрифтов
            font_sizes = [region.get('font_size_estimate', 0) for region in text_regions]
            font_sizes = [size for size in font_sizes if size > 0]
            
            if len(font_sizes) < 2:
                return False
            
            # Анализируем разброс размеров
            font_sizes = np.array(font_sizes)
            mean_size = np.mean(font_sizes)
            std_size = np.std(font_sizes)
            
            variation_ratio = std_size / mean_size if mean_size > 0 else 0
            
            logger.info(f"📏 Анализ размеров шрифтов: среднее={mean_size:.1f}, отклонение={std_size:.1f}, коэффициент={variation_ratio:.2f}")
            
            # Получаем конфигурацию детекции множественных шрифтов
            fonts_config = get_multiple_fonts_config()
            
            # Менее чувствительный порог для PaddleOCR
            if variation_ratio > fonts_config['size_variation_threshold']:
                logger.info("🔤 Обнаружены множественные шрифты (разные размеры)")
                return True
            
            # Дополнительная проверка по площади областей
            areas = [region.get('area', 0) for region in text_regions]
            areas = [area for area in areas if area > 0]
            
            if len(areas) >= 2:
                areas = np.array(areas)
                area_ratio = np.max(areas) / np.min(areas) if np.min(areas) > 0 else 0
                
                if area_ratio > fonts_config['area_ratio_threshold']:
                    logger.info("🔤 Обнаружены множественные шрифты (разные площади)")
                    return True
            
            # Дополнительная проверка: если много областей текста
            if len(text_regions) >= fonts_config['min_regions_count']:
                logger.info(f"🔤 Обнаружены множественные шрифты (много областей: {len(text_regions)})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка определения множественных шрифтов: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Проверка доступности PaddleOCR"""
        try:
            # Проверяем все компоненты
            library_available = PADDLEOCR_AVAILABLE
            object_created = self.ocr is not None
            
            # Дополнительная проверка - если объект есть, пробуем его использовать
            object_working = False
            if object_created:
                try:
                    # Быстрая проверка - просто проверяем что объект не сломан
                    object_working = hasattr(self.ocr, 'predict') and callable(getattr(self.ocr, 'predict', None))
                except Exception as check_error:
                    logger.error(f"❌ Ошибка проверки работоспособности объекта: {str(check_error)}")
                    object_working = False
            
            available = library_available and object_created and object_working
            
            logger.info(f"🔍 PaddleOCR диагностика:")
            logger.info(f"  - Библиотека доступна: {library_available}")
            logger.info(f"  - Объект создан: {object_created}")
            logger.info(f"  - Объект рабочий: {object_working}")
            logger.info(f"  - ИТОГО доступен: {available}")
            
            if not available:
                if not library_available:
                    logger.error("❌ PaddleOCR библиотека не установлена")
                elif not object_created:
                    logger.error("❌ PaddleOCR объект не создан")
                elif not object_working:
                    logger.error("❌ PaddleOCR объект создан, но не работает")
            
            return available
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки доступности PaddleOCR: {str(e)}")
            return False
    
    def reinitialize(self) -> bool:
        """Принудительная переинициализация PaddleOCR"""
        try:
            logger.info("🔄 Принудительная переинициализация PaddleOCR...")
            
            # Очищаем старый объект
            if self.ocr:
                logger.info("🗑️ Очищаем старый объект PaddleOCR...")
                self.ocr = None
            
            # Переинициализируем
            self._initialize_ocr()
            
            # Проверяем результат
            is_available = self.is_available()
            
            if is_available:
                logger.info("✅ PaddleOCR успешно переинициализирован")
            else:
                logger.error("❌ PaddleOCR не удалось переинициализировать")
            
            return is_available
            
        except Exception as e:
            logger.error(f"❌ Ошибка переинициализации PaddleOCR: {str(e)}")
            logger.error(f"💡 Тип ошибки: {type(e).__name__}")
            return False

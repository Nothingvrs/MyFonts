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
        self.ocr_loose = None
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
                # СТАБИЛЬНАЯ конфигурация по умолчанию — доверяем PaddleOCR выбор моделей
                safe_config = {
                    'lang': ocr_config.get('lang', 'ru'),
                    'use_angle_cls': True,
                }
                ocr_config = safe_config
                logger.info("🔄 Используем СУПЕР агрессивную конфигурацию")
                
            except Exception as config_error:
                logger.error(f"❌ Ошибка загрузки конфигурации OCR: {str(config_error)}")
                logger.info("🔄 Используем СУПЕР агрессивную конфигурацию по умолчанию...")
                ocr_config = {
                    'lang': 'ru',
                    'use_angle_cls': True,
                }
            
            logger.info("🚀 Инициализация PaddleOCR с переданной конфигурацией...")
            # Логируем только доступные ключи
            for k in ['lang', 'use_angle_cls', 'det_db_thresh', 'det_db_box_thresh', 'det_db_unclip_ratio', 'det_limit_side_len', 'det_limit_type']:
                if k in ocr_config:
                    logger.info(f"  - {k}: {ocr_config.get(k)}")
            
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
                    minimal_config = {'lang': 'ru'}
                    logger.info(f"📋 Минимальная конфигурация: {minimal_config}")
                    self.ocr = PaddleOCR(**minimal_config)
                    logger.info("✅ PaddleOCR создан с минимальной конфигурацией")
                except Exception as minimal_error:
                    logger.error(f"❌ Ошибка создания с минимальной конфигурацией: {str(minimal_error)}")
                    
                    # Последняя попытка - только базовые параметры
                    logger.info("🔄 Последняя попытка - только базовые параметры...")
                    try:
                        basic_config = {'lang': 'ru'}
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
                if not hasattr(self.ocr, 'ocr') or not callable(getattr(self.ocr, 'ocr', None)):
                    logger.error("❌ У объекта PaddleOCR нет метода ocr")
                    self.ocr = None
                    return
                
                logger.info("✅ Метод ocr найден, делаем тестовый вызов...")
                # PaddleOCR 3.x: ocr(img) без аргументов
                test_result = self.ocr.ocr(test_image)
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
    
    async def analyze_image(self, image: np.ndarray, sensitivity: Optional[str] = None) -> Dict[str, Any]:
        """Алиас для обратной совместимости"""
        return await self.detect_and_analyze_text(image, sensitivity=sensitivity)
    
    async def detect_and_analyze_text(self, image: np.ndarray, sensitivity: Optional[str] = None) -> Dict[str, Any]:
        """Детекция и анализ текста на изображении"""
        print("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: detect_and_analyze_text НАЧАЛСЯ")
        logger.info("🚀 === НАЧАЛО detect_and_analyze_text ===")
        print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Изображение: {image.shape}, {image.dtype}")
        logger.info(f"🖼️ Получено изображение: {image.shape}, {image.dtype}")
        
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
        
        logger.info("✅ PaddleOCR доступен, запускаем анализ...")
        
        try:
            # Запускаем OCR в отдельном потоке
            logger.info("🔄 Запускаем _run_ocr_sync в отдельном потоке...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_ocr_sync,
                image
            )
            logger.info(f"✅ _run_ocr_sync завершен, результат: {type(result)}")
            logger.info(f"🔍 Результат: {repr(result)}")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: detect_and_analyze_text завершен, результат: {type(result)}")
            logger.info("🚀 === КОНЕЦ detect_and_analyze_text ===")
            return result
            
        except Exception as e:
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: ОШИБКА в detect_and_analyze_text: {str(e)}")
            logger.error(f"💥 Техническая ошибка PaddleOCR: {str(e)}")
            logger.error(f"💡 Тип ошибки: {type(e).__name__}")
            logger.error(f"🔍 Детали ошибки: {repr(e)}")
            logger.error("🚀 === КОНЕЦ detect_and_analyze_text (С ОШИБКОЙ) ===")
            return {
                'has_text': False,
                'text_regions': [],
                'multiple_fonts': False,
                'confidence': 0.0,
                'text_content': '',
                'error': f"Техническая ошибка OCR: {str(e)}"
            }
    
    def _get_loose_ocr(self):
        """Возвращает основной OCR вместо создания второго экземпляра.
        На некоторых окружениях создание второго объекта может приводить к
        загрузке дополнительных doc-моделей и значительным задержкам.
        """
        if not PADDLEOCR_AVAILABLE:
            return None
        # Используем уже инициализированный основной OCR
        return self.ocr

    def _create_image_variants(self, image: np.ndarray) -> List[np.ndarray]:
        """Создание 10 самых эффективных вариантов изображения для агрессивного поиска текста"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            variants = []
            
            # 1. Оригинальное изображение (RGB)
            variants.append(image.copy())
            # 1b. Уменьшенная копия до 1536 по длинной стороне (улучшает распознавание крупных баннеров)
            try:
                base = image.copy()
                h0, w0 = (base.shape[:2] if len(base.shape) >= 2 else (0, 0))
                if min(h0, w0) > 0:
                    scale = 1536.0 / max(h0, w0)
                    if scale < 1.0:
                        new_w = int(w0 * scale)
                        new_h = int(h0 * scale)
                        resized_down = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        variants.append(resized_down)
            except Exception:
                pass
            
            # 2. Увеличенное изображение (для мелкого текста)
            try:
                h, w = gray.shape
                if min(h, w) < 800:
                    scale = max(4, 1000 // min(h, w))
                    if len(image.shape) == 3:
                        resized = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        resized_gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
                        resized = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)
                    variants.append(resized)
            except:
                pass

            # 9b. Black-hat трансформация для акцента на тёмном тексте на светлом фоне
            try:
                kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)
                blackhat_rgb = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2RGB)
                variants.append(blackhat_rgb)
            except:
                pass
            
            # 3. Экстремальное увеличение для очень мелкого текста
            try:
                h, w = gray.shape
                if min(h, w) < 400:
                    scale = max(6, 1200 // min(h, w))
                    if len(image.shape) == 3:
                        resized_extreme = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        resized_gray_extreme = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
                        resized_extreme = cv2.cvtColor(resized_gray_extreme, cv2.COLOR_GRAY2RGB)
                    variants.append(resized_extreme)
            except:
                pass
            
            # 4. Высокий контраст (умеренный)
            try:
                enhanced = cv2.convertScaleAbs(gray, alpha=1.6, beta=10)
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                variants.append(enhanced_rgb)
            except:
                pass
            
            # 5. CLAHE (адаптивная эквализация)
            try:
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
                clahe_image = clahe.apply(gray)
                clahe_rgb = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
                variants.append(clahe_rgb)
            except:
                pass
            
            # 6. Адаптивная бинаризация (более мягкие параметры)
            try:
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)
                adaptive_rgb = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)
                variants.append(adaptive_rgb)
            except:
                pass
            
            # 7. Инверсия (для белого текста на темном фоне) + бинаризация
            try:
                inverted = cv2.bitwise_not(gray)
                inv_bin = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 7)
                inverted_rgb = cv2.cvtColor(inv_bin, cv2.COLOR_GRAY2RGB)
                variants.append(inverted_rgb)
            except:
                pass
            
            # 8. Otsu-бинаризация (умеренная)
            try:
                _, extreme_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                extreme_binary_rgb = cv2.cvtColor(extreme_binary, cv2.COLOR_GRAY2RGB)
                variants.append(extreme_binary_rgb)
            except:
                pass
            
            # 9. Комбинированная обработка
            try:
                combined = cv2.convertScaleAbs(gray, alpha=2.5, beta=60)
                combined = cv2.GaussianBlur(combined, (3, 3), 0)
                combined = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
                combined_rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
                variants.append(combined_rgb)
            except:
                pass
            
            # 10. Предобработка для мелких надписей (умеренно)
            try:
                h, w = gray.shape
                if min(h, w) < 500:
                    scale = 2
                    large_gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
                    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(large_gray)
                    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                    variants.append(binary_rgb)
            except:
                pass

            # 11. Усиление чёрного тонкого текста: медианный блюр + адаптивный порог + морф.замыкание
            try:
                blur = cv2.medianBlur(gray, 3)
                th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
                kernel = np.ones((2, 2), np.uint8)
                closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
                closed_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
                variants.append(closed_rgb)
            except:
                pass

            # 12. Инвертированная Otsu + дилатация для тонких чёрных букв
            try:
                _, th_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = np.ones((2, 2), np.uint8)
                dil = cv2.dilate(th_inv, kernel, iterations=1)
                dil_rgb = cv2.cvtColor(dil, cv2.COLOR_GRAY2RGB)
                variants.append(dil_rgb)
            except:
                pass

            # 13. Подавление красных областей (чтобы выделить чёрный текст)
            try:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                h, s, v = cv2.split(hsv)
                # Маски красного (две дуги по кругу оттенков)
                lower_red1 = np.array([0, 80, 40], dtype=np.uint8)
                upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
                lower_red2 = np.array([170, 80, 40], dtype=np.uint8)
                upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)
                # Заменяем красные пиксели на белые
                no_red = image.copy()
                no_red[red_mask > 0] = [255, 255, 255]
                variants.append(no_red)

                # На результате без красного — дополнительная адаптивная бинаризация
                no_red_gray = cv2.cvtColor(no_red, cv2.COLOR_RGB2GRAY)
                nr_th = cv2.adaptiveThreshold(no_red_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
                nr_th_rgb = cv2.cvtColor(nr_th, cv2.COLOR_GRAY2RGB)
                variants.append(nr_th_rgb)

                # Увеличение no_red для тонких подписей + CLAHE + бинаризация (сильный режим)
                try:
                    up_scale = 3
                    h0, w0 = no_red_gray.shape
                    up = cv2.resize(no_red_gray, (w0 * up_scale, h0 * up_scale), interpolation=cv2.INTER_LANCZOS4)
                    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
                    up_enh = clahe.apply(up)
                    up_th = cv2.adaptiveThreshold(up_enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 5)
                    up_th_rgb = cv2.cvtColor(up_th, cv2.COLOR_GRAY2RGB)
                    variants.append(up_th_rgb)
                except Exception:
                    pass
            except:
                pass

            # 14. Маска «только тёмные пиксели» + закрытие
            try:
                dark = (gray < 160).astype(np.uint8) * 255
                kernel = np.ones((2, 2), np.uint8)
                dark_closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
                dark_closed_rgb = cv2.cvtColor(dark_closed, cv2.COLOR_GRAY2RGB)
                variants.append(dark_closed_rgb)
            except:
                pass

            # 15. Unsharp mask для усиления тонких штрихов
            try:
                blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
                unsharp = cv2.addWeighted(image, 1.6, blur, -0.6, 0)
                variants.append(unsharp)
            except:
                pass
            
            logger.info(f"✅ Создано {len(variants)} вариантов для OCR")
            return variants
            
        except Exception as e:
            logger.error(f"Ошибка создания вариантов: {str(e)}")
            return [image.copy()]
    
    def _run_ocr_sync(self, image: np.ndarray) -> Dict[str, Any]:
        """Синхронный запуск OCR с максимально агрессивными настройками"""
        print("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: _run_ocr_sync НАЧАЛСЯ")
        logger.info("🚀 === НАЧАЛО _run_ocr_sync ===")
        try:
            print("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Запуск PaddleOCR анализа...")
            logger.info("🔍 Запуск PaddleOCR анализа...")
            logger.info(f"🖼️ Размер изображения: {image.shape}")
            logger.info(f"🖼️ Тип данных изображения: {image.dtype}")
            logger.info(f"🖼️ Диапазон значений пикселей: [{image.min()}, {image.max()}]")
            
            # Создаем варианты изображения
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Создаем варианты изображения...")
            image_variants = self._create_image_variants(image)
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Создано {len(image_variants)} вариантов")
            logger.info(f"🔄 Создано {len(image_variants)} вариантов изображения")
            
            # Собираем ВСЕ найденные тексты из всех вариантов за один проход
            all_texts = []
            all_bboxes = []
            all_confidences = []
            
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: === СБОР ТЕКСТОВ ИЗ ВСЕХ ВАРИАНТОВ ===")
            
            for i, variant in enumerate(image_variants):
                try:
                    print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Обрабатываем вариант #{i+1}/{len(image_variants)}")
                    logger.info(f"🔍 Попытка OCR #{i+1}/{len(image_variants)}")
                    logger.info(f"  - Размер варианта: {variant.shape}")
                    
                    # Вызываем PaddleOCR
                    print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Вариант #{i+1}: вызываем PaddleOCR.ocr()...")
                    # PaddleOCR ожидает изображение в BGR (как из cv2.imread)
                    variant_input = variant
                    try:
                        if isinstance(variant, np.ndarray) and len(variant.shape) == 3 and variant.shape[2] == 3:
                            # наши варианты, как правило, в RGB → конвертируем в BGR
                            variant_input = cv2.cvtColor(variant, cv2.COLOR_RGB2BGR)
                    except Exception:
                        variant_input = variant
                    # PaddleOCR 3.x: ocr(img) без дополнительных аргументов
                    variant_result = self.ocr.ocr(variant_input)
                    
                    print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Вариант #{i+1}: PaddleOCR вернул: {type(variant_result)}")
                    logger.info(f"🔍 Вариант #{i+1}: результат PaddleOCR: {type(variant_result)}")
                    
                    # НОРМАЛИЗУЕМ РЕЗУЛЬТАТ ПОД ВСЕ СИГНАТУРЫ
                    parsed = self._normalize_ocr_result(variant_result)
                    if not parsed:
                        print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Вариант #{i+1}: распознанных строк нет")
                    for j, item in enumerate(parsed):
                        try:
                            text = str(item.get('text', '')).strip()
                            conf = float(item.get('confidence', 0.0))
                            bbox = item.get('bbox', [[0,0],[100,0],[100,100],[0,100]])
                            if text and conf > 0:
                                all_texts.append(text)
                                all_bboxes.append(bbox)
                                all_confidences.append(conf)
                                logger.info(f"🔍 Вариант #{i+1}: добавлен текст '{text}' (уверенность: {conf:.3f})")
                        except Exception as detection_error:
                            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: ОШИБКА при обработке детекции #{j+1}: {str(detection_error)}")
                            continue
                        
                except Exception as e:
                    print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: ОШИБКА обработки варианта #{i+1}: {str(e)}")
                    logger.warning(f"⚠️ Ошибка обработки варианта #{i+1}: {str(e)}")
                    continue
            
            # ДОПОЛНИТЕЛЬНЫЙ ПРОХОД временно отключён для ускорения первого ответа
            # try:
            #     extra_texts, extra_bboxes, extra_confs = self._detect_black_text_lines(image)
            #     for txt, bb, cf in zip(extra_texts, extra_bboxes, extra_confs):
            #         all_texts.append(txt)
            #         all_bboxes.append(bb)
            #         all_confidences.append(cf)
            #     logger.info(f"➕ Extra pass (dark lines): добавлено {len(extra_texts)} строк")
            # except Exception as extra_err:
            #     logger.warning(f"⚠️ Extra pass (dark lines) ошибка: {str(extra_err)}")

            # Убираем дубликаты, оставляя лучшую уверенность для каждого уникального текста
            unique_texts = {}
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: === ОБЪЕДИНЕНИЕ РЕЗУЛЬТАТОВ ===")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: all_texts: {all_texts}")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: all_bboxes: {all_bboxes}")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: all_confidences: {all_confidences}")
            
            for i, (text, bbox, conf) in enumerate(zip(all_texts, all_bboxes, all_confidences)):
                print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Обрабатываем элемент #{i+1}: text='{text}', bbox={repr(bbox)}, conf={conf}")
                try:
                    if text not in unique_texts or conf > unique_texts[text]['confidence']:
                        unique_texts[text] = {'bbox': bbox, 'confidence': conf}
                        print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Добавлен/обновлен: '{text}' -> уверенность {conf:.3f}")
                except Exception as e:
                    print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: ОШИБКА при обработке элемента #{i+1}: {str(e)}")
                    logger.error(f"❌ Ошибка при обработке элемента #{i+1}: {str(e)}")
                    continue
            
            logger.info(f"✅ Собрано {len(unique_texts)} уникальных текстов из всех вариантов")
            
            # Детальное логирование уникальных текстов
            for i, (text, info) in enumerate(unique_texts.items()):
                logger.info(f"  📝 Текст #{i+1}: '{text}' (уверенность: {info['confidence']:.3f})")
                print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Уникальный текст #{i+1}: '{text}' (уверенность: {info['confidence']:.3f})")
            
            # Создаем объединенный результат (или пустой список для дальнейшей диагностики)
            if len(unique_texts) == 0:
                logger.info("ℹ️ OCR не выделил уникальные строки, продолжаем с пустым результатом для диагностики")
                ocr_result = []
            else:
                ocr_result = [[unique_texts[text]['bbox'], [text, unique_texts[text]['confidence']]] 
                             for text in unique_texts.keys()]
            
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: === ИТОГОВЫЙ ОБЪЕДИНЕННЫЙ РЕЗУЛЬТАТ ===")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Всего уникальных текстов: {len(unique_texts)}")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: ocr_result содержит {len(ocr_result)} элементов")
            
            # Обрабатываем результат (итерация по строкам)
            logger.info(f"🔍 Обрабатываем финальный результат: элементов={len(ocr_result)}")
            text_regions = []
            all_text = []
            confidences = []
            
            for i, line in enumerate(ocr_result):
                try:
                    if isinstance(line, dict):
                        # Не ожидается после унификации, оставлено для совместимости
                        text = str(line.get('text', '')).strip()
                        confidence = float(line.get('confidence', 0.0))
                        bbox = line.get('bbox', [[0, 0], [100, 0], [100, 100], [0, 100]])
                    elif isinstance(line, (list, tuple)) and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0])
                            confidence = float(text_info[1])
                        elif isinstance(text_info, str):
                            text = text_info
                            confidence = 1.0
                        else:
                            continue
                    else:
                        continue
                    if not text:
                        continue
                    region_info = self._analyze_text_region(image, bbox, text, confidence)
                    text_regions.append(region_info)
                    all_text.append(text)
                    confidences.append(confidence)
                except Exception as parse_err:
                    logger.warning(f"⚠️ Ошибка разбора строки #{i+1}: {str(parse_err)}")
                    continue
            
            # Статистика и проверка качества
            avg_confidence = np.mean(confidences) if confidences else 0.0
            text_content = ' '.join(all_text)
            
            # Получаем конфигурацию качества текста
            quality_config = get_text_quality_config()
            
            # Улучшенная проверка качества текста (не роняемся при пустых bbox)
            valid_regions = []
            for r in text_regions:
                try:
                    if r.get('confidence', 0) >= quality_config['min_confidence']:
                        valid_regions.append(r)
                except Exception:
                    continue
            clean_text = ''.join(c for c in text_content if c.isalnum() or c.isspace()).strip()
            
            # ДЕТАЛЬНАЯ ДИАГНОСТИКА ПРОВЕРКИ КАЧЕСТВА
            logger.info(f"=== ДИАГНОСТИКА КАЧЕСТВА ТЕКСТА ===")
            logger.info(f"Всего областей: {len(text_regions)}")
            logger.info(f"Валидных областей: {len(valid_regions)}")
            logger.info(f"Чистый текст: '{clean_text}' (длина: {len(clean_text)})")
            logger.info(f"Средняя уверенность: {avg_confidence:.3f}")
            logger.info(f"Пороги: min_confidence={quality_config['min_confidence']}, min_text_length={quality_config['min_text_length']}, min_avg_confidence={quality_config['min_avg_confidence']}")
            
            # Проверяем каждое условие отдельно для лучшей диагностики
            cond1 = len(valid_regions) >= quality_config.get('min_regions_count', 1)
            cond2 = len(clean_text) >= quality_config['min_text_length']
            cond3 = avg_confidence >= quality_config['min_avg_confidence']
            cond4 = len(text_regions) > 0  # Базовая проверка наличия областей
            
            logger.info(f"Условие 1 (валидные области >= {quality_config.get('min_regions_count', 1)}): {cond1}")
            logger.info(f"Условие 2 (длина текста >= {quality_config['min_text_length']}): {cond2}")
            logger.info(f"Условие 3 (средняя уверенность >= {quality_config['min_avg_confidence']}): {cond3}")
            logger.info(f"Условие 4 (есть области текста): {cond4}")
            
            # Детали по каждой области для отладки
            for i, region in enumerate(text_regions):
                conf = region.get('confidence', 0)
                text = region.get('text', '')
                logger.info(f"Область #{i+1}: '{text}' confidence={conf:.3f}, проходит порог={conf >= quality_config['min_confidence']}")
            
            # Жесткая проверка наличия текста: должны сойтись базовые условия И достаточное количество букв
            # Опираемся на конфиг качества
            try:
                min_letters = int(quality_config.get('min_letters_count', 3))
            except Exception:
                min_letters = 3
            letters_count = sum(1 for c in clean_text if c.isalpha())
            has_text = cond4 and cond1 and cond2 and cond3 and letters_count >= min_letters
            
            logger.info(f"ИТОГОВЫЙ РЕЗУЛЬТАТ has_text = {has_text}")
            
            logger.info(f"🔍 Проверка качества: всего областей={len(text_regions)}, валидных={len(valid_regions)}")
            logger.info(f"📝 Чистый текст: '{clean_text[:50]}' (длина: {len(clean_text)})")
            logger.info(f"📊 Средняя уверенность: {avg_confidence:.2f}")
            logger.info(f"✅ Результат проверки: has_text={has_text} (letters={letters_count})")
            
            # ДЕТАЛЬНАЯ ДИАГНОСТИКА МНОЖЕСТВЕННЫХ ШРИФТОВ
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: === ДИАГНОСТИКА МНОЖЕСТВЕННЫХ ШРИФТОВ ===")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Анализируем {len(text_regions)} областей текста...")
            
            # Определяем множественные шрифты
            # Читаем конфиг чувствительности
            try:
                cfg = get_multiple_fonts_config(mode=sensitivity) if sensitivity else get_multiple_fonts_config()
            except Exception:
                cfg = get_multiple_fonts_config()
            multiple_fonts = self._detect_multiple_fonts_from_regions(text_regions)
            
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Результат анализа множественных шрифтов: {multiple_fonts}")
            
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
            logger.info(f"🔤 Результат множественных шрифтов: {multiple_fonts}")
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: _run_ocr_sync завершен, has_text={has_text}, multiple_fonts={multiple_fonts}")
            logger.info("🚀 === КОНЕЦ _run_ocr_sync ===")
            return result
            
        except Exception as e:
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: ОШИБКА в _run_ocr_sync: {str(e)}")
            logger.error(f"💥 Техническая ошибка в _run_ocr_sync: {str(e)}")
            logger.error(f"💡 Тип ошибки: {type(e).__name__}")
            logger.error(f"🔍 Детали ошибки: {repr(e)}")
            logger.error("🚀 === КОНЕЦ _run_ocr_sync (С ОШИБКОЙ) ===")
            raise

    def _detect_black_text_lines(self, image: np.ndarray) -> Tuple[List[str], List[List[List[int]]], List[float]]:
        """Поиск чёрного тонкого текста: вырезаем горизонтальные полосы и гоняем OCR по кропам.
        Возвращает списки (texts, bboxes, confidences). BBox задаём как прямоугольник кропа в координатах исходного изображения.
        """
        texts: List[str] = []
        bboxes: List[List[List[int]]] = []
        confs: List[float] = []

        # 1) Убираем красный, чтобы не мешал
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lower_red1 = np.array([0, 80, 40], dtype=np.uint8)
            upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
            lower_red2 = np.array([170, 80, 40], dtype=np.uint8)
            upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            no_red = image.copy()
            no_red[red_mask > 0] = [255, 255, 255]
        except Exception:
            no_red = image.copy()

        gray = cv2.cvtColor(no_red, cv2.COLOR_RGB2GRAY)

        # 2) Сильное усиление чёрного
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        enh = clahe.apply(gray)
        th = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 5)
        # Доп. маска K-канала (тёмные пиксели)
        try:
            k_channel = 255 - np.max(no_red, axis=2)
            k_channel = k_channel.astype(np.uint8)
            k_blur = cv2.GaussianBlur(k_channel, (3, 3), 0)
            k_th = cv2.adaptiveThreshold(k_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
        except Exception:
            k_th = np.zeros_like(th)
        # Убираем шум, соединяем символы в полосы (объединённая маска)
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed_k = cv2.morphologyEx(k_th, cv2.MORPH_CLOSE, kernel, iterations=1)
        line_mask = cv2.bitwise_or(closed, closed_k)
        # Усилим горизонтальные линии для более уверенной проекции
        try:
            h_kernel = np.ones((1, 9), np.uint8)
            line_mask = cv2.dilate(line_mask, h_kernel, iterations=1)
        except Exception:
            pass

        # 3) Горизонтальная проекция для поиска полос
        proj = np.sum(line_mask > 0, axis=1)
        h, w = closed.shape
        line_threshold = max(8, int(0.015 * w))
        bands: List[Tuple[int, int]] = []
        in_band = False
        band_start = 0
        for y in range(h):
            if proj[y] >= line_threshold and not in_band:
                in_band = True
                band_start = y
            elif proj[y] < line_threshold and in_band:
                in_band = False
                bands.append((band_start, y))
        if in_band:
            bands.append((band_start, h - 1))

        # 4) Для каждой полосы получаем кроп, увеличиваем и прогоняем OCR
        for (y1, y2) in bands:
            # защитимся от слишком тонких полос
            if y2 - y1 < 10:
                continue
            pad = 4
            y1p = max(0, y1 - pad)
            y2p = min(h - 1, y2 + pad)
            crop_rgb = no_red[y1p:y2p, :, :]
            if crop_rgb.size == 0:
                continue
            # upscale
            scale = 4
            crop_up = cv2.resize(crop_rgb, (crop_rgb.shape[1] * scale, crop_rgb.shape[0] * scale), interpolation=cv2.INTER_LANCZOS4)
            # OCR (BGR)
            try:
                # Доп. предобработка: L-канал LAB + адаптивная бинаризация (инверт.)
                try:
                    lab = cv2.cvtColor(crop_up, cv2.COLOR_RGB2LAB)
                    l = lab[:, :, 0]
                    clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l_enh = clahe2.apply(l)
                    l_th = cv2.adaptiveThreshold(l_enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
                    crop_pre = cv2.cvtColor(l_th, cv2.COLOR_GRAY2RGB)
                except Exception:
                    crop_pre = crop_up

                crop_bgr = cv2.cvtColor(crop_pre, cv2.COLOR_RGB2BGR)
                ocr_engine = self._get_loose_ocr() or self.ocr
                ocr_res = ocr_engine.ocr(crop_bgr)
                parsed = self._normalize_ocr_result(ocr_res)
                # Трансформируем bbox из координат кропа (после апскейла) в координаты исходного изображения
                for item in parsed:
                    text = str(item.get('text', '')).strip()
                    conf = float(item.get('confidence', 0.0))
                    raw_bbox = item.get('bbox')
                    try:
                        transformed_bbox = []
                        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4 and isinstance(raw_bbox[0], (list, tuple)):
                            for pt in raw_bbox:
                                x = int(pt[0] / float(scale))
                                y = int(pt[1] / float(scale)) + int(y1p)
                                transformed_bbox.append([x, y])
                        else:
                            # аварийно используем границы полосы
                            transformed_bbox = [[0, y1p], [w - 1, y1p], [w - 1, y2p], [0, y2p]]
                    except Exception:
                        transformed_bbox = [[0, y1p], [w - 1, y1p], [w - 1, y2p], [0, y2p]]

                    # Фильтр: кириллица + мягкий порог уверенности + отбрасываем очень короткие токены
                    has_cyr = any(1040 <= ord(c) <= 1103 for c in text)
                    if text and len(text) >= 3 and has_cyr and conf >= 0.45:
                        texts.append(text)
                        bboxes.append(transformed_bbox)
                        confs.append(conf)
            except Exception:
                continue

        return texts, bboxes, confs

    def _normalize_ocr_result(self, raw: Any) -> List[Dict[str, Any]]:
        """Приводит результат PaddleOCR (2.x/3.x, разные форматы) к унифицированному виду.
        Возвращает список элементов: { bbox: [...], text: str, confidence: float }
        """
        try:
            if raw is None:
                return []
            # Ожидаемый формат 3.x: список страниц; берём первую
            if isinstance(raw, list) and len(raw) > 0:
                first = raw[0]
                # Формат 2.x: список записей [[[x,y]x4], (text, score)]
                if isinstance(first, list):
                    normalized: List[Dict[str, Any]] = []
                    for det in first:
                        try:
                            if not isinstance(det, (list, tuple)) or len(det) < 2:
                                continue
                            bbox = det[0]
                            text_info = det[1]
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = str(text_info[0])
                                conf = float(text_info[1])
                            elif isinstance(text_info, str):
                                text = text_info
                                conf = 1.0
                            else:
                                continue
                            normalized.append({'bbox': bbox, 'text': text, 'confidence': conf})
                        except Exception:
                            continue
                    return normalized
                # Альтернативный формат: dict с ключами rec_texts/rec_scores/dt_polys
                if isinstance(first, dict):
                    rec_texts = first.get('rec_texts', [])
                    rec_scores = first.get('rec_scores', [])
                    dt_polys = first.get('dt_polys', [])
                    out: List[Dict[str, Any]] = []
                    for i, text in enumerate(rec_texts):
                        try:
                            conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                            bbox = dt_polys[i] if i < len(dt_polys) else [[0,0],[100,0],[100,100],[0,100]]
                            # Приводим bbox к списку списков, если пришёл numpy
                            try:
                                import numpy as _np
                                if isinstance(bbox, _np.ndarray):
                                    bbox = bbox.tolist()
                            except Exception:
                                pass
                            out.append({'bbox': bbox, 'text': str(text), 'confidence': conf})
                        except Exception:
                            continue
                    return out
            return []
        except Exception:
            return []
    
    def _analyze_text_region(self, image: np.ndarray, bbox: List, text: str, confidence: float) -> Dict[str, Any]:
        """Анализ отдельной области текста"""
        try:
            # Безопасное извлечение координат
            try:
                logger.info(f"🔍 Парсим bbox: {repr(bbox)}, тип: {type(bbox)}")
                
                if isinstance(bbox, (list, tuple)) and len(bbox) > 0:
                    # НОВЫЙ ФОРМАТ: bbox может быть списком координат в разных форматах
                    if isinstance(bbox[0], (list, tuple)):
                        # Формат: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        points = np.array(bbox, dtype=np.float32)
                        x_min = int(np.min(points[:, 0]))
                        y_min = int(np.min(points[:, 1]))
                        x_max = int(np.max(points[:, 0]))
                        y_max = int(np.max(points[:, 1]))
                        logger.info(f"✅ Парсинг bbox: [[x,y], [x,y], [x,y], [x,y]] -> x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    elif len(bbox) >= 4:
                        # Формат: [x1, y1, x2, y2] или [x1, y1, x2, y2, ...]
                        coords = [float(coord) for coord in bbox[:4]]
                        x_min, y_min, x_max, y_max = map(int, coords)
                        logger.info(f"✅ Парсинг bbox: [x1, y1, x2, y2] -> x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    elif len(bbox) == 2:
                        # Формат: [x, y] - одна точка, создаем область вокруг неё
                        x_min = int(float(bbox[0])) - 10
                        y_min = int(float(bbox[1])) - 10
                        x_max = int(float(bbox[0])) + 10
                        y_max = int(float(bbox[1])) + 10
                        logger.info(f"✅ Парсинг bbox: [x, y] -> создаем область вокруг точки: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    else:
                        # Неизвестный формат, создаем область по умолчанию
                        x_min, y_min, x_max, y_max = 0, 0, 100, 100
                        logger.warning(f"⚠️ Неизвестный формат bbox {bbox}, используем область по умолчанию")
                else:
                    # bbox пустой или None, создаем область по умолчанию
                    x_min, y_min, x_max, y_max = 0, 0, 100, 100
                    logger.warning(f"⚠️ bbox пустой или None, используем область по умолчанию")
                
                # Проверяем валидность координат
                if x_min >= x_max or y_min >= y_max:
                    logger.warning(f"⚠️ Некорректные координаты: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    # Исправляем координаты
                    if x_min >= x_max:
                        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                        if x_min == x_max:
                            x_max = x_min + 100
                    if y_min >= y_max:
                        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
                        if y_min == y_max:
                            y_max = y_min + 100
                    logger.info(f"✅ Исправлены координаты: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                    
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"❌ Ошибка парсинга bbox {bbox}: {str(e)}")
                logger.error(f"💡 Тип ошибки: {type(e).__name__}")
                logger.error(f"🔍 Детали: {repr(e)}")
                # Создаем область по умолчанию вместо пометки как невалидной
                x_min, y_min, x_max, y_max = 0, 0, 100, 100
                logger.info(f"🔄 Используем область по умолчанию: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            
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
                    'font_size_estimate': 0,
                    'region': region,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
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
                'region': region,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
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
                'font_size_estimate': 0,
                'x_min': 0,
                'y_min': 0,
                'x_max': 0,
                'y_max': 0
            }
    
    def _detect_multiple_fonts_from_regions(self, text_regions: List[Dict]) -> bool:
        """Робастное определение множественных шрифтов. Менее чувствительно к шуму."""
        try:
            logger.info("=== АНАЛИЗ МНОЖЕСТВЕННЫХ ШРИФТОВ (ROBUST) ===")
            logger.info(f"Всего областей для анализа: {len(text_regions)}")

            if len(text_regions) < 2:
                logger.info("Областей < 2 — считаем один шрифт")
                return False

            cfg = get_multiple_fonts_config()

            # 0) Жёсткая фильтрация шумов
            filtered: List[Dict[str, Any]] = []
            for r in text_regions:
                txt = str(r.get('text', '')).strip()
                conf = float(r.get('confidence', 0.0))
                h = float(r.get('height', 0) or 0)
                w = float(r.get('width', 0) or 0)
                if len(txt) >= 2 and conf >= 0.7 and h > 8 and w > 8:
                    filtered.append(r)
            logger.info(f"После фильтрации осталось регионов: {len(filtered)}")
            if len(filtered) < max(5, int(cfg.get('min_regions_count', 4))):
                logger.info("Данных мало после фильтрации — один шрифт")
                return False

            # Удаляем экстремальные по ширине/площади (частая причина ложных срабатываний)
            try:
                widths = [float(r.get('width', 0)) for r in filtered if float(r.get('width', 0)) > 0]
                areas_for_outliers = [float(r.get('area', 0)) for r in filtered if float(r.get('area', 0)) > 0]
                if widths:
                    w_med = float(np.median(np.array(widths, dtype=float)))
                    filtered = [r for r in filtered if float(r.get('width', 0) or 0) <= 2.2 * w_med]
                if areas_for_outliers:
                    a_med = float(np.median(np.array(areas_for_outliers, dtype=float)))
                    filtered = [r for r in filtered if float(r.get('area', 0) or 0) <= 3.0 * a_med]
                logger.info(f"После удаления аутлаеров по ширине/площади: {len(filtered)} регионов")
                if len(filtered) < 5:
                    return False
            except Exception:
                pass

            heights = [float(r.get('height', 0)) for r in filtered if float(r.get('height', 0)) > 8]
            if len(heights) < 2:
                return False

            heights_arr = np.array(heights, dtype=float)
            median_h = float(np.median(heights_arr))
            if median_h <= 0:
                return False

            # Ранний «один шрифт»: ≥70% высот в коридоре ±30% от медианы
            in_band = np.logical_and(heights_arr >= 0.7 * median_h, heights_arr <= 1.3 * median_h)
            frac_in_band = float(np.sum(in_band)) / float(len(heights_arr))
            logger.info(f"Доля высот в [0.7..1.3] от медианы: {frac_in_band:.2f}")
            likely_one_font = frac_in_band >= float(cfg.get('in_band_frac', 0.75))

            # Робастная дисперсия (MAD)
            mad = float(np.median(np.abs(heights_arr - median_h)) + 1e-6)
            robust_std = 1.4826 * mad
            height_variation = robust_std / median_h
            logger.info(f"Robust variation = {height_variation:.3f}")

            # Условие: большая вариация считает множественные шрифты
            if height_variation > max(0.7, float(cfg.get('size_variation_threshold', 0.4)) + 0.3):
                logger.info("✅ Очень большая вариация высот — множественные шрифты")
                return True

            # Простая двухкластерная проверка по высоте (порог 2.0x и поддержка >=3 в каждом)
            h_min = float(np.min(heights_arr))
            h_max = float(np.max(heights_arr))
            ratio = h_max / h_min if h_min > 0 else 1.0
            logger.info(f"Соотношение высот max/min: {ratio:.2f}")
            if ratio > float(cfg.get('height_ratio_threshold', 2.0)):
                # Оценим поддержку кластеров через пороги от медианы
                small = heights_arr <= 0.85 * median_h
                large = heights_arr >= 1.15 * median_h
                small_n = int(np.sum(small))
                large_n = int(np.sum(large))
                min_per_cluster = int(cfg.get('min_regions_per_cluster', 3))
                if small_n >= min_per_cluster and large_n >= min_per_cluster:
                    logger.info("✅ Два кластера по высоте с достаточной поддержкой")
                    # Доп. проверка: различие по яркости/плотности штрихов между кластерами
                    def _cluster_metrics(mask: np.ndarray) -> Tuple[float, float, float]:
                        L_vals = []
                        densities = []
                        sats = []
                        for idx, rr in enumerate(filtered):
                            if not mask[idx]:
                                continue
                            region_img = rr.get('region', None)
                            try:
                                if region_img is None or getattr(region_img, 'size', 0) == 0:
                                    continue
                                lab = cv2.cvtColor(region_img, cv2.COLOR_RGB2LAB)
                                L = float(np.mean(lab[:, :, 0]))
                                L_vals.append(L)
                                # Оценка «толщины»: доля тёмных пикселей
                                gray = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
                                _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                density = float(np.mean(bin_inv == 255))
                                densities.append(density)
                                # Оценка насыщенности цвета (отличает чёрный от яркого заголовка)
                                hsv = cv2.cvtColor(region_img, cv2.COLOR_RGB2HSV)
                                S = float(np.mean(hsv[:, :, 1]))
                                sats.append(S)
                            except Exception:
                                continue
                        L_mean = float(np.mean(L_vals)) if L_vals else 0.0
                        d_mean = float(np.mean(densities)) if densities else 0.0
                        s_mean = float(np.mean(sats)) if sats else 0.0
                        return L_mean, d_mean, s_mean

                    # Собираем маски индексов под small/large на отфильтрованном списке
                    # heights_arr соответствует filtered по порядку
                    small_mask = small
                    large_mask = large
                    L_small, D_small, S_small = _cluster_metrics(small_mask)
                    L_large, D_large, S_large = _cluster_metrics(large_mask)
                    logger.info(f"Сравнение кластеров: L_diff={abs(L_large - L_small):.1f}, D_diff={abs(D_large - D_small):.2f}, S_diff={abs(S_large - S_small):.1f}")
                    met_diff = 0
                    if abs(S_large - S_small) >= float(cfg.get('saturation_diff_threshold', 20.0)):
                        met_diff += 1
                    if abs(D_large - D_small) >= float(cfg.get('density_diff_threshold', 0.12)):
                        met_diff += 1
                    if abs(L_large - L_small) >= float(cfg.get('brightness_diff_threshold', 12.0)):
                        met_diff += 1
                    if met_diff >= int(cfg.get('require_metric_count', 2)):
                        logger.info("✅ Кластеры различаются по достаточному числу метрик — множественные шрифты")
                        return True

            # Площади как дополнительный критерий (более строгий порог)
            areas = [float(r.get('area', 0)) for r in filtered if float(r.get('area', 0)) > 100]
            if len(areas) >= 2:
                areas_arr = np.array(areas, dtype=float)
                a_ratio = float(np.max(areas_arr)) / float(np.min(areas_arr)) if float(np.min(areas_arr)) > 0 else 1.0
                logger.info(f"Соотношение площадей max/min: {a_ratio:.2f}")
                if a_ratio > float(cfg.get('area_ratio_threshold', 3.5)):
                    logger.info("✅ Очень разные площади — множественные шрифты")
                    return True

            # Дополнительная эвристика: группируем по тексту и сравниваем медианные высоты/плотности
            try:
                from collections import defaultdict
                groups_h: Dict[str, List[float]] = defaultdict(list)
                groups_d: Dict[str, List[float]] = defaultdict(list)
                for r in filtered:
                    txt = str(r.get('text', '')).strip()
                    if not txt:
                        continue
                    h = float(r.get('height', 0) or 0)
                    region_img = r.get('region', None)
                    dens = 0.0
                    try:
                        if region_img is not None and getattr(region_img, 'size', 0) > 0:
                            gray = cv2.cvtColor(region_img, cv2.COLOR_RGB2GRAY)
                            _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            dens = float(np.mean(bin_inv == 255))
                    except Exception:
                        pass
                    if h > 8:
                        groups_h[txt].append(h)
                        groups_d[txt].append(dens)
                if len(groups_h) >= 2:
                    # Берём две самые частые строки (обычно заголовок и подзаголовок)
                    items = sorted(groups_h.items(), key=lambda kv: len(kv[1]), reverse=True)
                    a_txt, a_vals = items[0][0], items[0][1]
                    b_txt, b_vals = items[1][0], items[1][1]
                    a_h, b_h = float(np.median(a_vals)), float(np.median(b_vals))
                    a_d = float(np.median(groups_d.get(a_txt, [0.0])))
                    b_d = float(np.median(groups_d.get(b_txt, [0.0])))
                    h_ratio = max(a_h, b_h) / max(1.0, min(a_h, b_h))
                    d_diff = abs(a_d - b_d)
                    logger.info(f"Группы '{a_txt[:12]}...' vs '{b_txt[:12]}...': h_ratio={h_ratio:.2f}, d_diff={d_diff:.2f}")
                    if h_ratio >= float(cfg.get('height_ratio_threshold', 2.0)) or d_diff >= float(cfg.get('density_diff_threshold', 0.12)):
                        logger.info("✅ Различие между самыми частыми строками — множественные шрифты")
                        return True
            except Exception:
                pass

            if likely_one_font:
                logger.info("ℹ️ Признаки множественных шрифтов не подтверждены, доминирует один кластер — один шрифт")
                return False
            logger.info("❌ Признаков множественных шрифтов недостаточно — один шрифт")
            return False

        except Exception as e:
            logger.error(f"Ошибка определения множественных шрифтов: {str(e)}")
            return False
    
    def _cluster_font_sizes(self, sizes: List[float], threshold: float = 0.3) -> List[List[float]]:
        """Кластеризация размеров шрифтов для выявления групп"""
        if len(sizes) < 2:
            return [sizes]
        
        sorted_sizes = sorted(sizes)
        clusters = []
        current_cluster = [sorted_sizes[0]]
        
        for size in sorted_sizes[1:]:
            # Если размер близок к среднему текущего кластера, добавляем в него
            cluster_mean = np.mean(current_cluster)
            relative_diff = abs(size - cluster_mean) / cluster_mean
            
            if relative_diff <= threshold:
                current_cluster.append(size)
            else:
                # Начинаем новый кластер
                clusters.append(current_cluster)
                current_cluster = [size]
        
        clusters.append(current_cluster)
        return clusters
    
    def _analyze_text_content_for_fonts(self, text_regions: List[Dict]) -> bool:
        """Анализ текстового содержимого для выявления разных шрифтов"""
        try:
            texts = [region.get('text', '') for region in text_regions]
            texts = [text.strip() for text in texts if text and text.strip()]
            
            if len(texts) < 2:
                return False
            
            # Анализ стилей текста
            has_uppercase = any(text.isupper() for text in texts)
            has_lowercase = any(text.islower() for text in texts)
            has_mixed_case = any(text[0].isupper() and any(c.islower() for c in text[1:]) for text in texts if len(text) > 1)
            has_numbers = any(any(c.isdigit() for c in text) for text in texts)
            
            # Анализ длин слов
            word_lengths = []
            for text in texts:
                words = text.split()
                if words:
                    avg_word_len = sum(len(word) for word in words) / len(words)
                    word_lengths.append(avg_word_len)
            
            # Если есть существенные различия в стилях или длинах слов
            style_variety_score = sum([has_uppercase, has_lowercase, has_mixed_case, has_numbers])
            
            if style_variety_score >= 3:  # Много разных стилей
                logger.info(f"Обнаружено разнообразие стилей текста: uppercase={has_uppercase}, lowercase={has_lowercase}, mixed={has_mixed_case}, numbers={has_numbers}")
                return True
            
            if len(word_lengths) >= 2:
                word_len_variation = np.std(word_lengths) / np.mean(word_lengths) if np.mean(word_lengths) > 0 else 0
                if word_len_variation > 0.5:  # Большая вариация в длинах слов
                    logger.info(f"Обнаружена большая вариация в длинах слов: {word_len_variation:.3f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка анализа содержимого: {str(e)}")
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
                    # Быстрая проверка - проверяем наличие метода ocr
                    object_working = hasattr(self.ocr, 'ocr') and callable(getattr(self.ocr, 'ocr', None))
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

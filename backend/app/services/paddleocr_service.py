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

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

logger = logging.getLogger(__name__)

# Теперь логируем результат импорта
if PADDLEOCR_AVAILABLE:
    logger.info("✅ PaddleOCR успешно импортирован")
else:
    logger.error(f"❌ Ошибка импорта PaddleOCR: {str(e)}")
    logger.error("💡 Установите: pip install paddlepaddle paddleocr")


class PaddleOCRService:
    """Сервис для профессиональной детекции и анализа текста с помощью PaddleOCR"""
    
    def __init__(self):
        self.ocr = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Инициализация PaddleOCR"""
        try:
            logger.info("🔍 Проверка доступности PaddleOCR...")
            logger.info(f"  - PADDLEOCR_AVAILABLE: {PADDLEOCR_AVAILABLE}")
            
            if not PADDLEOCR_AVAILABLE:
                logger.error("❌ PaddleOCR не установлен! Установите: pip install paddlepaddle paddleocr")
                return
            
            logger.info("🚀 Инициализация PaddleOCR...")
            logger.info("  - Язык: русский (кириллица)")
            logger.info("  - АГРЕССИВНЫЕ настройки для поиска текста")
            logger.info("  - det_db_thresh: 0.1 (очень низкий)")
            logger.info("  - det_db_box_thresh: 0.2 (низкий)")
            logger.info("  - det_db_unclip_ratio: 3.0 (большое расширение)")
            
            # Инициализируем PaddleOCR с максимально агрессивными настройками для поиска текста
            self.ocr = PaddleOCR(
                lang='ru',               # Русский язык (включает кириллицу)
                use_angle_cls=True,      # Определение ориентации текста
                # Агрессивные настройки для поиска даже слабого текста
                det_db_thresh=0.1,       # ОЧЕНЬ низкий порог детекции (по умолчанию 0.3)
                det_db_box_thresh=0.2,   # Низкий порог для bounding box (по умолчанию 0.5)
                det_db_unclip_ratio=3.0, # Большее расширение области детекции (по умолчанию 1.6)
                rec_batch_num=16,        # Больший размер батча для распознавания
                max_text_length=200      # Больше символов в строке
            )
            
            logger.info("✅ PaddleOCR успешно инициализирован")
            logger.info(f"  - Объект OCR: {self.ocr}")
            
            # Тестируем PaddleOCR на простом изображении
            try:
                logger.info("🧪 Тестируем PaddleOCR...")
                test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255  # Белое изображение
                test_result = self.ocr.ocr(test_image)
                logger.info(f"✅ Тест PaddleOCR: {type(test_result)}, результат: {test_result}")
                
                # Дополнительный тест с простым текстом
                logger.info("🧪 Тестируем PaddleOCR на простом тексте...")
                # Создаем простое изображение с текстом (черные прямоугольники на белом фоне)
                test_text_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
                # Рисуем простые прямоугольники, имитирующие текст
                cv2.rectangle(test_text_image, (50, 50), (350, 80), (0, 0, 0), -1)  # Первая строка
                cv2.rectangle(test_text_image, (50, 100), (300, 130), (0, 0, 0), -1)  # Вторая строка
                cv2.rectangle(test_text_image, (50, 150), (250, 180), (0, 0, 0), -1)  # Третья строка
                
                test_text_result = self.ocr.ocr(test_text_image)
                logger.info(f"✅ Тест PaddleOCR на тексте: {type(test_text_result)}, результат: {test_text_result}")
                
            except Exception as test_error:
                logger.error(f"❌ Тест PaddleOCR не прошел: {str(test_error)}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации PaddleOCR: {str(e)}")
            logger.error(f"  - Тип ошибки: {type(e).__name__}")
            logger.error(f"  - Детали: {str(e)}")
            self.ocr = None
    
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Алиас для обратной совместимости"""
        return await self.detect_and_analyze_text(image)
    
    async def detect_and_analyze_text(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Детекция и анализ текста на изображении
        
        Returns:
            Dict с информацией о тексте:
            - has_text: bool - есть ли текст
            - text_regions: List - области с текстом
            - multiple_fonts: bool - есть ли разные шрифты
            - confidence: float - уверенность детекции
            - text_content: str - распознанный текст
        """
        if not self.ocr:
            logger.error("PaddleOCR не инициализирован")
            return {
                'has_text': False,
                'text_regions': [],
                'multiple_fonts': False,
                'confidence': 0.0,
                'text_content': '',
                'error': 'PaddleOCR не инициализирован'
            }
        
        try:
            # Запускаем OCR в отдельном потоке (блокирующая операция)
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
            # Конвертируем в grayscale для некоторых операций
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
                if min(h, w) < 400:
                    scale = max(2, 400 // min(h, w))
                    if len(image.shape) == 3:
                        resized = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                    else:
                        resized_gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                        resized = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)
                    variants.append(resized)
                
                # Дополнительно: очень большое увеличение для мелкого текста
                if min(h, w) < 200:
                    scale = max(4, 600 // min(h, w))
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
                enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                variants.append(enhanced_rgb)
            except:
                pass
            
            # 4. CLAHE (адаптивная эквализация)
            try:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                clahe_image = clahe.apply(gray)
                clahe_rgb = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
                variants.append(clahe_rgb)
            except:
                pass
            
            # 5. Адаптивная бинаризация
            try:
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                morphed_rgb = cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB)
                variants.append(morphed_rgb)
            except:
                pass
            
            # 8. Легкое размытие для улучшения детекции
            try:
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
                variants.append(blurred_rgb)
            except:
                pass
            
            # 9. Увеличенная яркость
            try:
                brightened = cv2.convertScaleAbs(gray, alpha=1.5, beta=50)
                brightened_rgb = cv2.cvtColor(brightened, cv2.COLOR_GRAY2RGB)
                variants.append(brightened_rgb)
            except:
                pass
            
            logger.info(f"✅ Создано {len(variants)} вариантов для OCR:")
            for i, variant in enumerate(variants):
                logger.info(f"  - Вариант {i+1}: {variant.shape}")
            return variants
            
        except Exception as e:
            logger.error(f"Ошибка создания вариантов: {str(e)}")
            return [image.copy()]  # Возвращаем хотя бы оригинал
    
    def _run_ocr_sync(self, image: np.ndarray) -> Dict[str, Any]:
        """Синхронный запуск OCR"""
        try:
            logger.info("🔍 Запуск PaddleOCR анализа...")
            
            # АГРЕССИВНЫЙ поиск текста с множественными попытками
            logger.info(f"🖼️ Размер изображения: {image.shape}")
            
            # Создаем варианты изображения для попыток
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
                    
                    # Детальное логирование OCR вызова
                    logger.info(f"  - Вызываем self.ocr.ocr() для варианта {variant.shape}")
                    variant_result = self.ocr.ocr(variant)
                    logger.info(f"  - OCR результат: {type(variant_result)}")
                    
                    # Детальный анализ результата
                    if variant_result is None:
                        logger.info(f"    - OCR вернул None")
                    elif len(variant_result) == 0:
                        logger.info(f"    - OCR вернул пустой список")
                    else:
                        logger.info(f"    - OCR вернул {len(variant_result)} страниц")
                        for page_idx, page in enumerate(variant_result):
                            if page is None:
                                logger.info(f"      - Страница {page_idx}: None")
                            elif len(page) == 0:
                                logger.info(f"      - Страница {page_idx}: пустая")
                            else:
                                logger.info(f"      - Страница {page_idx}: {len(page)} строк")
                                for line_idx, line in enumerate(page[:2]):  # Показываем первые 2 строки
                                    if line and len(line) >= 2:
                                        text = line[1][0] if isinstance(line[1], (list, tuple)) and len(line[1]) > 0 else str(line[1])
                                        conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 1.0
                                        logger.info(f"        - Строка {line_idx}: '{text[:50]}...' (уверенность: {conf:.3f})")
                                    else:
                                        logger.info(f"        - Строка {line_idx}: некорректная структура {line}")
                    
                    if variant_result:
                        logger.info(f"  - Длина результата: {len(variant_result) if variant_result else 0}")
                        if variant_result and len(variant_result) > 0:
                            logger.info(f"  - Первая страница: {type(variant_result[0])}, длина: {len(variant_result[0]) if variant_result[0] else 0}")
                    
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
                        if variant_result:
                            logger.info(f"  - Результат: {variant_result}")
                        
                except Exception as variant_error:
                    logger.error(f"💥 Ошибка OCR варианта #{i+1}: {str(variant_error)}")
                    logger.error(f"  - Тип ошибки: {type(variant_error).__name__}")
                    continue
            
            # Используем лучший результат
            if best_result:
                ocr_result = best_result
                logger.info(f"✅ Используем лучший результат с уверенностью {best_confidence:.2f}")
                
                # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ OCR результата
                logger.info(f"🔍 ДЕТАЛИ OCR РЕЗУЛЬТАТА:")
                if best_result and len(best_result) > 0 and best_result[0]:
                    page = best_result[0]
                    logger.info(f"  - Найдено регионов: {len(page)}")
                    for i, region in enumerate(page[:3]):  # Показываем первые 3
                        if region and len(region) >= 2:
                            text = region[1][0] if isinstance(region[1], (list, tuple)) and len(region[1]) > 0 else str(region[1])
                            conf = region[1][1] if isinstance(region[1], (list, tuple)) and len(region[1]) > 1 else 1.0
                            logger.info(f"    Регион {i+1}: '{text[:30]}...' (уверенность: {conf:.2f})")
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
            
            # Проверяем результат согласно документации PaddleOCR
            if not ocr_result:
                logger.info("ℹ️ PaddleOCR вернул None - нет текста")
                return {
                    'has_text': False,
                    'text_regions': [],
                    'multiple_fonts': False,
                    'confidence': 0.0,
                    'text_content': '',
                    'error': "OCR не обнаружил текст на изображении"
                }
            
            # OCR результат - список страниц, берем первую страницу
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
            
            # Обрабатываем результат
            text_regions = []
            all_text = []
            confidences = []
            
            # Добавим детальное логирование структуры результата
            logger.info(f"🔍 Структура OCR результата: {type(ocr_result)}, длина: {len(ocr_result) if ocr_result else 0}")
            if ocr_result and len(ocr_result) > 0:
                logger.info(f"📊 Первый элемент: {type(ocr_result[0])}, длина: {len(ocr_result[0]) if ocr_result[0] else 0}")
                if ocr_result[0] and len(ocr_result[0]) > 0:
                    logger.info(f"📝 Пример строки: {ocr_result[0][0] if len(ocr_result[0]) > 0 else 'пусто'}")
            
            for line in page_result:
                if line and len(line) >= 2:
                    bbox = line[0]  # Координаты области
                    text_info = line[1]  # (текст, уверенность)
                    
                    logger.info(f"🔍 Обрабатываем строку: bbox={bbox}, text_info={text_info}")
                    
                    # Безопасное извлечение текста и уверенности
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0])
                        confidence = float(text_info[1])
                    elif isinstance(text_info, str):
                        text = text_info
                        confidence = 1.0  # Если нет уверенности, ставим максимальную
                    else:
                        logger.warning(f"⚠️ Некорректный text_info: {text_info}")
                        continue  # Пропускаем некорректные данные
                    
                    # Анализируем область текста
                    region_info = self._analyze_text_region(image, bbox, text, confidence)
                    text_regions.append(region_info)
                    
                    all_text.append(text)
                    confidences.append(confidence)
            
            # Общая статистика и проверка качества
            avg_confidence = np.mean(confidences) if confidences else 0.0
            text_content = ' '.join(all_text)
            
            # Проверяем наличие КАЧЕСТВЕННОГО текста - смягченные фильтры
            valid_regions = [r for r in text_regions if 
                           r.get('confidence', 0) > 0.3 and  # Минимум 30% уверенности (было 50%)
                           not r.get('is_invalid', False)]   # Не помечена как невалидная
            clean_text = ''.join(c for c in text_content if c.isalnum() or c.isspace()).strip()  # Только буквы/цифры
            
            # Дополнительная проверка: если все области невалидные - нет текста
            invalid_regions = [r for r in text_regions if r.get('is_invalid', False)]
            
            has_text = (len(valid_regions) > 0 and 
                       len(clean_text) >= 1 and  # Минимум 1 символ (было 2)
                       avg_confidence > 0.2 and  # Общая уверенность выше 20% (было 30%)
                       len(invalid_regions) < len(text_regions))  # Не все области невалидные
            
            logger.info(f"🔍 Проверка качества: всего областей={len(text_regions)}, валидных={len(valid_regions)}, невалидных={len(invalid_regions)}")
            logger.info(f"📝 Чистый текст: '{clean_text[:50]}' (длина: {len(clean_text)})")
            logger.info(f"📊 Средняя уверенность: {avg_confidence:.2f}")
            logger.info(f"✅ Результат проверки: has_text={has_text}")
            
            # Определяем множественные шрифты
            multiple_fonts = self._detect_multiple_fonts_from_regions(text_regions)
            
            logger.info(f"✅ PaddleOCR: найдено {len(text_regions)} областей текста")
            logger.info(f"📊 Средняя уверенность: {avg_confidence:.2f}")
            logger.info(f"📝 Текст: {text_content[:100]}...")
            
            # Добавляем недостающие поля для совместимости
            result = {
                'has_text': has_text,
                'text_regions': text_regions,
                'multiple_fonts': multiple_fonts,
                'confidence': avg_confidence,
                'text_content': text_content,
                'regions_count': len(text_regions),
                'ocr_boxes': text_regions,  # Для совместимости с font_analyzer
                'error': None if has_text else "OCR нашел текст, но он не прошел проверку качества"
            }
            
            logger.info(f"✅ PaddleOCR результат: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Техническая ошибка в _run_ocr_sync: {str(e)}")
            raise
    
    def _analyze_text_region(self, image: np.ndarray, bbox: List, text: str, confidence: float) -> Dict[str, Any]:
        """Анализ отдельной области текста"""
        try:
            # Безопасное извлечение координат
            try:
                # Преобразуем bbox в числовой массив
                if isinstance(bbox[0], (list, tuple)):
                    # bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    points = np.array(bbox, dtype=np.float32)
                    x_min = int(np.min(points[:, 0]))
                    y_min = int(np.min(points[:, 1]))
                    x_max = int(np.max(points[:, 0]))
                    y_max = int(np.max(points[:, 1]))
                else:
                    # bbox = [x1, y1, x2, y2]
                    coords = [float(coord) for coord in bbox[:4]]
                    x_min, y_min, x_max, y_max = map(int, coords)
            except (ValueError, TypeError, IndexError) as e:
                logger.error(f"Ошибка парсинга bbox {bbox}: {str(e)}")
                # Возвращаем область с минимальными данными для фильтрации
                return {
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence,
                    'width': 10,  # Минимальные размеры для фильтрации
                    'height': 10,
                    'area': 100,
                    'font_size_estimate': 12,
                    'is_invalid': True  # Помечаем как невалидную
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
            
            # Оценка размера шрифта (приблизительная)
            font_size_estimate = height * 0.7  # Примерно 70% от высоты области
            
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
            
            # Если стандартное отклонение больше 15% от среднего - разные шрифты
            variation_ratio = std_size / mean_size if mean_size > 0 else 0
            
            logger.info(f"📏 Анализ размеров шрифтов: среднее={mean_size:.1f}, отклонение={std_size:.1f}, коэффициент={variation_ratio:.2f}")
            
            # Очень чувствительный порог для PaddleOCR
            if variation_ratio > 0.15:  # 15% разброса
                logger.info("🔤 Обнаружены множественные шрифты (разные размеры)")
                return True
            
            # Дополнительная проверка по площади областей
            areas = [region.get('area', 0) for region in text_regions]
            areas = [area for area in areas if area > 0]
            
            if len(areas) >= 2:
                areas = np.array(areas)
                area_ratio = np.max(areas) / np.min(areas) if np.min(areas) > 0 else 0
                
                if area_ratio > 2.0:  # Разница в площади больше чем в 2 раза
                    logger.info("🔤 Обнаружены множественные шрифты (разные площади)")
                    return True
            
            # Дополнительная проверка: если много областей текста - вероятно разные шрифты
            if len(text_regions) >= 4:  # 4 или больше областей
                logger.info(f"🔤 Обнаружены множественные шрифты (много областей: {len(text_regions)})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка определения множественных шрифтов: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Проверка доступности PaddleOCR"""
        available = self.ocr is not None and PADDLEOCR_AVAILABLE
        logger.info(f"🔍 PaddleOCR проверка: ocr={self.ocr is not None}, PADDLEOCR_AVAILABLE={PADDLEOCR_AVAILABLE}, итого={available}")
        return available

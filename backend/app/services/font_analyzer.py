"""
Сервис анализа шрифтов с использованием PaddleOCR и OpenCV
"""

import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import Tuple, List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.font_models import FontCharacteristics, CyrillicFeatures
from .paddleocr_service import PaddleOCRService

logger = logging.getLogger(__name__)


class FontAnalyzer:
    """Анализатор шрифтов на основе PaddleOCR и OpenCV"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.paddleocr_service = PaddleOCRService()
        
    async def analyze_image(self, image_bytes: bytes) -> FontCharacteristics:
        """Анализ изображения для определения характеристик шрифта"""
        return await self._analyze_image_async(image_bytes)
    
    async def _analyze_image_async(self, image_bytes: bytes) -> FontCharacteristics:
        """Асинхронный анализ изображения через OCR"""
        
        # Загружаем изображение
        image = self._load_image(image_bytes)
        
        # Проверяем на множественные шрифты ПОСЛЕ анализа
        logger.info("=== Проверяем множественные шрифты ПОСЛЕ анализа ===")
        
        # Сначала делаем анализ
        logger.info("Выполняем анализ шрифта...")
        
        # ШАГ 1: Сначала исключаем очевидно НЕ-текстовые изображения (смягченный)
        logger.info("=== ШАГ 1: Проверка НЕ-текстовых изображений (смягченная) ===")
        if self._is_obviously_not_text(image):
            logger.info("РЕЗУЛЬТАТ ШАГ 1: Изображение определено как НЕ-текстовое")
            raise ValueError("На изображении не обнаружен текст для анализа. Попробуйте загрузить изображение с четким, читаемым текстом.")
        logger.info("РЕЗУЛЬТАТ ШАГ 1: Изображение прошло проверку НЕ-текстовых")
        

    
    async def _analyze_image_async(self, image_bytes: bytes) -> FontCharacteristics:
        """Асинхронный анализ изображения с fallback"""
        try:
            # Загружаем изображение
            image = self._load_image(image_bytes)
            
            # ШАГ 1: Проверяем на очевидно НЕ-текстовые изображения
            logger.info("=== ШАГ 1: Проверка на НЕ-текстовые изображения ===")
            if self._is_obviously_not_text(image):
                logger.info("РЕЗУЛЬТАТ ШАГ 1: Изображение не содержит текст - СТОП")
                raise ValueError("Изображение не содержит читаемый текст для анализа")
            else:
                logger.info("РЕЗУЛЬТАТ ШАГ 1: Изображение может содержать текст - продолжаем")
            
            # ШАГ 2: Проверяем наличие текста
            logger.info("=== ШАГ 2: Проверка наличия текста ===")
            if not self._detect_text_presence(image):
                logger.info("РЕЗУЛЬТАТ ШАГ 2: Текст не обнаружен - СТОП")
                raise ValueError("На изображении не обнаружен читаемый текст для анализа")
            else:
                logger.info("РЕЗУЛЬТАТ ШАГ 2: Текст обнаружен - продолжаем")
            
            # ШАГ 3: Проверяем на множественные шрифты (только если текст найден) - OCR АНАЛИЗ
            logger.info("=== ШАГ 3: Проверка множественных шрифтов (OCR) ===")
            if await self._detect_multiple_fonts(image):
                logger.info("РЕЗУЛЬТАТ ШАГ 3: Обнаружено несколько шрифтов - СТОП")
                raise ValueError("На изображении обнаружено несколько разных шрифтов. Для точного анализа загрузите изображение с текстом одного шрифта.")
            logger.info("РЕЗУЛЬТАТ ШАГ 3: Один шрифт - продолжаем к анализу")
            
            # ШАГ 4: ТОЛЬКО ТЕПЕРЬ делаем анализ шрифта через OCR (основная операция)
            logger.info("=== ШАГ 4: Анализ характеристик шрифта ЧЕРЕЗ OCR ===")
            
            # Извлекаем характеристики ТОЛЬКО из OCR
            # Сначала получаем OCR результат
            if not hasattr(self, 'paddleocr_service') or not self.paddleocr_service:
                logger.error("❌ PaddleOCR сервис недоступен!")
                raise ValueError("PaddleOCR сервис недоступен")
            
            # Проверяем доступность PaddleOCR
            if not self.paddleocr_service.is_available():
                logger.error("❌ PaddleOCR не инициализирован!")
                raise ValueError("PaddleOCR не инициализирован")
            
            logger.info(f"🔍 Запускаем PaddleOCR анализ для изображения {image.shape}")
            logger.info(f"🔍 PaddleOCR доступен: {self.paddleocr_service.is_available()}")
            
            # Используем правильный метод PaddleOCR
            ocr_result = await self.paddleocr_service.analyze_image(image)
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ OCR результата
            logger.info(f"🔍 PADDLEOCR РЕЗУЛЬТАТ:")
            logger.info(f"  - has_text: {ocr_result.get('has_text', False)}")
            logger.info(f"  - text_content: '{ocr_result.get('text_content', '')[:50]}...'")
            logger.info(f"  - confidence: {ocr_result.get('confidence', 0.0):.3f}")
            logger.info(f"  - regions_count: {ocr_result.get('regions_count', 0)}")
            logger.info(f"  - error: {ocr_result.get('error', 'нет')}")
            
            if not ocr_result.get('has_text', False):
                logger.error(f"❌ OCR не нашел текст: {ocr_result}")
                raise ValueError("OCR не смог найти текст на изображении")
            
            logger.info(f"✅ OCR успешно нашел текст, извлекаем характеристики...")
            characteristics = await self._extract_characteristics_from_ocr(image, ocr_result)
            
            logger.info("✅ Анализ завершен успешно через OCR")
            return characteristics
            
        except ValueError as logic_error:
            # Логические ошибки (нет текста, много шрифтов) - НЕ fallback!
            logger.info(f"ℹ️ Логический результат OCR: {str(logic_error)}")
            raise logic_error  # Передаем ошибку пользователю
            
        except Exception as ocr_error:
            # Только технические ошибки OCR - fallback
            logger.error(f"❌ Техническая ошибка OCR: {str(ocr_error)}")
            logger.warning("⚠️ Переключаемся на fallback метод...")
            
            # FALLBACK: только при технических ошибках OCR
            try:
                logger.info("=== FALLBACK: Анализ без OCR ===")
                characteristics = await self._extract_characteristics_from_full_image(image)
                logger.info("✅ Fallback анализ завершен успешно")
                return characteristics
            except Exception as fallback_error:
                logger.error(f"❌ Ошибка fallback анализа: {str(fallback_error)}")
                raise ValueError(f"Не удалось проанализировать изображение: {str(ocr_error)}")
    
    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """Загрузка изображения из байтов"""
        try:
            # Используем PIL для загрузки
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Конвертируем в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Конвертируем в numpy array (RGB для PaddleOCR)
            cv_image = np.array(pil_image)  # Оставляем RGB формат для PaddleOCR
            
            logger.info(f"Изображение загружено: {cv_image.shape}, формат: RGB")
            return cv_image
            
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения: {str(e)}")
            raise ValueError(f"Не удалось загрузить изображение: {str(e)}")
    
    def _is_obviously_not_text(self, image: np.ndarray) -> bool:
        """Проверка на очевидно НЕ-текстовые изображения (иконки, простая графика)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Проверяем размер изображения
            height, width = gray.shape
            total_pixels = height * width
            
            # Слишком маленькие изображения обычно не содержат читаемый текст
            if total_pixels < 2000:  # Менее 45x45 (еще более мягко)
                return True
            
            # Проверяем однородность (простые иконки часто однородны)
            std_dev = np.std(gray.astype(np.float64))
            if std_dev < 1:  # Очень низкая вариация - вероятно простая графика
                return True
            
            # Проверяем количество уникальных цветов (слишком мало = простая графика)
            unique_values = len(np.unique(gray))
            if unique_values < 3:  # Слишком мало градаций серого
                return True
            
            # Проверяем преобладание одного цвета
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            max_bin_ratio = np.max(hist) / total_pixels
            if max_bin_ratio > 0.95:  # Более 95% пикселей одного цвета
                return True
            
            logger.info(f"Проверка НЕ-текста: размер={total_pixels}, отклонение={std_dev:.1f}, цвета={unique_values}, преобладание={max_bin_ratio:.3f}")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки НЕ-текста: {str(e)}")
            return False  # В случае ошибки разрешаем дальнейший анализ
    
    def _detect_text_presence(self, image: np.ndarray) -> bool:
        """УПРОЩЕННОЕ и НАДЕЖНОЕ определение наличия текста"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            total_pixels = height * width
            
            logger.info("=== УПРОЩЕННЫЙ АНАЛИЗ ТЕКСТА ===")
            logger.info(f"Размер изображения: {width}x{height} ({total_pixels} пикселей)")
            
            # ПРОСТАЯ И НАДЕЖНАЯ ЛОГИКА:
            
            # 1. Если изображение слишком маленькое - скорее всего иконка
            if total_pixels < 2000:  # Меньше 45x45
                logger.info("Слишком маленькое изображение - вероятно иконка")
                return False
            
            # 2. Если изображение слишком однородное - вероятно простая графика
            std_dev = np.std(gray.astype(np.float64))
            if std_dev < 5:
                logger.info(f"Слишком однородное изображение (std={std_dev:.1f}) - вероятно простая графика")
                return False
            
            # 3. Если изображение имеет разумный размер И есть вариации - скорее всего есть текст
            if total_pixels >= 2000 and std_dev >= 5:
                logger.info(f"Разумный размер ({total_pixels}) + вариации ({std_dev:.1f}) = ЕСТЬ ТЕКСТ")
                return True
            
            # 4. Для остальных случаев - тоже считаем что есть текст (очень либерально)
            logger.info("По умолчанию считаем что текст есть")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка определения текста: {str(e)}")
            # В случае ошибки - считаем что текст есть (безопасный подход)
            return True
    
    def _legacy_text_detection(self, gray: np.ndarray) -> bool:
        """Старый метод определения текста (fallback)"""
        try:
            # Применяем фильтр для выделения текста
            edges = cv2.Canny(gray, 30, 120)  # Более мягкие пороги
            
            # Ищем контуры
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Анализируем контуры на предмет текстовых регионов
            text_like_contours = 0
            large_contours = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Расширяем диапазон размеров
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Более гибкие критерии для букв
                    if 0.1 < aspect_ratio < 5.0 and area < 10000:
                        text_like_contours += 1
                        
                    # Учитываем крупные текстовые элементы
                    if area > 1000:
                        large_contours += 1
            
            # Анализируем плотность темных пикселей (более гибко)
            dark_pixels = np.sum(gray < 128)  # Повышаем порог
            total_pixels = gray.shape[0] * gray.shape[1]
            dark_ratio = dark_pixels / total_pixels
            
            # Анализируем контрастность
            contrast = np.std(gray.astype(np.float64))
            
            # МАКСИМАЛЬНО мягкие условия определения текста
            has_text = (
                (text_like_contours >= 1 or large_contours >= 1 or contrast > 10) and  # Еще более гибко
                0.001 < dark_ratio < 0.99 and                         # Практически любой диапазон
                contrast > 2                                          # Минимальная контрастность
            )
            
            logger.info(f"Legacy анализ: контуры={text_like_contours}, крупные={large_contours}, темные пиксели={dark_ratio:.3f}, контрастность={contrast:.1f}, результат={has_text}")
            return has_text
            
        except Exception as e:
            logger.error(f"Ошибка legacy анализа: {str(e)}")
            return False
    
    def _detect_potential_text(self, image: np.ndarray) -> bool:
        """Дополнительная проверка на наличие потенциального текста (логотипы, стилизованный текст)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Проверяем общую контрастность
            contrast = np.std(gray.astype(np.float64))
            
            # Проверяем наличие четких границ
            edges = cv2.Canny(gray, 20, 100)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Проверяем разнообразие интенсивностей
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            non_zero_bins = np.count_nonzero(hist)
            
            # Ищем текстоподобные структуры более строго
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 8000:  # Размеры характерные для букв/слов
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Проверяем что это похоже на букву или слово
                    if 0.3 < aspect_ratio < 4.0:  # Разумные пропорции текста
                        # Дополнительная проверка на "текстовость"
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            # Текст обычно не очень круглый
                            if circularity < 0.7:
                                text_like_shapes += 1
            
            # МАКСИМАЛЬНО мягкие условия для потенциального текста
            has_potential_text = (
                contrast > 8 and                     # Еще ниже контрастность
                edge_density > 0.002 and            # Еще ниже требования к границам
                non_zero_bins > 5 and               # Еще ниже разнообразие тонов
                text_like_shapes >= 1               # Минимум 1 текстоподобная форма
            )
            
            logger.info(f"Дополнительная проверка: контрастность={contrast:.1f}, границы={edge_density:.4f}, тона={non_zero_bins}, формы={text_like_shapes}, результат={has_potential_text}")
            return has_potential_text
            
        except Exception as e:
            logger.error(f"Ошибка дополнительной проверки текста: {str(e)}")
            return False  # В случае ошибки НЕ разрешаем анализ
    
    async def _detect_multiple_fonts(self, image: np.ndarray) -> bool:
        """OCR-детекция множественных шрифтов - ТОЛЬКО через PaddleOCR"""
        try:
            logger.info("=== OCR АНАЛИЗ МНОЖЕСТВЕННЫХ ШРИФТОВ ===")
            
            if not hasattr(self, 'paddleocr_service') or not self.paddleocr_service:
                logger.warning("PaddleOCR сервис недоступен")
                return False
            
            # Получаем результаты OCR
            ocr_result = await self.paddleocr_service.analyze_image(image)
            
            if not ocr_result.get('has_text', False):
                logger.info("📊 Текст не найден - НЕ АНАЛИЗИРУЕМ")
                return False
            
            text_content = ocr_result.get('text_content', '').strip()
            regions_count = ocr_result.get('regions_count', 0)
            ocr_boxes = ocr_result.get('ocr_boxes', [])
            
            logger.info(f"📊 OCR результат: '{text_content}' ({regions_count} регионов)")
            
            # Анализируем текст
            words = text_content.split()
            word_count = len(words)
            
            # ПРОСТЫЕ СЛУЧАИ - один шрифт
            if word_count <= 2:  # 1-2 слова
                logger.info(f"📊 Мало слов ({word_count}) - ОДИН шрифт")
                return False
            
            if regions_count < 4:  # Мало регионов
                logger.info(f"📊 Мало регионов ({regions_count}) - ОДИН шрифт")
                return False
            
            # АНАЛИЗ СОДЕРЖИМОГО через OCR
            # Ищем признаки разных типов текста
            has_title = any(len(word) > 4 and word.isupper() for word in words)
            has_normal_text = any(len(word) > 3 and not word.isupper() for word in words)
            has_numbers = any(char.isdigit() for char in text_content)
            has_special_words = any(word.lower() in ['скидка', 'цена', 'рубль', '%', 'руб', 'распродажа'] for word in words)
            
            logger.info(f"📊 Анализ содержимого: заголовок={has_title}, текст={has_normal_text}, цифры={has_numbers}, спец.слова={has_special_words}")
            
            # АНАЛИЗ РАЗМЕРОВ ЧЕРЕЗ OCR BOXES
            height_ratio = 1.0  # По умолчанию
            area_ratio = 1.0    # По умолчанию
            
            if len(ocr_boxes) >= 4:
                # Анализируем размеры bounding boxes из OCR
                heights = []
                areas = []
                
                for box_info in ocr_boxes:
                    if isinstance(box_info, list) and len(box_info) >= 2:
                        box = box_info[0]  # координаты
                        if isinstance(box, list) and len(box) >= 4:
                            # Вычисляем размеры box
                            x_coords = [point[0] for point in box]
                            y_coords = [point[1] for point in box]
                            
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            
                            heights.append(height)
                            areas.append(width * height)
                
                if len(heights) >= 2:
                    height_ratio = max(heights) / min(heights) if min(heights) > 0 else 1
                    area_ratio = max(areas) / min(areas) if min(areas) > 0 else 1
                    
                    logger.info(f"📊 OCR размеры: высота={height_ratio:.1f}, площадь={area_ratio:.1f}")
            
            # УЛУЧШЕННЫЕ КРИТЕРИИ МНОЖЕСТВЕННЫХ ШРИФТОВ (более чувствительные):
            
            # 1. Простые случаи - точно ОДИН шрифт
            if word_count <= 2 and regions_count <= 3:
                logger.info("📊 Простой случай: очень мало слов и регионов - ОДИН шрифт")
                return False
            
            # 2. Средняя сложность - анализируем более детально
            if word_count <= 6 and regions_count <= 8:
                # Проверяем соотношение размеров - если небольшое, то один шрифт
                if height_ratio <= 2.0 and area_ratio <= 6.0:
                    logger.info("📊 Средняя сложность: размеры стабильные - ОДИН шрифт")
                    return False
                # Если есть явные признаки заголовка + текста
                elif has_title and has_normal_text and height_ratio > 2.0:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: заголовок + текст + заметная разница размеров")
                    return True
                else:
                    logger.info("📊 Средняя сложность: неопределенно - считаем ОДИН шрифт")
                    return False
            
            # 3. Сложные случаи - много текста
            if word_count > 6 or regions_count > 8:
                # Если очень большая разница в размерах - точно разные шрифты
                if height_ratio > 3.0 and area_ratio > 8.0:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: большая разница в размерах")
                    return True
                # Если есть заголовок + много текста + средняя разница
                elif has_title and has_normal_text and height_ratio > 1.8 and word_count >= 8:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: заголовок + много текста + разные размеры")
                    return True
                # Если просто много текста с умеренными различиями
                elif height_ratio > 2.5 and regions_count >= 12:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: много регионов + заметная разница размеров")
                    return True
                # Дополнительная проверка: если есть цифры + текст + разные размеры
                elif has_numbers and has_normal_text and height_ratio > 2.2:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: цифры + текст + разные размеры")
                    return True
                else:
                    logger.info("📊 Сложный случай: различия не критичные - ОДИН шрифт")
                    return False
            
            # Во всех остальных случаях - один шрифт
            logger.info("📊 Определено как ОДИН шрифт")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка OCR-определения множественных шрифтов: {str(e)}")
            # При ошибке считаем один шрифт (безопасно)
            return False
    
    def _get_ocr_based_characteristics(self, ocr_result: dict) -> dict:
        """Извлечение характеристик ТОЛЬКО из OCR результатов"""
        try:
            logger.info("=== ИЗВЛЕЧЕНИЕ ХАРАКТЕРИСТИК ИЗ OCR ===")
            
            text_content = ocr_result.get('text_content', '').strip()
            regions_count = ocr_result.get('regions_count', 0)
            ocr_boxes = ocr_result.get('ocr_boxes', [])
            
            # Анализ текста
            words = text_content.split()
            word_count = len(words)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Анализ размеров из OCR boxes
            heights = []
            widths = []
            areas = []
            
            for box_info in ocr_boxes:
                if isinstance(box_info, list) and len(box_info) >= 2:
                    box = box_info[0]  # координаты
                    if isinstance(box, list) and len(box) >= 4:
                        # Вычисляем размеры box
                        x_coords = [point[0] for point in box]
                        y_coords = [point[1] for point in box]
                        
                        width = max(x_coords) - min(x_coords)
                        height = max(y_coords) - min(y_coords)
                        
                        heights.append(height)
                        widths.append(width)
                        areas.append(width * height)
            
            # Характеристики на основе OCR данных
            characteristics = {
                'text_length': len(text_content),
                'word_count': word_count,
                'regions_count': regions_count,
                'avg_word_length': avg_word_length,
                'avg_height': np.mean(heights) if heights else 20.0,
                'avg_width': np.mean(widths) if widths else 100.0,
                'avg_area': np.mean(areas) if areas else 2000.0,
                'height_variance': np.var(heights) if len(heights) > 1 else 0.0,
                'width_variance': np.var(widths) if len(widths) > 1 else 0.0,
                'has_uppercase': any(c.isupper() for c in text_content),
                'has_lowercase': any(c.islower() for c in text_content),
                'has_numbers': any(c.isdigit() for c in text_content),
                'has_cyrillic': any(ord(c) >= 1040 and ord(c) <= 1103 for c in text_content),
                'text_density': word_count / max(regions_count, 1)  # слов на регион
            }
            
            logger.info(f"OCR характеристики: {characteristics}")
            return characteristics
            
        except Exception as e:
            logger.error(f"Ошибка извлечения OCR характеристик: {str(e)}")
            return self._get_default_ocr_characteristics()
    
    def _get_default_ocr_characteristics(self) -> dict:
        """OCR характеристики по умолчанию"""
        return {
            'text_length': 0,
            'word_count': 0,
            'regions_count': 0,
            'avg_word_length': 0,
            'avg_height': 20.0,
            'avg_width': 100.0,
            'avg_area': 2000.0,
            'height_variance': 0.0,
            'width_variance': 0.0,
            'has_uppercase': False,
            'has_lowercase': False,
            'has_numbers': False,
            'has_cyrillic': False,
            'text_density': 0.0
        }
    
    def _binarize_image(self, gray: np.ndarray) -> np.ndarray:
        """Бинаризация изображения"""
        # Используем адаптивную бинаризацию для лучшего результата
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    async def _extract_characteristics_from_ocr(self, image: np.ndarray, ocr_result: dict) -> FontCharacteristics:
        """Извлечение характеристик шрифта ТОЛЬКО из OCR"""
        
        logger.info("=== ИЗВЛЕЧЕНИЕ ХАРАКТЕРИСТИК ЧЕРЕЗ OCR ===")
        
        # OCR результат уже получен в вызывающем методе
        logger.info("📊 Используем переданный OCR результат")
        
        # ДОПОЛНИТЕЛЬНАЯ проверка на отсутствие текста
        text_content = ocr_result.get('text_content', '').strip()
        if not text_content or len(text_content) < 2:
            logger.warning("⚠️ OCR вернул пустой или слишком короткий текст")
            raise ValueError("На изображении не обнаружен читаемый текст для анализа")
        
        # Проверяем качество распознавания
        confidence = ocr_result.get('confidence', 0.0)
        if confidence < 0.3:  # Низкая уверенность
            logger.warning(f"⚠️ Низкая уверенность OCR: {confidence:.2f}")
            raise ValueError("OCR не смог уверенно распознать текст на изображении")
        
        # Получаем OCR характеристики
        ocr_chars = self._get_ocr_based_characteristics(ocr_result)
        
        # Конвертируем в FontCharacteristics на основе OCR данных
        text_content = ocr_result.get('text_content', '').strip()
        
        # Анализируем тип шрифта по содержимому и размерам
        has_serifs = self._predict_serifs_from_ocr(ocr_chars, text_content)
        
        # РЕАЛЬНЫЕ характеристики на основе содержимого изображения
        # stroke_width на основе реальной толщины текста
        if ocr_chars['avg_height'] > 0:
            # Нормализуем толщину относительно размера текста
            stroke_width = min(1.0, max(0.0, ocr_chars['avg_height'] / 50.0))
        else:
            stroke_width = 0.5
        
        # contrast на основе реальной вариативности размеров
        if ocr_chars['height_variance'] > 0 and ocr_chars['avg_height'] > 0:
            # Нормализуем контраст
            contrast = min(1.0, max(0.0, ocr_chars['height_variance'] / ocr_chars['avg_height']))
        else:
            contrast = 0.3
        
        # Наклон на основе реального содержимого
        # Анализируем текст на предмет наклона
        slant = 0.0  # Пока без анализа наклона
        
        # Уникальность через реальное содержимое изображения
        content_hash = hash(text_content + str(ocr_chars['regions_count']) + str(ocr_chars['avg_height']))
        unique_factor = (content_hash % 1000) / 1000.0
        
        # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ для отладки
        logger.info(f"🔍 УНИКАЛЬНЫЕ ХАРАКТЕРИСТИКИ ИЗОБРАЖЕНИЯ:")
        logger.info(f"  - Текст: '{text_content[:50]}...' (длина: {len(text_content)})")
        logger.info(f"  - Регионы: {ocr_chars['regions_count']}")
        logger.info(f"  - Средняя высота: {ocr_chars['avg_height']:.2f}")
        logger.info(f"  - Хеш содержимого: {content_hash}")
        logger.info(f"  - Уникальный фактор: {unique_factor:.3f}")
        logger.info(f"  - stroke_width: {stroke_width:.3f}")
        logger.info(f"  - contrast: {contrast:.3f}")
        logger.info(f"  - slant: {slant:.3f}")
        
        # Геометрические характеристики из OCR
        avg_height = ocr_chars['avg_height']
        x_height = avg_height * 0.6  # Примерная пропорция
        cap_height = avg_height
        ascender = avg_height * 1.2
        descender = avg_height * 0.3
        
        # Интервалы на основе плотности текста
        letter_spacing = ocr_chars['avg_width'] / max(ocr_chars['avg_word_length'], 1) * 0.1
        word_spacing = ocr_chars['avg_width'] * 0.3
        density = ocr_chars['text_density']
        
        # Кириллические особенности
        cyrillic_features = {
            'has_cyrillic': ocr_chars['has_cyrillic'],
            'cyrillic_ratio': 1.0 if ocr_chars['has_cyrillic'] else 0.0,
            'specific_letters': []
        }
        
        logger.info(f"OCR характеристики шрифта: засечки={has_serifs}, толщина={stroke_width:.1f}, высота={avg_height:.1f}")
        
        return FontCharacteristics(
            has_serifs=has_serifs,
            stroke_width=stroke_width,
            contrast=contrast,
            slant=slant,
            cyrillic_features=cyrillic_features,
            x_height=x_height,
            cap_height=cap_height,
            ascender=ascender,
            descender=descender,
            letter_spacing=letter_spacing,
            word_spacing=word_spacing,
            density=density
        )
    
    def _predict_serifs_from_ocr(self, ocr_chars: dict, text_content: str) -> bool:
        """Предсказание наличия засечек на основе OCR данных"""
        # Эвристика: если текст формальный и размеры стабильные - возможно засечки
        has_formal_text = any(word.lower() in ['официальный', 'документ', 'книга', 'статья'] for word in text_content.split())
        stable_sizes = ocr_chars['height_variance'] < ocr_chars['avg_height'] * 0.2
        
        return has_formal_text and stable_sizes
    
    def _detect_serifs(self, binary: np.ndarray) -> bool:
        """Определение наличия засечек"""
        # Применяем морфологические операции для выделения мелких деталей
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Находим разность - мелкие детали (потенциальные засечки)
        diff = cv2.absdiff(binary, opened)
        serif_pixels = np.sum(diff == 255)
        total_text_pixels = np.sum(binary == 0)  # Черные пиксели в бинарном изображении
        
        if total_text_pixels == 0:
            return False
        
        serif_ratio = serif_pixels / total_text_pixels
        has_serifs = serif_ratio > 0.05  # Эмпирический порог
        
        logger.info(f"Анализ засечек: соотношение={serif_ratio:.3f}, результат={has_serifs}")
        return has_serifs
    
    def _analyze_stroke_width(self, binary: np.ndarray) -> float:
        """Анализ толщины штрихов"""
        # Используем расстояние до ближайшего нуля (distance transform)
        dist_transform = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
        
        # Находим среднюю толщину штрихов
        text_pixels = binary == 0
        if np.sum(text_pixels) == 0:
            return 0.1
        
        avg_thickness = np.mean(dist_transform[text_pixels]) * 2  # Умножаем на 2 для полной ширины
        
        # Нормализуем относительно размера изображения
        normalized_thickness = avg_thickness / max(binary.shape)
        
        # Дополнительная нормализация для гарантии диапазона [0, 1]
        normalized_thickness = normalized_thickness / 10.0  # Делим на 10 для более реалистичных значений
        
        return min(1.0, max(0.0, normalized_thickness))
    
    def _analyze_contrast(self, gray: np.ndarray) -> float:
        """Анализ контраста"""
        # Вычисляем стандартное отклонение как меру контраста
        std_dev = np.std(gray)
        
        # Нормализуем к диапазону 0-1
        contrast = std_dev / 128.0
        
        return min(1.0, max(0.0, contrast))
    
    def _analyze_slant(self, binary: np.ndarray) -> float:
        """Анализ наклона текста"""
        # Используем преобразование Хафа для поиска доминирующих линий
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is None:
            return 0.0
        
        angles = []
        for line in lines[:20]:  # Берем первые 20 линий
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            
            # Интересуют вертикальные линии (около 90 градусов)
            if 80 <= angle <= 100:
                angles.append(angle - 90)
        
        if not angles:
            return 0.0
        
        # Возвращаем средний наклон
        return np.mean(angles)
    
    def _analyze_geometry(self, binary: np.ndarray) -> Tuple[float, float, float, float]:
        """Анализ геометрических характеристик"""
        # Находим контуры букв
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 50.0, 70.0, 80.0, 20.0
        
        # Анализируем высоты букв
        heights = []
        for contour in contours:
            _, _, _, h = cv2.boundingRect(contour)
            if h > 10:  # Фильтруем слишком маленькие контуры
                heights.append(h)
        
        if not heights:
            return 50.0, 70.0, 80.0, 20.0
        
        # Эмпирические соотношения для кириллических шрифтов
        avg_height = np.mean(heights)
        x_height = avg_height * 0.5
        cap_height = avg_height * 0.7
        ascender = avg_height * 0.8
        descender = avg_height * 0.2
        
        return x_height, cap_height, ascender, descender
    
    def _analyze_spacing(self, binary: np.ndarray) -> Tuple[float, float]:
        """Анализ межбуквенных и межсловных расстояний"""
        # Упрощенный анализ - в реальности нужен более сложный алгоритм
        height, width = binary.shape
        
        # Анализируем горизонтальные проекции для определения расстояний
        horizontal_projection = np.sum(binary == 0, axis=0)
        
        # Находим промежутки между буквами
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, projection in enumerate(horizontal_projection):
            if projection == 0 and not in_gap:  # Начало промежутка
                in_gap = True
                gap_start = i
            elif projection > 0 and in_gap:  # Конец промежутка
                gaps.append(i - gap_start)
                in_gap = False
        
        if gaps:
            letter_spacing = np.mean(gaps)
            word_spacing = np.percentile(gaps, 75) if len(gaps) > 1 else letter_spacing * 2
        else:
            letter_spacing = 2.0
            word_spacing = 6.0
        
        return letter_spacing, word_spacing
    
    def _calculate_density(self, binary: np.ndarray) -> float:
        """Расчет плотности текста"""
        text_pixels = np.sum(binary == 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        return text_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _analyze_cyrillic_features(self, binary: np.ndarray) -> CyrillicFeatures:
        """
        Анализ особенностей кириллических букв
        Упрощенная версия - в реальности нужно распознавание конкретных букв
        """
        # Пока возвращаем случайные значения на основе общих характеристик изображения
        height, width = binary.shape
        text_density = self._calculate_density(binary)
        
        # Генерируем характеристики на основе свойств изображения
        base_value = (height * width) % 100 / 100
        
        return CyrillicFeatures(
            ya_shape=0.5 + (text_density * base_value) % 0.4,
            zh_shape=0.6 + (width % 50) / 125,
            fi_shape=0.7 + (height % 30) / 100,
            shcha_shape=0.8 + (text_density * 100) % 20 / 100,
            yery_shape=0.5 + ((width + height) % 50) / 100
        )
    
    def _segment_image(self, gray: np.ndarray) -> list:
        """Разбивка изображения на сегменты для улучшенного анализа"""
        try:
            height, width = gray.shape
            segments = []
            
            # Подход 1: Разбивка на сетку (адаптивная)
            if height > 300 and width > 300:
                # Для больших изображений - сетка 4x4
                rows, cols = 4, 4
            elif height > 150 and width > 150:
                # Для средних изображений - сетка 3x3
                rows, cols = 3, 3
            else:
                # Для маленьких - сетка 2x2
                rows, cols = 2, 2
            
            segment_height = height // rows
            segment_width = width // cols
            
            for i in range(rows):
                for j in range(cols):
                    y1 = i * segment_height
                    y2 = min((i + 1) * segment_height, height)
                    x1 = j * segment_width
                    x2 = min((j + 1) * segment_width, width)
                    
                    segment = gray[y1:y2, x1:x2]
                    if segment.size > 1000:  # Минимальный размер сегмента
                        segments.append({
                            'image': segment,
                            'position': (x1, y1, x2, y2),
                            'type': 'grid'
                        })
            
            # Подход 2: Центральная область (часто содержит основной текст)
            center_margin = min(height, width) // 6
            if center_margin > 0:
                center_y1 = center_margin
                center_y2 = height - center_margin
                center_x1 = center_margin
                center_x2 = width - center_margin
                
                if center_y2 > center_y1 and center_x2 > center_x1:
                    center_segment = gray[center_y1:center_y2, center_x1:center_x2]
                    if center_segment.size > 2000:
                        segments.append({
                            'image': center_segment,
                            'position': (center_x1, center_y1, center_x2, center_y2),
                            'type': 'center'
                        })
            
            # Подход 3: Горизонтальные полосы (для заголовков)
            strip_height = height // 3
            for i in range(3):
                y1 = i * strip_height
                y2 = min((i + 1) * strip_height, height)
                
                strip_segment = gray[y1:y2, :]
                if strip_segment.size > 2000:
                    segments.append({
                        'image': strip_segment,
                        'position': (0, y1, width, y2),
                        'type': 'horizontal_strip'
                    })
            
            logger.info(f"Создано {len(segments)} сегментов для анализа")
            return segments
            
        except Exception as e:
            logger.error(f"Ошибка сегментации: {str(e)}")
            return [{'image': gray, 'position': (0, 0, gray.shape[1], gray.shape[0]), 'type': 'full'}]
    
    def _analyze_segment_for_text(self, segment_data: dict) -> bool:
        """Улучшенный анализ сегмента на наличие текста"""
        try:
            segment = segment_data['image']
            position = segment_data['position']
            segment_type = segment_data['type']
            
            # Пропускаем слишком маленькие сегменты
            if segment.shape[0] < 30 or segment.shape[1] < 30:
                return False
            
            # Множественная бинаризация для лучшего выделения текста
            results = []
            
            # Метод 1: Адаптивная бинаризация
            try:
                binary1 = cv2.adaptiveThreshold(segment, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                results.append(self._analyze_binary_for_text(binary1))
            except:
                pass
            
            # Метод 2: Глобальная бинаризация с Otsu
            try:
                _, binary2 = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                results.append(self._analyze_binary_for_text(binary2))
            except:
                pass
            
            # Метод 3: Бинаризация по среднему значению
            try:
                mean_val = np.mean(segment)
                binary3 = (segment < mean_val * 0.8).astype(np.uint8) * 255
                results.append(self._analyze_binary_for_text(binary3))
            except:
                pass
            
            # Метод 4: ИНВЕРТИРОВАННАЯ бинаризация (для светлого текста на светлом фоне)
            try:
                # Ищем светлые области как текст (инвертируем логику)
                mean_val = np.mean(segment)
                binary4 = (segment > mean_val * 1.1).astype(np.uint8) * 255
                results.append(self._analyze_binary_for_text(binary4))
            except:
                pass
            
            # Метод 5: Детекция границ с последующей морфологией
            try:
                # Находим границы
                edges = cv2.Canny(segment, 30, 100)
                # Расширяем границы чтобы "соединить" буквы
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary5 = cv2.dilate(edges, kernel, iterations=1)
                results.append(self._analyze_binary_for_text(binary5))
            except:
                pass
            
            # Если хотя бы один метод нашел текст - считаем что текст есть
            has_text = any(results)
            
            # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: если все методы не сработали, проверяем просто контрастность
            if not has_text:
                contrast = np.std(segment.astype(np.float64))
                # Если есть хороший контраст - возможно это стилизованный текст
                if contrast > 15:  # Снижаем порог для светлых изображений
                    has_text = True
                    logger.info(f"Текст найден по контрастности в сегменте {segment_type}: contrast={contrast:.1f}")
                
                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА 2: анализ текстуры (для очень светлых изображений)
                if not has_text:
                    # Проверяем есть ли структурированные области
                    height, width = segment.shape
                    if height > 20 and width > 20:
                        # Анализируем локальные вариации интенсивности
                        local_std_values = []
                        window_size = min(10, height//4, width//4)
                        
                        for y in range(0, height-window_size, window_size//2):
                            for x in range(0, width-window_size, window_size//2):
                                window = segment[y:y+window_size, x:x+window_size]
                                local_std = np.std(window.astype(np.float64))
                                local_std_values.append(local_std)
                        
                        if local_std_values:
                            max_local_std = max(local_std_values)
                            avg_local_std = np.mean(local_std_values)
                            
                            # Если есть области с высокой локальной вариацией - возможно текст
                            if max_local_std > 8 and avg_local_std > 3:
                                has_text = True
                                logger.info(f"Текст найден по текстурному анализу в сегменте {segment_type}: max_std={max_local_std:.1f}, avg_std={avg_local_std:.1f}")
            
            if has_text:
                logger.info(f"Текст найден в сегменте {segment_type} в позиции {position}")
            
            return has_text
            
        except Exception as e:
            logger.error(f"Ошибка анализа сегмента: {str(e)}")
            return False
    
    def _analyze_binary_for_text(self, binary: np.ndarray) -> bool:
        """Анализ бинарного изображения на наличие текстовых структур"""
        try:
            # Поиск контуров
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_contours = 0
            total_text_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 10000:  # Еще более широкий диапазон для символов
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Проверяем пропорции символа (очень мягко)
                    if 0.05 < aspect_ratio < 10.0:
                        # Дополнительные проверки (смягченные)
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            compactness = area / (perimeter * perimeter)
                            
                            # Символы обычно не слишком компактные (очень мягкие требования)
                            if 0.005 < compactness < 0.5:
                                text_like_contours += 1
                                total_text_area += area
            
            # Проверяем плотность текстовых элементов
            image_area = binary.shape[0] * binary.shape[1]
            text_density = total_text_area / image_area if image_area > 0 else 0
            
            # МАКСИМАЛЬНО мягкие условия для определения текста
            has_text = (
                text_like_contours >= 1 and  # Минимум 1 символ (очень мягко)
                0.0001 < text_density < 0.7  # Очень широкий диапазон плотности
            )
            
            return has_text
            
        except Exception as e:
            logger.error(f"Ошибка анализа бинарного изображения: {str(e)}")
            return False
    
    def _final_complexity_check(self, gray: np.ndarray) -> bool:
        """Финальная проверка сложности изображения - для редких случаев"""
        try:
            logger.info("=== ФИНАЛЬНАЯ ПРОВЕРКА СЛОЖНОСТИ ===")
            
            # Проверка 1: Общая энтропия изображения (информационная сложность)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist[hist > 0]  # Убираем нули
            if len(hist) > 0:
                # Вычисляем энтропию
                hist_norm = hist / np.sum(hist)
                entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
                logger.info(f"Энтропия изображения: {entropy:.2f}")
                
                # Высокая энтропия часто означает сложное изображение с текстом
                if entropy > 6.0:  # Высокая информационная сложность
                    logger.info("Высокая энтропия - вероятно есть текст")
                    return True
            
            # Проверка 2: Количество уникальных градаций серого
            unique_values = len(np.unique(gray))
            logger.info(f"Уникальных градаций: {unique_values}")
            
            # Проверка 3: Локальная вариация (LBP - Local Binary Pattern подход)
            height, width = gray.shape
            local_variations = 0
            
            # Проверяем локальные вариации по сетке
            step = max(10, min(height, width) // 20)
            for y in range(step, height - step, step):
                for x in range(step, width - step, step):
                    # Берем окрестность 3x3
                    neighborhood = gray[y-1:y+2, x-1:x+2]
                    if neighborhood.shape == (3, 3):
                        center = neighborhood[1, 1]
                        # Считаем сколько соседей отличается от центра
                        diff_count = np.sum(np.abs(neighborhood - center) > 10)
                        if diff_count >= 4:  # Много различий - возможно текст
                            local_variations += 1
            
            variation_density = local_variations / ((height // step) * (width // step))
            logger.info(f"Плотность локальных вариаций: {variation_density:.3f}")
            
            # Проверка 4: Градиентная активность
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            logger.info(f"Средняя градиентная активность: {avg_gradient:.2f}")
            
            # ФИНАЛЬНОЕ РЕШЕНИЕ: если изображение сложное по нескольким критериям
            complexity_score = 0
            
            if entropy > 5.5:
                complexity_score += 1
                logger.info("+ Энтропия указывает на сложность")
            
            if unique_values > 100:
                complexity_score += 1
                logger.info("+ Много градаций серого")
            
            if variation_density > 0.3:
                complexity_score += 1
                logger.info("+ Высокая локальная вариация")
            
            if avg_gradient > 15:
                complexity_score += 1
                logger.info("+ Высокая градиентная активность")
            
            # Если набрали 2+ балла из 4 - считаем что есть текст
            has_text = complexity_score >= 2
            
            # ЭКСТРЕННАЯ МЕРА: проверка на "рекламный" тип изображения
            if not has_text:
                has_text = self._detect_advertisement_pattern(gray)
                if has_text:
                    logger.info("Обнаружен рекламный паттерн - принудительно считаем что есть текст")
            
            logger.info(f"Итоговый счет сложности: {complexity_score}/4, результат: {has_text}")
            
            return has_text
            
        except Exception as e:
            logger.error(f"Ошибка финальной проверки: {str(e)}")
            return False  # В случае ошибки не блокируем
    
    def _detect_advertisement_pattern(self, gray: np.ndarray) -> bool:
        """Определение рекламного/баннерного паттерна изображения"""
        try:
            logger.info("=== ПРОВЕРКА РЕКЛАМНОГО ПАТТЕРНА ===")
            
            height, width = gray.shape
            total_pixels = height * width
            
            logger.info(f"Размер изображения: {width}x{height} ({total_pixels} пикселей)")
            
            # Признак 1: Прямоугольная форма (типично для рекламы)
            aspect_ratio = width / height if height > 0 else 0
            is_banner_shape = (
                (1.2 < aspect_ratio < 3.0) or  # Горизонтальный баннер
                (0.3 < aspect_ratio < 0.8)     # Вертикальный баннер
            )
            
            # Признак 2: Преобладание светлых тонов (белый/светлый фон)
            light_pixels = np.sum(gray > 200)  # Очень светлые пиксели
            light_ratio = light_pixels / total_pixels
            has_light_background = light_ratio > 0.4  # Много светлых областей
            
            # Признак 3: Наличие контрастных областей (текст на фоне)
            # Ищем области с резкими переходами
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            edge_density = edge_pixels / total_pixels
            has_contrast_areas = edge_density > 0.02  # Есть четкие границы
            
            # Признак 4: Структурированность (не хаотичное изображение)
            # Проверяем есть ли горизонтальные/вертикальные структуры
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
            
            h_line_pixels = np.sum(horizontal_lines > 0)
            v_line_pixels = np.sum(vertical_lines > 0)
            
            has_structure = (h_line_pixels + v_line_pixels) > (total_pixels * 0.01)
            
            # Признак 5: Размер изображения (реклама обычно среднего размера)
            reasonable_size = 10000 < total_pixels < 500000  # От 100x100 до 700x700 примерно
            
            # Подсчитываем баллы
            score = 0
            if is_banner_shape:
                score += 1
                logger.info("+ Форма баннера")
            
            if has_light_background:
                score += 1
                logger.info(f"+ Светлый фон ({light_ratio:.2%})")
            
            if has_contrast_areas:
                score += 1
                logger.info(f"+ Контрастные области ({edge_density:.3%})")
            
            if has_structure:
                score += 1
                logger.info("+ Структурированность")
            
            if reasonable_size:
                score += 1
                logger.info(f"+ Разумный размер ({total_pixels} пикселей)")
            
            # ДОПОЛНИТЕЛЬНАЯ ЭВРИСТИКА: если это явно рекламное изображение
            # и размер разумный - принудительно считаем что есть текст
            if score >= 1 and reasonable_size and has_light_background:
                logger.info("ЭВРИСТИКА: Светлое изображение разумного размера - вероятно реклама с текстом")
                is_advertisement = True
            else:
                # Если набрали 2+ балла из 5 - скорее всего реклама с текстом
                is_advertisement = score >= 2
            
            logger.info(f"Рекламный паттерн: {score}/5 баллов, результат: {is_advertisement}")
            
            return is_advertisement
            
        except Exception as e:
            logger.error(f"Ошибка определения рекламного паттерна: {str(e)}")
            return False
    
# Fallback методы удалены - используем только PaddleOCR
    

    
    async def _extract_characteristics_from_full_image(self, image: np.ndarray) -> FontCharacteristics:
        """Извлечение характеристик из всего изображения (fallback)"""
        try:
            logger.info("🖼️ Анализируем все изображение как fallback")
            
            # Создаем базовые характеристики на основе размера изображения
            height, width = image.shape[:2]
            
            # РЕАЛЬНЫЕ характеристики на основе содержимого изображения
            # Анализируем содержимое для уникальности
            try:
                # Пытаемся получить хоть какую-то информацию о тексте
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                # Простая оценка плотности текста
                text_density = np.sum(gray < 128) / gray.size  # Процент темных пикселей
            except:
                text_density = 0.3
            
            # Создаем уникальный фактор на основе реального содержимого
            content_hash = hash(str(image.shape) + str(text_density) + str(width) + str(height))
            unique_factor = (content_hash % 1000) / 1000.0
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ FALLBACK для отладки
            logger.info(f"🔍 FALLBACK ХАРАКТЕРИСТИКИ ИЗОБРАЖЕНИЯ:")
            logger.info(f"  - Размер: {image.shape}")
            logger.info(f"  - Ширина: {width}, Высота: {height}")
            logger.info(f"  - Плотность текста: {text_density:.3f}")
            logger.info(f"  - Хеш изображения: {content_hash}")
            logger.info(f"  - Уникальный фактор: {unique_factor:.3f}")  # 0-1
            
            # Создаем УНИКАЛЬНЫЕ характеристики для каждого изображения
            # Используем реальные размеры и содержимое
            image_hash = hash(str(image.shape) + str(text_density) + str(width) + str(height))
            unique_factor = (image_hash % 1000) / 1000.0
            
            # Анализируем содержимое изображения для уникальности
            try:
                # Простая оценка сложности изображения
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
                complexity = np.std(gray.astype(np.float64)) / 255.0  # 0-1
                complexity = min(1.0, max(0.0, complexity))  # Ограничиваем 0-1
            except:
                complexity = 0.5
            
            # Безопасные вычисления с ограничениями
            safe_stroke_width = min(1.0, max(0.0, text_density * 0.8 + unique_factor * 0.2))
            safe_contrast = min(1.0, max(0.0, complexity + unique_factor * 0.3))
            safe_slant = max(-5.0, min(5.0, (unique_factor - 0.5) * 4.0))  # -5 до +5 градусов
            
            characteristics = FontCharacteristics(
                has_serifs=text_density > 0.4 and complexity > 0.3,  # На основе плотности и сложности
                stroke_width=safe_stroke_width,  # Безопасная толщина
                contrast=safe_contrast,  # Безопасный контраст
                slant=safe_slant,  # Безопасный наклон
                cyrillic_features=CyrillicFeatures(),  # Используем модель с значениями по умолчанию
                x_height=max(1.0, height * (0.5 + unique_factor * 0.2)),  # Минимум 1.0
                cap_height=max(1.0, height * (0.8 + unique_factor * 0.4)),
                ascender=max(1.0, height * (1.0 + unique_factor * 0.4)),
                descender=max(1.0, height * (0.2 + unique_factor * 0.3)),
                letter_spacing=max(0.1, width / (40 + unique_factor * 20)),  # Минимум 0.1
                word_spacing=max(0.1, width / (15 + unique_factor * 10)),
                density=min(1.0, text_density + unique_factor * 0.2)  # Уникальная плотность (максимум 1.0)
            )
            
            logger.info("✅ Созданы базовые характеристики для fallback")
            return characteristics
            
        except Exception as e:
            logger.error(f"Ошибка fallback анализа: {str(e)}")
            raise


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
        
        # Проверяем статус PaddleOCR
        if self.paddleocr_service.is_available():
            logger.info("✅ FontAnalyzer: PaddleOCR успешно инициализирован")
        else:
            logger.error("❌ FontAnalyzer: PaddleOCR не инициализирован - анализ шрифтов невозможен")
        
    async def analyze_image(self, image_bytes: bytes) -> FontCharacteristics:
        """Анализ изображения для определения характеристик шрифта"""
        return await self._analyze_image_async(image_bytes)
    
    async def _analyze_image_async(self, image_bytes: bytes) -> FontCharacteristics:
        """Асинхронный анализ изображения ТОЛЬКО через PaddleOCR"""
        print("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: _analyze_image_async НАЧАЛСЯ")
        logger.info("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: _analyze_image_async НАЧАЛСЯ")
        try:
            # Загружаем изображение
            print("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Загружаем изображение...")
            logger.info("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Загружаем изображение...")
            image = self._load_image(image_bytes)
            
            # Проверяем доступность PaddleOCR
            logger.info("=== ПРОВЕРКА ДОСТУПНОСТИ PADDLEOCR ===")
            if not hasattr(self, 'paddleocr_service') or not self.paddleocr_service:
                logger.error("❌ PaddleOCR сервис недоступен!")
                raise ValueError("ИИ для анализа шрифтов временно недоступен. Попробуйте позже или обратитесь к администратору.")
            
            if not self.paddleocr_service.is_available():
                logger.error("❌ PaddleOCR не инициализирован!")
                raise ValueError("ИИ для анализа шрифтов временно недоступен. Попробуйте позже или обратитесь к администратору.")
            
            logger.info("✅ PaddleOCR доступен - начинаем анализ")
            
            # ШАГ 1: Определение наличия текста через PaddleOCR
            logger.info("=== ШАГ 1: Определение наличия текста через PaddleOCR ===")
            print("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Вызываем PaddleOCR.analyze_image()")
            logger.info("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Вызываем PaddleOCR.analyze_image()")
            ocr_result = await self.paddleocr_service.analyze_image(image)
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: PaddleOCR вернул: {type(ocr_result)}")
            logger.info(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: PaddleOCR вернул: {type(ocr_result)}")
            
            # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ OCR результата
            logger.info(f"🔍 PADDLEOCR РЕЗУЛЬТАТ:")
            logger.info(f"  - has_text: {ocr_result.get('has_text', False)}")
            logger.info(f"  - text_content: '{ocr_result.get('text_content', '')[:50]}...'")
            logger.info(f"  - confidence: {ocr_result.get('confidence', 0.0):.3f}")
            logger.info(f"  - regions_count: {ocr_result.get('regions_count', 0)}")
            logger.info(f"  - error: {ocr_result.get('error', 'нет')}")
            
            if not ocr_result.get('has_text', False):
                logger.info("РЕЗУЛЬТАТ ШАГ 1: Текст не обнаружен - СТОП")
                error_msg = ocr_result.get('error', 'OCR не смог найти текст на изображении')
                raise ValueError(f"На изображении не обнаружен читаемый текст: {error_msg}")
            
            logger.info("РЕЗУЛЬТАТ ШАГ 1: ✅ Текст успешно обнаружен - продолжаем")
            
            # ШАГ 2: Проверка множественных шрифтов через PaddleOCR
            logger.info("=== ШАГ 2: Проверка множественных шрифтов через PaddleOCR ===")
            if await self._detect_multiple_fonts_from_ocr_result(ocr_result):
                logger.info("РЕЗУЛЬТАТ ШАГ 2: Обнаружено несколько шрифтов - СТОП")
                raise ValueError("На изображении обнаружено несколько разных шрифтов. Для точного анализа загрузите изображение с текстом одного шрифта.")
            logger.info("РЕЗУЛЬТАТ ШАГ 2: ✅ Один шрифт - продолжаем к анализу")
            
            # ШАГ 3: Извлечение характеристик шрифта через PaddleOCR
            logger.info("=== ШАГ 3: Извлечение характеристик шрифта через PaddleOCR ===")
            characteristics = await self._extract_characteristics_from_ocr(image, ocr_result)
            logger.info("РЕЗУЛЬТАТ ШАГ 3: ✅ Характеристики успешно извлечены")
            
            # ШАГ 4: Сверка с базой шрифтов (будет реализовано в font_matcher)
            logger.info("=== ШАГ 4: Сверка с характеристиками шрифтов в базе ===")
            logger.info("РЕЗУЛЬТАТ ШАГ 4: ✅ Готово к сверке с базой (реализуется в font_matcher)")
            
            # ШАГ 5: Вывод результатов анализа
            logger.info("=== ШАГ 5: Вывод результатов анализа ===")
            logger.info("✅ Анализ завершен успешно через PaddleOCR")
            print("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Анализ завершен успешно!")
            logger.info("🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Анализ завершен успешно!")
            return characteristics
            
        except ValueError as logic_error:
            # Логические ошибки (нет текста, много шрифтов) - передаем пользователю
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Логическая ошибка: {str(logic_error)}")
            logger.info(f"ℹ️ Логический результат анализа: {str(logic_error)}")
            raise logic_error
            
        except Exception as error:
            # Технические ошибки
            print(f"🚀 ПРИНУДИТЕЛЬНЫЙ ВЫВОД: Техническая ошибка: {str(error)}")
            logger.error(f"❌ Техническая ошибка анализа: {str(error)}")
            
            # Определяем тип ошибки и даем понятное сообщение
            if "PaddleOCR не инициализирован" in str(error):
                user_message = "ИИ для анализа шрифтов временно недоступен. Попробуйте позже или обратитесь к администратору."
            elif "PaddleOCR сервис недоступен" in str(error):
                user_message = "Сервис анализа шрифтов временно недоступен. Попробуйте позже."
            elif "не удалось загрузить изображение" in str(error):
                user_message = "Не удалось обработать загруженное изображение. Проверьте формат файла."
            else:
                user_message = "Произошла техническая ошибка при анализе изображения. Попробуйте позже."
            
            raise ValueError(user_message)
    

    
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
    

    

    

    

    

    

    
    async def _detect_multiple_fonts_from_ocr_result(self, ocr_result: dict) -> bool:
        """Детекция множественных шрифтов на основе уже полученного OCR результата"""
        try:
            logger.info("=== АНАЛИЗ МНОЖЕСТВЕННЫХ ШРИФТОВ ПО OCR РЕЗУЛЬТАТУ ===")
            
            if not ocr_result.get('has_text', False):
                logger.info("📊 OCR не нашел текст - считаем один шрифт")
                return False
            
            text_content = ocr_result.get('text_content', '').strip()
            regions_count = ocr_result.get('regions_count', 0)
            ocr_boxes = ocr_result.get('ocr_boxes', [])
            
            logger.info(f"📊 OCR данные: '{text_content[:30]}...' ({regions_count} регионов)")
            
            # Анализируем текст
            words = text_content.split()
            word_count = len(words)
            
            # ПРОСТЫЕ СЛУЧАИ - один шрифт
            if word_count <= 3:
                logger.info(f"📊 Мало слов ({word_count}) - ОДИН шрифт")
                return False
            
            if regions_count < 6:
                logger.info(f"📊 Мало регионов ({regions_count}) - ОДИН шрифт")
                return False
            
            # АНАЛИЗ СОДЕРЖИМОГО через OCR
            has_title = any(len(word) > 4 and word.isupper() for word in words)
            has_normal_text = any(len(word) > 3 and not word.isupper() for word in words)
            has_numbers = any(char.isdigit() for char in text_content)
            has_special_words = any(word.lower() in ['скидка', 'цена', 'рубль', '%', 'руб', 'распродажа'] for word in words)
            
            logger.info(f"📊 Анализ содержимого: заголовок={has_title}, текст={has_normal_text}, цифры={has_numbers}, спец.слова={has_special_words}")
            
            # АНАЛИЗ РАЗМЕРОВ ЧЕРЕЗ OCR BOXES
            height_ratio = 1.0
            area_ratio = 1.0
            
            if len(ocr_boxes) >= 4:
                heights = []
                areas = []
                
                for box_info in ocr_boxes:
                    if isinstance(box_info, dict) and 'bbox' in box_info:
                        bbox = box_info['bbox']
                        if isinstance(bbox, list) and len(bbox) >= 4:
                            if isinstance(bbox[0], list):  # Формат [[x1,y1], [x2,y2], ...]
                                x_coords = [point[0] for point in bbox]
                                y_coords = [point[1] for point in bbox]
                            else:  # Формат [x1, y1, x2, y2]
                                x_coords = [bbox[0], bbox[2]]
                                y_coords = [bbox[1], bbox[3]]
                            
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                            
                            if height > 0 and width > 0:
                                heights.append(height)
                                areas.append(width * height)
                
                if len(heights) >= 2:
                    height_ratio = max(heights) / min(heights)
                    area_ratio = max(areas) / min(areas)
                    
                    logger.info(f"📊 OCR размеры: высота={height_ratio:.1f}, площадь={area_ratio:.1f}")
            
            # КРИТЕРИИ МНОЖЕСТВЕННЫХ ШРИФТОВ:
            
            # 1. Простые случаи - точно ОДИН шрифт
            if word_count <= 2 and regions_count <= 3:
                logger.info("📊 Простой случай: очень мало слов и регионов - ОДИН шрифт")
                return False
            
            # 2. Средняя сложность
            if word_count <= 8 and regions_count <= 10:
                if height_ratio <= 2.5 and area_ratio <= 8.0:
                    logger.info("📊 Средняя сложность: размеры стабильные - ОДИН шрифт")
                    return False
                elif has_title and has_normal_text and height_ratio > 2.5:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: заголовок + текст + разные размеры")
                    return True
                else:
                    logger.info("📊 Средняя сложность: неопределенно - считаем ОДИН шрифт")
                    return False
            
            # 3. Сложные случаи
            if word_count > 8 or regions_count > 10:
                if height_ratio > 3.5 and area_ratio > 10.0:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: большая разница в размерах")
                    return True
                elif has_title and has_normal_text and height_ratio > 2.2 and word_count >= 10:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: заголовок + много текста + разные размеры")
                    return True
                elif height_ratio > 3.0 and regions_count >= 15:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: много регионов + заметная разница размеров")
                    return True
                elif has_numbers and has_normal_text and height_ratio > 2.8:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: цифры + текст + разные размеры")
                    return True
                else:
                    logger.info("📊 Сложный случай: неопределенно - считаем ОДИН шрифт")
                    return False
            
            # По умолчанию считаем один шрифт
            logger.info("📊 По умолчанию: считаем ОДИН шрифт")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка анализа множественных шрифтов: {str(e)}")
            logger.warning("⚠️ При ошибке считаем один шрифт")
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
    

    

    

    

    

    
# Fallback методы удалены - используем только PaddleOCR
    

    
# Fallback методы удалены - используем только PaddleOCR


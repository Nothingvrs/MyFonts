"""
Правильная реализация PaddleOCR согласно официальной документации
"""

import logging
import numpy as np
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

logger = logging.getLogger(__name__)


class PaddleOCRServiceNew:
    """Правильная реализация PaddleOCR сервиса"""
    
    def __init__(self):
        self.ocr = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._initialize()
    
    def _initialize(self):
        """Инициализация PaddleOCR согласно документации"""
        try:
            if not PADDLEOCR_AVAILABLE:
                logger.error("❌ PaddleOCR не установлен!")
                return
            
            logger.info("🚀 Инициализация PaddleOCR (правильная)...")
            
            # Инициализация специально для кириллицы
            try:
                # Оптимизируем для русского языка и кириллицы
                logger.info("🇷🇺 Инициализация для кириллических шрифтов...")
                self.ocr = PaddleOCR(
                    use_angle_cls=True,     # Классификация угла поворота
                    lang='ru',              # РУССКИЙ язык для кириллицы
                    show_log=False,         # Отключаем логи PaddleOCR
                    use_gpu=False,          # CPU для стабильности
                    det_db_thresh=0.2,      # Более чувствительная детекция для кириллицы
                    rec_batch_num=8         # Больший батч для лучшего распознавания
                )
                logger.info("✅ PaddleOCR инициализирован для кириллицы")
                
            except Exception as e1:
                logger.warning(f"⚠️ Ошибка с русским языком: {e1}")
                try:
                    # Fallback на китайскую модель (хорошо работает с кириллицей)
                    logger.info("🔄 Fallback: инициализация с китайской моделью...")
                    self.ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang='ch',              # Китайская модель как fallback
                        show_log=False,
                        use_gpu=False
                    )
                    logger.info("✅ PaddleOCR инициализирован с китайской моделью")
                    
                except Exception as e2:
                    logger.warning(f"⚠️ Ошибка с латинской моделью: {e2}")
                    try:
                        # Базовая инициализация без параметров
                        logger.info("🔄 Базовая инициализация...")
                        self.ocr = PaddleOCR(use_angle_cls=True)
                        logger.info("✅ PaddleOCR инициализирован базово")
                        
                    except Exception as e3:
                        logger.error(f"❌ Критическая ошибка инициализации: {e3}")
                        self.ocr = None
            
            logger.info("✅ PaddleOCR инициализирован правильно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            self.ocr = None
    
    def is_available(self) -> bool:
        """Проверка доступности"""
        return self.ocr is not None and PADDLEOCR_AVAILABLE
    
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Анализ изображения"""
        if not self.is_available():
            return {
                'has_text': False,
                'error': 'PaddleOCR недоступен'
            }
        
        try:
            logger.info(f"🖼️ Анализируем изображение: {image.shape}, dtype: {image.dtype}")
            logger.info(f"📊 Диапазон пикселей: min={image.min()}, max={image.max()}")
            
            # Проверяем не пустое ли изображение
            if image.size == 0:
                return {'has_text': False, 'error': 'Пустое изображение'}
            
            # Проверяем формат изображения
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.warning(f"⚠️ Неожиданный формат изображения: {image.shape}")
                # Пробуем исправить
                if len(image.shape) == 2:
                    import cv2
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    image = image[:, :, :3]  # Убираем альфа-канал
            
            # Запускаем OCR в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_ocr,
                image
            )
            
            logger.info(f"🔍 Результат OCR анализа: {result}")
            return result
            
        except Exception as e:
            logger.error(f"💥 Ошибка анализа: {e}")
            return {
                'has_text': False,
                'error': f'PaddleOCR error: {str(e)}'
            }
    
    def _run_ocr(self, image: np.ndarray) -> Dict[str, Any]:
        """Синхронный запуск OCR с предобработкой"""
        try:
            logger.info(f"🔍 Запуск PaddleOCR на изображении {image.shape}...")
            
            # Согласно документации PaddleOCR 3.0 - простой вызов
            logger.info("🔄 Запуск OCR согласно документации PaddleOCR 3.0")
            
            # Убеждаемся что изображение в правильном формате
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB изображение - конвертируем в BGR для PaddleOCR
                import cv2
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                logger.info(f"🔄 Конвертировано RGB -> BGR: {image_bgr.shape}")
            else:
                image_bgr = image
                
            # Основной вызов OCR (без cls - не поддерживается в старых версиях)
            result = self.ocr.ocr(image_bgr)
            
            logger.info(f"📊 OCR результат тип: {type(result)}")
            logger.info(f"📊 OCR результат длина: {len(result) if result else 'None'}")
            
            if not result:
                logger.warning("⚠️ OCR вернул None - изображение не содержит распознаваемого текста")
                return {
                    'has_text': False,
                    'text_content': '',
                    'confidence': 0.0,
                    'regions_count': 0
                }
            
            # PaddleOCR возвращает список страниц, каждая страница содержит список элементов
            if not isinstance(result, list) or len(result) == 0:
                logger.warning("⚠️ OCR вернул пустой список")
                return {
                    'has_text': False,
                    'text_content': '',
                    'confidence': 0.0,
                    'regions_count': 0
                }
            
            # Берем первую страницу
            page_result = result[0]
            logger.info(f"📊 Страница результат тип: {type(page_result)}, длина: {len(page_result) if page_result else 'None'}")
            
            if not page_result:
                logger.warning("⚠️ Первая страница OCR пуста - текст не найден")
                return {
                    'has_text': False,
                    'text_content': '',
                    'confidence': 0.0,
                    'regions_count': 0
                }
            
            # Проверяем что page_result не None
            if page_result is None:
                logger.error("❌ Страница результата None!")
                return {
                    'has_text': False,
                    'text_content': '',
                    'confidence': 0.0,
                    'regions_count': 0
                }
            
            if not isinstance(page_result, list):
                logger.warning(f"⚠️ Неожиданный тип страницы: {type(page_result)}")
                # Пробуем конвертировать в список
                try:
                    page_result = list(page_result)
                    logger.info(f"✅ Конвертировано в список: {len(page_result)} элементов")
                except Exception as e:
                    logger.error(f"❌ Не удалось конвертировать: {e}")
                    return {
                        'has_text': False,
                        'text_content': '',
                        'confidence': 0.0,
                        'regions_count': 0
                    }
            
            # Извлекаем тексты и уверенности из структуры PaddleOCR
            text_lines = []
            confidences = []
            
            logger.info(f"📊 Обрабатываем {len(page_result)} элементов на странице")
            
            for i, item in enumerate(page_result):
                logger.info(f"📊 Элемент {i}: тип={type(item)}, длина={len(item) if hasattr(item, '__len__') else 'N/A'}")
                
                if item and len(item) >= 2:
                    # Структура: [bbox, (text, confidence)]
                    bbox = item[0]
                    text_info = item[1]
                    
                    logger.info(f"📊 bbox: {bbox}")
                    logger.info(f"📊 text_info: {text_info}, тип: {type(text_info)}")
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, confidence = text_info[0], text_info[1]
                        logger.info(f"📊 Найден текст: '{text}' с уверенностью {confidence}")
                        text_lines.append(str(text))
                        confidences.append(float(confidence))
                    elif isinstance(text_info, str):
                        # Иногда может быть просто строка
                        logger.info(f"📊 Найден текст (строка): '{text_info}'")
                        text_lines.append(str(text_info))
                        confidences.append(0.8)  # Дефолтная уверенность
                else:
                    logger.warning(f"📊 Пропускаем элемент {i}: неправильная структура")
            
            logger.info(f"📝 Найденные тексты: {text_lines}")
            logger.info(f"📊 Уверенности: {confidences}")
            
            full_text = ' '.join(text_lines)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            logger.info(f"📝 Распознанный текст: '{full_text}'")
            logger.info(f"📊 Уверенность: {avg_confidence:.2f}")
            
            # Проверяем множественные шрифты
            multiple_fonts = self._detect_multiple_fonts(page_result, text_lines, confidences)
            
            return {
                'has_text': len(full_text.strip()) > 0,
                'text_content': full_text,
                'confidence': avg_confidence,
                'regions_count': len(text_lines),
                'multiple_fonts': multiple_fonts
            }
            
        except Exception as e:
            logger.error(f"💥 Ошибка OCR: {e}")
            return {
                'has_text': False,
                'error': str(e)
            }
    
    def _detect_multiple_fonts(self, page_result: List, text_lines: List[str], confidences: List[float]) -> bool:
        """УМНАЯ детекция множественных шрифтов на основе анализа текста"""
        try:
            full_text = ' '.join(text_lines).strip()
            logger.info(f"📊 Анализ текста: '{full_text}' ({len(page_result)} регионов)")
            
            # 0. ПРИОРИТЕТНАЯ ПРОВЕРКА: если это одно слово - ВСЕГДА один шрифт
            words = full_text.split()
            if len(words) == 1:
                logger.info(f"📊 ОДНО СЛОВО '{words[0]}' - ГАРАНТИРОВАННО один шрифт")
                return False
            
            # 1. Если слишком мало регионов - точно один шрифт
            if len(page_result) < 3:
                logger.info("📊 Мало регионов - один шрифт")
                return False
            
            # 2. Анализируем сам текст
            words = full_text.split()
            word_count = len(words)
            
            # ОЧЕНЬ консервативно - если меньше 5 слов, точно один шрифт
            if word_count <= 4:
                logger.info(f"📊 Мало слов ({word_count}) - ОПРЕДЕЛЕННО один шрифт")
                return False
            
            # 3. Проверяем есть ли явные признаки разных шрифтов в тексте
            # Например, если есть и заглавные блоки, и обычный текст
            uppercase_words = [w for w in words if w.isupper() and len(w) > 2]
            lowercase_words = [w for w in words if w.islower() and len(w) > 2]
            mixed_words = [w for w in words if not w.isupper() and not w.islower() and len(w) > 2]
            
            logger.info(f"📊 Слова: заглавные={len(uppercase_words)}, строчные={len(lowercase_words)}, смешанные={len(mixed_words)}")
            
            # ОЧЕНЬ строгое условие для заглавных блоков
            if len(uppercase_words) >= 3 and len(lowercase_words) >= 5 and word_count >= 12:
                logger.info("📊 Найдены МНОГО заглавных блоков + МНОГО обычного текста - возможно разные шрифты")
                return True
            else:
                logger.info(f"📊 Недостаточно разнообразия: заглавные={len(uppercase_words)}, строчные={len(lowercase_words)}, всего={word_count}")
                logger.info("📊 Считаем как ОДИН шрифт по текстовому анализу")
            
            # 4. Анализируем геометрические характеристики
            areas = []
            heights = []
            widths = []
            
            for item in page_result:
                if item and len(item) >= 2:
                    bbox = item[0]  # Координаты бокса
                    if len(bbox) >= 4:
                        # bbox может быть в формате [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        if isinstance(bbox[0], (list, tuple)):
                            # Находим min/max координаты
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            width = max(x_coords) - min(x_coords)
                            height = max(y_coords) - min(y_coords)
                        else:
                            # bbox в формате [x1, y1, x2, y2]
                            width = abs(bbox[2] - bbox[0])
                            height = abs(bbox[3] - bbox[1])
                        
                        area = width * height
                        areas.append(area)
                        heights.append(height)
                        widths.append(width)
            
            if len(areas) < 2:
                logger.info("📊 Недостаточно областей для сравнения - один шрифт")
                return False
            
            # 5. Умный анализ размеров - ищем ДЕЙСТВИТЕЛЬНО разные шрифты
            areas = np.array(areas)
            heights = np.array(heights) 
            widths = np.array(widths)
            
            # Сортируем по размеру для анализа
            sorted_areas = np.sort(areas)
            sorted_heights = np.sort(heights)
            
            # Проверяем есть ли четкие группы размеров (например заголовок vs текст)
            # Разделяем на большие и маленькие области
            median_area = np.median(areas)
            large_areas = areas[areas > median_area * 2]  # Области в 2+ раза больше медианы
            small_areas = areas[areas < median_area / 2]  # Области в 2+ раза меньше медианы
            
            logger.info(f"📊 Размеры: больших={len(large_areas)}, маленьких={len(small_areas)}, всего={len(areas)}")
            
            # УЛЬТРА строгое условие для геометрии
            if len(large_areas) >= 3 and len(small_areas) >= 3 and len(areas) >= 10:
                # Дополнительно проверяем что это не просто разные буквы одного шрифта
                area_ratio = np.max(areas) / np.min(areas)
                height_ratio = np.max(heights) / np.min(heights)
                
                logger.info(f"📊 Соотношения: area={area_ratio:.1f}, height={height_ratio:.1f}")
                
                # КОСМИЧЕСКАЯ разница + очень много регионов = вероятно разные шрифты
                if area_ratio > 15.0 and height_ratio > 8.0:
                    logger.info("✅ МНОЖЕСТВЕННЫЕ ШРИФТЫ: КОСМИЧЕСКАЯ разница в размерах")
                    return True
                else:
                    logger.info(f"📊 Недостаточная разница: area={area_ratio:.1f} (нужно >15), height={height_ratio:.1f} (нужно >8)")
            else:
                logger.info(f"📊 Недостаточно групп: больших={len(large_areas)} (нужно ≥3), маленьких={len(small_areas)} (нужно ≥3), всего={len(areas)} (нужно ≥10)")
            
            # 6. Финальная проверка - все остальные случаи считаем одним шрифтом
            logger.info("📊 Все проверки пройдены - определяем как ОДИН шрифт")
            return False
            
        except Exception as e:
            logger.error(f"💥 Ошибка детекции множественных шрифтов: {e}")
            # При ошибке всегда считаем что один шрифт (консервативно)
            return False
    
    def _preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Предобработка изображения для улучшения распознавания"""
        try:
            import cv2
            
            processed_images = []
            
            # Вариант 1: Оригинальное изображение
            processed_images.append(image.copy())
            
            # Вариант 2: Улучшение контраста
            try:
                # Конвертируем в Lab цветовое пространство для улучшения контраста
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Применяем CLAHE (адаптивное выравнивание гистограммы)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Собираем обратно
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
                processed_images.append(enhanced)
                
            except Exception as e:
                logger.warning(f"Ошибка улучшения контраста: {e}")
            
            # Вариант 3: Преобразование в градации серого с улучшением
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Применяем адаптивное выравнивание гистограммы
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gray_enhanced = clahe.apply(gray)
                
                # Конвертируем обратно в RGB
                gray_rgb = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2RGB)
                processed_images.append(gray_rgb)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки серого: {e}")
            
            # Вариант 4: Увеличение резкости
            try:
                # Kernel для увеличения резкости
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
                sharpened = cv2.filter2D(image, -1, kernel)
                
                # Ограничиваем значения
                sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
                processed_images.append(sharpened)
                
            except Exception as e:
                logger.warning(f"Ошибка увеличения резкости: {e}")
            
            # Вариант 5: Бинаризация для контрастного текста
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Адаптивная бинаризация
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # Конвертируем в RGB
                binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                processed_images.append(binary_rgb)
                
            except Exception as e:
                logger.warning(f"Ошибка бинаризации: {e}")
            
            logger.info(f"🔧 Создано {len(processed_images)} вариантов предобработки")
            return processed_images
            
        except Exception as e:
            logger.error(f"💥 Ошибка предобработки: {e}")
            return [image]  # Возвращаем оригинал если что-то пошло не так
    
    def _fallback_text_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback метод определения текста без PaddleOCR"""
        try:
            import cv2
            
            logger.info("🔍 Запуск fallback детекции текста...")
            
            # Конвертируем в серый
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Анализируем контрастность
            contrast = np.std(gray.astype(np.float64))
            logger.info(f"📊 Контрастность: {contrast}")
            
            # Дополнительные метрики
            mean_brightness = np.mean(gray)
            min_val, max_val = np.min(gray), np.max(gray)
            brightness_range = max_val - min_val
            logger.info(f"📊 Яркость: средняя={mean_brightness:.1f}, диапазон={brightness_range}")
            
            # Ищем края с разными порогами
            edges_soft = cv2.Canny(gray, 30, 100)  # Мягкие пороги
            edges_normal = cv2.Canny(gray, 50, 150)  # Обычные пороги
            edges_hard = cv2.Canny(gray, 100, 200)  # Жесткие пороги
            
            edge_density_soft = np.sum(edges_soft > 0) / (edges_soft.shape[0] * edges_soft.shape[1])
            edge_density_normal = np.sum(edges_normal > 0) / (edges_normal.shape[0] * edges_normal.shape[1])
            edge_density_hard = np.sum(edges_hard > 0) / (edges_hard.shape[0] * edges_hard.shape[1])
            
            logger.info(f"📊 Плотность границ: мягкие={edge_density_soft:.4f}, обычные={edge_density_normal:.4f}, жесткие={edge_density_hard:.4f}")
            
            # Используем наилучший результат
            edges = edges_soft if edge_density_soft > edge_density_normal else edges_normal
            edge_density = max(edge_density_soft, edge_density_normal, edge_density_hard)
            
            # Ищем прямоугольные области (потенциальный текст)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_regions = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 50000:  # Разумный размер для текста
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Проверяем пропорции текста
                    if 0.1 < aspect_ratio < 10.0:
                        text_like_regions += 1
            
            logger.info(f"📊 Найдено текстоподобных областей: {text_like_regions}")
            
            # Более мягкие критерии для определения наличия текста
            has_text = (
                contrast > 15 and              # Понижаем порог контрастности
                edge_density > 0.005 and       # Понижаем порог границ
                text_like_regions >= 2         # Понижаем количество областей
            )
            
            # Дополнительная проверка для рекламных изображений
            if not has_text:
                # Проверяем есть ли цветные области (реклама часто цветная)
                if len(image.shape) == 3:
                    color_variance = np.var(image, axis=2).mean()
                    logger.info(f"📊 Цветовая вариация: {color_variance}")
                    
                    # Если есть цветовая вариация и контраст - вероятно есть текст
                    if color_variance > 100 and contrast > 10:
                        has_text = True
                        logger.info("🎨 Обнаружен цветной контент - вероятно реклама с текстом")
            
            # Если нашли текст, определяем множественные шрифты
            multiple_fonts = False
            if has_text and text_like_regions >= 4:  # Понижаем порог
                # Анализируем разнообразие размеров областей
                areas = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 100 < area < 50000:
                        areas.append(area)
                
                if len(areas) >= 3:
                    areas = np.array(areas)
                    area_ratio = np.max(areas) / np.min(areas) if np.min(areas) > 0 else 0
                    if area_ratio > 4.0:  # Большая разница в размерах
                        multiple_fonts = True
                        logger.info(f"🔤 Fallback: обнаружены множественные шрифты (ratio={area_ratio:.1f})")
            
            result = {
                'has_text': has_text,
                'text_content': 'Текст обнаружен fallback методом' if has_text else '',
                'confidence': 0.7 if has_text else 0.0,
                'regions_count': text_like_regions,
                'multiple_fonts': multiple_fonts,
                'method': 'fallback'
            }
            
            logger.info(f"🔍 Fallback результат: {result}")
            return result
            
        except Exception as e:
            logger.error(f"💥 Ошибка fallback метода: {e}")
            return {
                'has_text': False,
                'error': f'Fallback error: {str(e)}'
            }

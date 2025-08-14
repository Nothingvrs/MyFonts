"""
Сервис анализа шрифтов с использованием OpenCV и PIL
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

logger = logging.getLogger(__name__)


class FontAnalyzer:
    """Анализатор шрифтов на основе OpenCV"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def analyze_image(self, image_bytes: bytes) -> FontCharacteristics:
        """
        Анализ изображения и извлечение характеристик шрифта
        """
        try:
            # Выполняем анализ в отдельном потоке для неблокирующей работы
            loop = asyncio.get_event_loop()
            characteristics = await loop.run_in_executor(
                self.executor, 
                self._analyze_image_sync, 
                image_bytes
            )
            return characteristics
            
        except Exception as e:
            logger.error(f"Ошибка при анализе изображения: {str(e)}")
            raise
    
    def _analyze_image_sync(self, image_bytes: bytes) -> FontCharacteristics:
        """Синхронный анализ изображения"""
        
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
        
        # ШАГ 2: Проверяем наличие текста (основная проверка)
        logger.info("=== ШАГ 2: Основная проверка наличия текста ===")
        text_detected = self._detect_text_presence(image)
        if not text_detected:
            logger.info("РЕЗУЛЬТАТ ШАГ 2: Основная проверка НЕ прошла, пробуем дополнительную")
            # Дополнительная проверка - возможно это логотип или стилизованный текст
            if not self._detect_potential_text(image):
                logger.info("РЕЗУЛЬТАТ ШАГ 2: Дополнительная проверка тоже НЕ прошла - СТОП")
                raise ValueError("На изображении не обнаружен текст для анализа. Попробуйте загрузить изображение с четким, читаемым текстом.")
            else:
                logger.info("РЕЗУЛЬТАТ ШАГ 2: Дополнительная проверка прошла - продолжаем")
        else:
            logger.info("РЕЗУЛЬТАТ ШАГ 2: Основная проверка прошла - продолжаем")
        
        # ШАГ 3: Проверяем на множественные шрифты (только если текст найден)
        logger.info("=== ШАГ 3: Проверка множественных шрифтов ===")
        if self._detect_multiple_fonts(image):
            logger.info("РЕЗУЛЬТАТ ШАГ 3: Обнаружено несколько шрифтов - СТОП")
            raise ValueError("На изображении обнаружено несколько разных шрифтов. Для точного анализа загрузите изображение с текстом одного шрифта.")
        logger.info("РЕЗУЛЬТАТ ШАГ 3: Один шрифт - продолжаем к анализу")
        
        # ШАГ 4: ТОЛЬКО ТЕПЕРЬ делаем анализ шрифта (самая тяжелая операция)
        logger.info("=== ШАГ 4: Анализ характеристик шрифта ===")
        # Предварительная обработка
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = self._binarize_image(gray)
        
        # Извлекаем характеристики
        characteristics = self._extract_characteristics(image, gray, binary)
        
        logger.info("Анализ завершен успешно")
        
        return characteristics
    
    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """Загрузка изображения из байтов"""
        try:
            # Используем PIL для загрузки
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Конвертируем в RGB если нужно
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Конвертируем в OpenCV формат
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            logger.info(f"Изображение загружено: {cv_image.shape}")
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
    
    def _detect_multiple_fonts(self, image: np.ndarray) -> bool:
        """СБАЛАНСИРОВАННОЕ определение множественных шрифтов"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            logger.info("=== СБАЛАНСИРОВАННЫЙ АНАЛИЗ МНОЖЕСТВЕННЫХ ШРИФТОВ ===")
            
            # ПОДХОД 1: Анализ по областям (простой и быстрый)
            strip_height = height // 3
            if strip_height > 10:
                strips = [
                    gray[0:strip_height, :],                    # Верх
                    gray[strip_height:2*strip_height, :],       # Середина
                    gray[2*strip_height:, :]                    # Низ
                ]
                
                # Анализируем вариации
                strip_variations = []
                for i, strip in enumerate(strips):
                    if strip.size > 0:
                        variation = np.std(strip.astype(np.float64))
                        strip_variations.append(variation)
                        logger.info(f"Полоса {i+1}: вариация = {variation:.1f}")
                
                if len(strip_variations) >= 2:
                    max_var = max(strip_variations)
                    min_var = min([v for v in strip_variations if v > 0])
                    
                    if min_var > 0:
                        var_ratio = max_var / min_var
                        logger.info(f"Соотношение вариаций: {var_ratio:.2f}")
                        
                        # Более чувствительный порог
                        if var_ratio > 1.8:  # Снижаем с 2.5 до 1.8
                            logger.info("ПОДХОД 1: Обнаружены различия по областям")
                            return True
            
            # ПОДХОД 2: Анализ контуров (более точный)
            # Применяем адаптивную бинаризацию
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Находим контуры
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Собираем размеры контуров
            contour_sizes = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Минимальный размер
                    x, y, w, h = cv2.boundingRect(contour)
                    # Анализируем размер и пропорции
                    size_metric = max(w, h)  # Берем большую сторону как показатель размера
                    contour_sizes.append(size_metric)
            
            if len(contour_sizes) >= 3:  # Нужно минимум 3 контура для анализа
                contour_sizes.sort(reverse=True)  # Сортируем по убыванию
                
                # Берем самые крупные контуры
                large_sizes = contour_sizes[:max(3, len(contour_sizes)//2)]
                
                # Анализируем разброс размеров
                if len(large_sizes) >= 2:
                    max_size = max(large_sizes)
                    min_size = min(large_sizes)
                    
                    if min_size > 0:
                        size_ratio = max_size / min_size
                        logger.info(f"Анализ контуров: найдено {len(contour_sizes)} контуров")
                        logger.info(f"Соотношение размеров: {size_ratio:.2f} (макс={max_size}, мин={min_size})")
                        
                        # Если есть контуры очень разных размеров - возможно разные шрифты
                        if size_ratio > 2.0:  # Умеренный порог
                            logger.info("ПОДХОД 2: Обнаружены контуры разных размеров")
                            return True
            
            logger.info("Множественные шрифты не обнаружены")
            return False
            
        except Exception as e:
            logger.error(f"Ошибка определения множественных шрифтов: {str(e)}")
            return False
    
    def _analyze_region_characteristics(self, region: np.ndarray, width: int, height: int) -> dict:
        """Анализ характеристик отдельного региона текста"""
        try:
            # Анализ толщины штрихов (упрощенный метод без ximgproc)
            stroke_width = np.mean(region == 0) * 20  # Процент черных пикселей * 20
            
            # Анализ плотности
            density = np.sum(region == 0) / (region.shape[0] * region.shape[1])
            
            # Анализ пропорций
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            aspect_ratios = []
            
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    if h > 0:
                        aspect_ratios.append(w / h)
            
            avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 1.0
            
            # Добавляем размер как характеристику (размер шрифта)
            font_size = max(width, height)  # Приблизительный размер
            
            return {
                'stroke_width': stroke_width,
                'density': density,
                'avg_aspect_ratio': avg_aspect_ratio,
                'font_size': font_size
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа региона: {str(e)}")
            return {
                'stroke_width': 5.0,
                'density': 0.2,
                'avg_aspect_ratio': 1.0,
                'font_size': 50
            }
    
    def _binarize_image(self, gray: np.ndarray) -> np.ndarray:
        """Бинаризация изображения"""
        # Используем адаптивную бинаризацию для лучшего результата
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    def _extract_characteristics(self, image: np.ndarray, gray: np.ndarray, binary: np.ndarray) -> FontCharacteristics:
        """Извлечение характеристик шрифта"""
        
        # Основные характеристики
        has_serifs = self._detect_serifs(binary)
        stroke_width = self._analyze_stroke_width(binary)
        contrast = self._analyze_contrast(gray)
        slant = self._analyze_slant(binary)
        
        # Геометрические характеристики
        x_height, cap_height, ascender, descender = self._analyze_geometry(binary)
        
        # Интервалы
        letter_spacing, word_spacing = self._analyze_spacing(binary)
        density = self._calculate_density(binary)
        
        # Кириллические особенности (упрощенная версия)
        cyrillic_features = self._analyze_cyrillic_features(binary)
        
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


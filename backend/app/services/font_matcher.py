"""
Сервис сопоставления шрифтов
"""

import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models.font_models import FontCharacteristics, FontMatch, FontInfo, FontCategory
from ..database.font_database import FontDatabase

logger = logging.getLogger(__name__)


class FontMatcher:
    """Сопоставление шрифтов на основе характеристик"""
    
    def __init__(self):
        self.font_database = FontDatabase()
        
        # УЛУЧШЕННЫЕ веса для различных характеристик
        self.weights = {
            'serifs': 0.25,        # Важность засечек (повышена)
            'stroke_width': 0.20,   # Важность толщины штрихов (повышена)
            'contrast': 0.15,       # Важность контраста
            'cyrillic': 0.30,       # Важность кириллических особенностей (повышена)
            'geometric': 0.10,      # Важность геометрических характеристик (понижена)
            'spacing': 0.10         # Интервальные характеристики (добавлено)
        }
        
        # Дополнительные параметры для улучшенного сопоставления
        self.similarity_thresholds = {
            'excellent': 0.85,      # Отличное совпадение
            'good': 0.70,           # Хорошее совпадение
            'acceptable': 0.50,     # Приемлемое совпадение
            'poor': 0.30            # Плохое совпадение
        }
        
        # Веса для разных типов шрифтов
        self.font_type_weights = {
            'serif': {
                'serifs': 0.30,     # Засечки критичны для serif
                'stroke_width': 0.25,
                'contrast': 0.20,
                'cyrillic': 0.25
            },
            'sans-serif': {
                'serifs': 0.15,     # Засечки не важны для sans-serif
                'stroke_width': 0.30,
                'contrast': 0.20,
                'cyrillic': 0.35
            },
            'display': {
                'serifs': 0.20,
                'stroke_width': 0.35,  # Толщина важна для display
                'contrast': 0.25,
                'cyrillic': 0.20
            }
        }
    
    async def find_matches(self, characteristics: FontCharacteristics, max_results: int = 10) -> List[FontMatch]:
        """
        Улучшенный поиск наиболее похожих шрифтов с динамическим доступом ко всем Google Fonts
        """
        try:
            logger.info("🔍 Начинаем УЛУЧШЕННЫЙ поиск совпадений...")
            
            # Получаем локальные шрифты (быстро)
            local_fonts = self.font_database.fonts.copy()
            
            # Получаем ВСЕ Google Fonts динамически (медленнее, но полный охват)
            google_fonts = await self.font_database.google_fonts_service.get_all_fonts_for_matching()
            
            # Объединяем все источники
            all_fonts = local_fonts + google_fonts
            
            if not all_fonts:
                logger.warning("База данных шрифтов пуста")
                return []
            
            logger.info(f"📊 Анализируем {len(all_fonts)} шрифтов ({len(local_fonts)} локальных + {len(google_fonts)} Google Fonts)")
            
            # ПРЕДВАРИТЕЛЬНАЯ ФИЛЬТРАЦИЯ для оптимизации
            prefiltered_fonts = self._prefilter_fonts(all_fonts, characteristics)
            logger.info(f"📊 После предфильтрации: {len(prefiltered_fonts)} шрифтов")
            
            # Вычисляем совпадения для отфильтрованных шрифтов
            matches = []
            processed = 0
            
            for font in prefiltered_fonts:
                try:
                    # УЛУЧШЕННОЕ вычисление совпадения
                    confidence = self._calculate_enhanced_match(characteristics, font.characteristics, font.category)
                    match_details = self._calculate_detailed_match(characteristics, font.characteristics)
                    
                    # Дополнительная проверка качества совпадения
                    if confidence >= self.similarity_thresholds['acceptable']:
                        matches.append(FontMatch(
                            font_info=font,
                            confidence=confidence,
                            match_details=match_details
                        ))
                    
                    processed += 1
                    
                    # Логируем прогресс каждые 200 шрифтов
                    if processed % 200 == 0:
                        logger.info(f"⏳ Обработано {processed}/{len(prefiltered_fonts)} шрифтов...")
                    
                except Exception as e:
                    logger.error(f"Ошибка при сопоставлении шрифта {font.name}: {str(e)}")
                    continue
            
            # Сортируем по убыванию уверенности
            matches.sort(key=lambda x: x.confidence, reverse=True)
            
            # Возвращаем топ результатов
            result = matches[:max_results]
            
            logger.info(f"✅ Найдено {len(result)} лучших совпадений из {len(all_fonts)} шрифтов")
            
            # Логируем топ-3 результата для отладки
            for i, match in enumerate(result[:3], 1):
                logger.info(f"  {i}. {match.font_info.name} - {match.confidence:.1%} ({match.font_info.category.value})")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при поиске совпадений: {str(e)}")
            return []
    
    def _prefilter_fonts(self, fonts: List[FontInfo], characteristics: FontCharacteristics) -> List[FontInfo]:
        """Предварительная фильтрация шрифтов для оптимизации"""
        try:
            prefiltered = []
            
            for font in fonts:
                # 1. Фильтр по засечкам (критично)
                if font.characteristics.has_serifs != characteristics.has_serifs:
                    continue
                
                # 2. Фильтр по поддержке кириллицы
                if not font.cyrillic_support:
                    continue
                
                # 3. Фильтр по категории (если есть предпочтения)
                # Пока пропускаем все категории
                
                prefiltered.append(font)
            
            logger.info(f"📊 Предфильтрация: {len(fonts)} → {len(prefiltered)} шрифтов")
            return prefiltered
            
        except Exception as e:
            logger.error(f"Ошибка предфильтрации: {str(e)}")
            return fonts  # Возвращаем все шрифты при ошибке
    
    def _calculate_enhanced_match(self, analyzed: FontCharacteristics, reference: FontCharacteristics, 
                                 font_category: FontCategory) -> float:
        """
        УЛУЧШЕННОЕ вычисление коэффициента сходства с учетом категории шрифта
        """
        try:
            # Выбираем веса в зависимости от категории шрифта
            category_weights = self.font_type_weights.get(
                font_category.value, 
                self.weights  # Используем базовые веса если категория не найдена
            )
            
            # Сравнение засечек (бинарная характеристика)
            serif_match = 1.0 if analyzed.has_serifs == reference.has_serifs else 0.0
            
            # УЛУЧШЕННОЕ сравнение толщины штрихов
            stroke_match = self._compare_enhanced_numeric(
                analyzed.stroke_width, reference.stroke_width, 
                tolerance=0.3  # Более мягкая толерантность
            )
            
            # УЛУЧШЕННОЕ сравнение контраста
            contrast_match = self._compare_enhanced_numeric(
                analyzed.contrast, reference.contrast, 
                tolerance=0.4  # Контраст может сильно варьироваться
            )
            
            # УЛУЧШЕННОЕ сравнение кириллических особенностей
            cyrillic_match = self._compare_enhanced_cyrillic_features(
                analyzed.cyrillic_features, 
                reference.cyrillic_features
            )
            
            # Вычисляем взвешенную сумму с учетом категории
            total_score = (
                serif_match * category_weights['serifs'] +
                stroke_match * category_weights['stroke_width'] +
                contrast_match * category_weights['contrast'] +
                cyrillic_match * category_weights['cyrillic']
            )
            
            # ДОПОЛНИТЕЛЬНЫЕ БОНУСЫ
            bonus = 0.0
            
            # Бонус за точное совпадение засечек
            if serif_match == 1.0:
                bonus += 0.05
            
            # Бонус за высокое качество кириллической поддержки
            if cyrillic_match > 0.8:
                bonus += 0.03
            
            # Бонус за сбалансированность характеристик
            if (stroke_match > 0.7 and contrast_match > 0.6):
                bonus += 0.02
            
            # Применяем бонусы
            total_score = min(1.0, total_score + bonus)
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении улучшенного совпадения: {str(e)}")
            return 0.0
    
    def _compare_enhanced_numeric(self, value1: float, value2: float, tolerance: float = 0.2) -> float:
        """
        УЛУЧШЕННОЕ сравнение числовых характеристик с настраиваемой толерантностью
        """
        try:
            if max(value1, value2) == 0:
                return 1.0 if value1 == value2 else 0.0
            
            # Вычисляем относительную разность
            diff = abs(value1 - value2)
            max_val = max(value1, value2)
            relative_diff = diff / max_val
            
            # Если разность меньше толерантности - отличное совпадение
            if relative_diff <= tolerance:
                return 1.0
            
            # Если разность больше толерантности - плавное снижение
            # Используем экспоненциальное затухание для более естественного снижения
            decay_factor = 2.0  # Скорость затухания
            similarity = np.exp(-decay_factor * (relative_diff - tolerance))
            
            return max(0.0, similarity)
            
        except Exception as e:
            logger.error(f"Ошибка при улучшенном числовом сравнении: {str(e)}")
            return 0.0
    
    def _compare_enhanced_cyrillic_features(self, analyzed, reference) -> float:
        """
        УЛУЧШЕННОЕ сравнение кириллических особенностей
        """
        try:
            features = ['ya_shape', 'zh_shape', 'fi_shape', 'shcha_shape', 'yery_shape']
            
            total_match = 0.0
            feature_weights = [0.25, 0.20, 0.20, 0.20, 0.15]  # Разные веса для разных букв
            
            for i, feature in enumerate(features):
                try:
                    analyzed_val = getattr(analyzed, feature, 0.5)
                    reference_val = getattr(reference, feature, 0.5)
                    
                    # Используем улучшенное числовое сравнение для каждой особенности
                    match = self._compare_enhanced_numeric(analyzed_val, reference_val, tolerance=0.3)
                    
                    # Применяем вес особенности
                    total_match += match * feature_weights[i]
                    
                except AttributeError:
                    # Если особенность не найдена, используем среднее значение
                    total_match += 0.5 * feature_weights[i]
            
            return total_match
            
        except Exception as e:
            logger.error(f"Ошибка при улучшенном сравнении кириллических характеристик: {str(e)}")
            return 0.0
    
    def _calculate_match(self, analyzed: FontCharacteristics, reference: FontCharacteristics) -> float:
        """
        Вычисление общего коэффициента сходства
        """
        try:
            # Сравнение засечек (бинарная характеристика)
            serif_match = 1.0 if analyzed.has_serifs == reference.has_serifs else 0.0
            
            # Сравнение толщины штрихов
            stroke_match = self._compare_numeric(analyzed.stroke_width, reference.stroke_width)
            
            # Сравнение контраста
            contrast_match = self._compare_numeric(analyzed.contrast, reference.contrast)
            
            # Сравнение кириллических особенностей
            cyrillic_match = self._compare_cyrillic_features(
                analyzed.cyrillic_features, 
                reference.cyrillic_features
            )
            
            # Сравнение геометрических характеристик
            geometric_match = self._compare_geometric_features(analyzed, reference)
            
            # Сравнение интервалов
            spacing_match = self._compare_spacing_features(analyzed, reference)
            
            # Вычисляем взвешенную сумму
            total_score = (
                serif_match * self.weights['serifs'] +
                stroke_match * self.weights['stroke_width'] +
                contrast_match * self.weights['contrast'] +
                cyrillic_match * self.weights['cyrillic'] +
                geometric_match * self.weights['geometric'] +
                spacing_match * self.weights['spacing']
            )
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении совпадения: {str(e)}")
            return 0.0
    
    def _calculate_detailed_match(self, analyzed: FontCharacteristics, reference: FontCharacteristics) -> Dict[str, float]:
        """
        Вычисление детальных метрик сходства
        """
        try:
            serif_match = 1.0 if analyzed.has_serifs == reference.has_serifs else 0.0
            stroke_match = self._compare_numeric(analyzed.stroke_width, reference.stroke_width)
            contrast_match = self._compare_numeric(analyzed.contrast, reference.contrast)
            cyrillic_match = self._compare_cyrillic_features(
                analyzed.cyrillic_features, 
                reference.cyrillic_features
            )
            geometric_match = self._compare_geometric_features(analyzed, reference)
            spacing_match = self._compare_spacing_features(analyzed, reference)
            
            overall_score = (
                serif_match * self.weights['serifs'] +
                stroke_match * self.weights['stroke_width'] +
                contrast_match * self.weights['contrast'] +
                cyrillic_match * self.weights['cyrillic'] +
                geometric_match * self.weights['geometric'] +
                spacing_match * self.weights['spacing']
            )
            
            return {
                'overall_score': overall_score,
                'serif_match': serif_match,
                'stroke_match': stroke_match,
                'contrast_match': contrast_match,
                'cyrillic_match': cyrillic_match,
                'geometric_match': geometric_match,
                'spacing_match': spacing_match
            }
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении детального совпадения: {str(e)}")
            return {
                'overall_score': 0.0,
                'serif_match': 0.0,
                'stroke_match': 0.0,
                'contrast_match': 0.0,
                'cyrillic_match': 0.0,
                'geometric_match': 0.0,
                'spacing_match': 0.0
            }
    
    def _compare_numeric(self, value1: float, value2: float) -> float:
        """
        Сравнение числовых характеристик
        """
        if max(value1, value2) == 0:
            return 1.0 if value1 == value2 else 0.0
        
        diff = abs(value1 - value2)
        max_val = max(value1, value2)
        
        return max(0.0, 1.0 - diff / max_val)
    
    def _compare_cyrillic_features(self, analyzed, reference) -> float:
        """
        Сравнение кириллических особенностей
        """
        try:
            features = ['ya_shape', 'zh_shape', 'fi_shape', 'shcha_shape', 'yery_shape']
            
            total_match = 0.0
            for feature in features:
                analyzed_val = getattr(analyzed, feature)
                reference_val = getattr(reference, feature)
                match = self._compare_numeric(analyzed_val, reference_val)
                total_match += match
            
            return total_match / len(features)
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении кириллических характеристик: {str(e)}")
            return 0.0
    
    def _compare_geometric_features(self, analyzed: FontCharacteristics, reference: FontCharacteristics) -> float:
        """
        Сравнение геометрических характеристик
        """
        try:
            features = [
                ('x_height', 0.3),
                ('cap_height', 0.3),
                ('ascender', 0.2),
                ('descender', 0.2)
            ]
            
            weighted_match = 0.0
            total_weight = 0.0
            
            for feature_name, weight in features:
                analyzed_val = getattr(analyzed, feature_name)
                reference_val = getattr(reference, feature_name)
                
                match = self._compare_numeric(analyzed_val, reference_val)
                weighted_match += match * weight
                total_weight += weight
            
            return weighted_match / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении геометрических характеристик: {str(e)}")
            return 0.0
    
    def _compare_spacing_features(self, analyzed: FontCharacteristics, reference: FontCharacteristics) -> float:
        """
        Сравнение характеристик интервалов
        """
        try:
            features = [
                ('letter_spacing', 0.4),
                ('word_spacing', 0.3),
                ('density', 0.3)
            ]
            
            weighted_match = 0.0
            total_weight = 0.0
            
            for feature_name, weight in features:
                analyzed_val = getattr(analyzed, feature_name)
                reference_val = getattr(reference, feature_name)
                
                match = self._compare_numeric(analyzed_val, reference_val)
                weighted_match += match * weight
                total_weight += weight
            
            return weighted_match / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении характеристик интервалов: {str(e)}")
            return 0.0
    
    def filter_by_confidence(self, matches: List[FontMatch], min_confidence: float) -> List[FontMatch]:
        """
        Фильтрация результатов по минимальной уверенности
        """
        return [match for match in matches if match.confidence >= min_confidence]
    
    def group_by_category(self, matches: List[FontMatch]) -> Dict[str, List[FontMatch]]:
        """
        Группировка результатов по категориям
        """
        grouped = {}
        
        for match in matches:
            category = match.font_info.category.value
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(match)
        
        return grouped


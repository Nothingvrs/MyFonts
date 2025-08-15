"""
Сервис сопоставления шрифтов
"""

import logging
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..models.font_models import FontCharacteristics, FontMatch, FontInfo
from ..database.font_database import FontDatabase

logger = logging.getLogger(__name__)


class FontMatcher:
    """Сопоставление шрифтов на основе характеристик"""
    
    def __init__(self):
        self.font_database = FontDatabase()
        
        # Веса для различных характеристик
        self.weights = {
            'serifs': 0.2,        # Важность засечек
            'stroke_width': 0.15,  # Важность толщины штрихов
            'contrast': 0.15,      # Важность контраста
            'cyrillic': 0.25,      # Важность кириллических особенностей
            'geometric': 0.15,     # Важность геометрических характеристик
            'spacing': 0.1         # Важность интервалов
        }
    
    async def find_matches(self, characteristics: FontCharacteristics, max_results: int = 10) -> List[FontMatch]:
        """
        Поиск наиболее похожих шрифтов с динамическим доступом ко всем Google Fonts
        """
        try:
            logger.info("🔍 Начинаем поиск совпадений с полной базой...")
            
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
            
            # Вычисляем совпадения для каждого шрифта
            matches = []
            processed = 0
            
            for font in all_fonts:
                try:
                    confidence = self._calculate_match(characteristics, font.characteristics)
                    match_details = self._calculate_detailed_match(characteristics, font.characteristics)
                    
                    matches.append(FontMatch(
                        font_info=font,
                        confidence=confidence,
                        match_details=match_details
                    ))
                    
                    processed += 1
                    
                    # Логируем прогресс каждые 500 шрифтов
                    if processed % 500 == 0:
                        logger.info(f"⏳ Обработано {processed}/{len(all_fonts)} шрифтов...")
                    
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
                logger.info(f"  {i}. {match.font_info.name} - {match.confidence:.1%}")
            
            # ОТЛАДКА: Логируем анализируемые характеристики
            logger.info(f"🔍 АНАЛИЗИРУЕМЫЕ ХАРАКТЕРИСТИКИ:")
            logger.info(f"  - Засечки: {characteristics.has_serifs}")
            logger.info(f"  - Толщина: {characteristics.stroke_width:.3f}")
            logger.info(f"  - Контраст: {characteristics.contrast:.3f}")
            logger.info(f"  - Наклон: {characteristics.slant:.3f}")
            
            # ОТЛАДКА: Логируем первые 3 шрифта из базы
            logger.info(f"🔍 ПЕРВЫЕ 3 ШРИФТА ИЗ БАЗЫ:")
            for i, font in enumerate(all_fonts[:3], 1):
                logger.info(f"  {i}. {font.name}:")
                logger.info(f"     - Засечки: {font.characteristics.has_serifs}")
                logger.info(f"     - Толщина: {font.characteristics.stroke_width:.3f}")
                logger.info(f"     - Контраст: {font.characteristics.contrast:.3f}")
                logger.info(f"     - Наклон: {font.characteristics.slant:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при поиске совпадений: {str(e)}")
            return []
    
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


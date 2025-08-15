"""
Сервис для интеграции с Google Fonts API
"""

import logging
import asyncio
import aiohttp
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from ..models.font_models import FontInfo, FontCategory, FontCharacteristics, CyrillicFeatures

logger = logging.getLogger(__name__)


class GoogleFontsService:
    """Сервис для работы с Google Fonts API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/webfonts/v1/webfonts"
        self.cache_file = Path("google_fonts_cache.json")
        self.cache_duration = timedelta(hours=24)  # Кэш на 24 часа
        self._fonts_cache: List[Dict] = []
        self._last_update: Optional[datetime] = None
        self._all_fonts_cache: List[FontInfo] = []  # Кэш для всех конвертированных шрифтов
        self._all_fonts_cache_time: Optional[datetime] = None
    
    async def get_fonts(self, force_refresh: bool = False) -> List[Dict]:
        """Получение списка шрифтов из Google Fonts"""
        try:
            # Проверяем кэш
            if not force_refresh and self._is_cache_valid():
                logger.info("Используем кэшированные Google Fonts")
                return self._fonts_cache
            
            # Загружаем из файла кэша
            if not force_refresh and self._load_from_file_cache():
                logger.info("Загружены Google Fonts из файлового кэша")
                return self._fonts_cache
            
            # Загружаем с API
            logger.info("Загружаем Google Fonts с API...")
            fonts = await self._fetch_from_api()
            
            if fonts:
                self._fonts_cache = fonts
                self._last_update = datetime.now()
                self._save_to_file_cache()
                logger.info(f"Загружено {len(fonts)} шрифтов из Google Fonts API")
            
            return fonts
            
        except Exception as e:
            logger.error(f"Ошибка получения Google Fonts: {str(e)}")
            # Возвращаем кэш если есть
            return self._fonts_cache if self._fonts_cache else []
    
    async def _fetch_from_api(self) -> List[Dict]:
        """Загрузка с Google Fonts API"""
        try:
            # Google Fonts API работает без ключа с ограничениями
            params = {"sort": "popularity"}
            if self.api_key:
                params["key"] = self.api_key
            
            headers = {
                'User-Agent': 'MyFonts/1.0 (Font Identification Service)'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                logger.info(f"Запрос к Google Fonts API: {self.base_url}")
                async with session.get(self.base_url, params=params) as response:
                    logger.info(f"Статус ответа Google Fonts API: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        items = data.get("items", [])
                        logger.info(f"Получено {len(items)} шрифтов из Google Fonts API")
                        return items
                    else:
                        error_text = await response.text()
                        logger.error(f"Google Fonts API вернул статус {response.status}: {error_text}")
                        return []
                        
        except asyncio.TimeoutError:
            logger.error("Таймаут при запросе к Google Fonts API")
            return []
        except Exception as e:
            logger.error(f"Ошибка запроса к Google Fonts API: {str(e)}")
            return []
    
    def _is_cache_valid(self) -> bool:
        """Проверка валидности кэша"""
        if not self._fonts_cache or not self._last_update:
            return False
        
        return datetime.now() - self._last_update < self.cache_duration
    
    def _load_from_file_cache(self) -> bool:
        """Загрузка из файлового кэша"""
        try:
            if not self.cache_file.exists():
                return False
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Проверяем время кэша
            cache_time = datetime.fromisoformat(cache_data.get("timestamp", ""))
            if datetime.now() - cache_time > self.cache_duration:
                return False
            
            self._fonts_cache = cache_data.get("fonts", [])
            self._last_update = cache_time
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки файлового кэша: {str(e)}")
            return False
    
    def _save_to_file_cache(self):
        """Сохранение в файловый кэш"""
        try:
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "fonts": self._fonts_cache
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка сохранения файлового кэша: {str(e)}")
    
    def convert_to_font_info(self, google_font: Dict, font_id: int) -> FontInfo:
        """Конвертация Google Font в FontInfo"""
        try:
            # Определяем категорию
            category_map = {
                "serif": FontCategory.SERIF,
                "sans-serif": FontCategory.SANS_SERIF,
                "monospace": FontCategory.MONOSPACE,
                "display": FontCategory.DISPLAY,
                "handwriting": FontCategory.SCRIPT
            }
            
            category = category_map.get(
                google_font.get("category", "sans-serif"),
                FontCategory.SANS_SERIF
            )
            
            # Проверяем поддержку кириллицы
            subsets = google_font.get("subsets", [])
            cyrillic_support = "cyrillic" in subsets or "cyrillic-ext" in subsets
            
            # Генерируем характеристики на основе категории и названия
            characteristics = self._generate_characteristics(google_font, category, cyrillic_support)
            
            # Определяем популярность (Google Fonts отсортированы по популярности)
            # Первые 50 - очень популярные, следующие 200 - популярные, остальные - средние/низкие
            popularity = 0.9 if font_id <= 50 else (0.7 if font_id <= 250 else 0.4)
            
            return FontInfo(
                id=font_id,
                name=google_font["family"],
                category=category,
                characteristics=characteristics,
                popularity=popularity,
                cyrillic_support=cyrillic_support,
                designer=None,  # Google Fonts API не предоставляет эту информацию
                year=None,
                foundry="Google Fonts",
                description=f"Шрифт из коллекции Google Fonts, категория: {category.value}",
                license="OFL"  # Большинство Google Fonts под OFL лицензией
            )
            
        except Exception as e:
            logger.error(f"Ошибка конвертации Google Font {google_font.get('family', 'Unknown')}: {str(e)}")
            return None
    
    def _generate_characteristics(self, google_font: Dict, category: FontCategory, cyrillic_support: bool) -> FontCharacteristics:
        """Генерация характеристик шрифта на основе доступной информации"""
        
        # Базовые характеристики по категории
        base_chars = {
            FontCategory.SERIF: {
                "has_serifs": True,
                "stroke_width": 0.14,
                "contrast": 0.7,
                "slant": 0.0,
                "x_height": 48.0,
                "cap_height": 68.0,
                "ascender": 82.0,
                "descender": 22.0,
                "letter_spacing": 2.2,
                "word_spacing": 6.5,
                "density": 0.38
            },
            FontCategory.SANS_SERIF: {
                "has_serifs": False,
                "stroke_width": 0.16,
                "contrast": 0.25,
                "slant": 0.0,
                "x_height": 53.0,
                "cap_height": 70.0,
                "ascender": 80.0,
                "descender": 20.0,
                "letter_spacing": 1.6,
                "word_spacing": 5.2,
                "density": 0.43
            },
            FontCategory.MONOSPACE: {
                "has_serifs": False,
                "stroke_width": 0.16,
                "contrast": 0.2,
                "slant": 0.0,
                "x_height": 52.0,
                "cap_height": 70.0,
                "ascender": 80.0,
                "descender": 20.0,
                "letter_spacing": 0.0,
                "word_spacing": 10.0,
                "density": 0.4
            },
            FontCategory.DISPLAY: {
                "has_serifs": False,
                "stroke_width": 0.22,
                "contrast": 0.1,
                "slant": 0.0,
                "x_height": 60.0,
                "cap_height": 70.0,
                "ascender": 75.0,
                "descender": 15.0,
                "letter_spacing": 2.0,
                "word_spacing": 6.0,
                "density": 0.5
            },
            FontCategory.SCRIPT: {
                "has_serifs": False,
                "stroke_width": 0.2,
                "contrast": 0.5,
                "slant": 15.0,
                "x_height": 45.0,
                "cap_height": 65.0,
                "ascender": 80.0,
                "descender": 25.0,
                "letter_spacing": 1.0,
                "word_spacing": 4.0,
                "density": 0.35
            }
        }
        
        chars = base_chars.get(category, base_chars[FontCategory.SANS_SERIF]).copy()
        
        # Добавляем вариацию на основе названия шрифта
        font_name = google_font.get("family", "")
        name_hash = hash(font_name) % 1000  # Получаем стабильный хеш от 0 до 999
        
        # Добавляем небольшую вариацию к каждой характеристике
        variation = (name_hash / 1000.0 - 0.5) * 0.3  # От -0.15 до +0.15
        
        # Применяем вариацию к числовым характеристикам
        chars["stroke_width"] = max(0.05, min(1.0, chars["stroke_width"] + variation * 0.2))
        chars["contrast"] = max(0.0, min(1.0, chars["contrast"] + variation * 0.4))
        chars["slant"] = max(-45.0, min(45.0, chars["slant"] + variation * 10))
        chars["x_height"] = max(20.0, chars["x_height"] + variation * 15)
        chars["cap_height"] = max(30.0, chars["cap_height"] + variation * 10)
        chars["ascender"] = max(40.0, chars["ascender"] + variation * 8)
        chars["descender"] = max(5.0, chars["descender"] + variation * 5)
        chars["letter_spacing"] = max(0.0, chars["letter_spacing"] + variation * 1.0)
        chars["word_spacing"] = max(2.0, chars["word_spacing"] + variation * 2.0)
        chars["density"] = max(0.1, min(1.0, chars["density"] + variation * 0.2))
        
        # Генерируем кириллические характеристики с вариацией
        if cyrillic_support:
            # Хорошая поддержка кириллицы + вариация
            cyrillic_features = CyrillicFeatures(
                ya_shape=max(0.0, min(1.0, 0.85 + variation * 0.2)),
                zh_shape=max(0.0, min(1.0, 0.85 + variation * 0.2)),
                fi_shape=max(0.0, min(1.0, 0.85 + variation * 0.2)),
                shcha_shape=max(0.0, min(1.0, 0.9 + variation * 0.15)),
                yery_shape=max(0.0, min(1.0, 0.85 + variation * 0.2))
            )
        else:
            # Слабая или отсутствующая поддержка + вариация
            cyrillic_features = CyrillicFeatures(
                ya_shape=max(0.0, min(1.0, 0.5 + variation * 0.3)),
                zh_shape=max(0.0, min(1.0, 0.5 + variation * 0.3)),
                fi_shape=max(0.0, min(1.0, 0.5 + variation * 0.3)),
                shcha_shape=max(0.0, min(1.0, 0.5 + variation * 0.3)),
                yery_shape=max(0.0, min(1.0, 0.5 + variation * 0.3))
            )
        
        return FontCharacteristics(
            cyrillic_features=cyrillic_features,
            **chars
        )
    
    async def search_fonts(self, query: str, limit: int = 50) -> List[FontInfo]:
        """Поиск шрифтов по названию во всей базе Google Fonts"""
        try:
            google_fonts = await self.get_fonts()
            query_lower = query.lower()
            
            results = []
            font_id = 10000  # Начинаем с большого ID чтобы не пересекаться с локальными
            
            for font_data in google_fonts:
                if len(results) >= limit:
                    break
                
                font_name = font_data.get("family", "").lower()
                if query_lower in font_name:
                    font_info = self.convert_to_font_info(font_data, font_id)
                    if font_info:
                        results.append(font_info)
                        font_id += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка поиска Google Fonts: {str(e)}")
            return []
    
    async def get_all_fonts_for_matching(self) -> List[FontInfo]:
        """Получение ВСЕХ Google Fonts для сопоставления шрифтов (с кэшированием)"""
        try:
            # Проверяем кэш конвертированных шрифтов
            if (self._all_fonts_cache and 
                self._all_fonts_cache_time and 
                datetime.now() - self._all_fonts_cache_time < self.cache_duration):
                logger.info(f"📋 Используем кэшированные {len(self._all_fonts_cache)} конвертированных Google Fonts")
                return self._all_fonts_cache
            
            logger.info("🔍 Загружаем и конвертируем ВСЕ Google Fonts для сопоставления...")
            google_fonts = await self.get_fonts()
            
            results = []
            font_id = 10000
            
            # Конвертируем ТОЛЬКО кириллические шрифты (оптимизация для нашего сервиса)
            total = len(google_fonts)
            cyrillic_count = 0
            
            for i, font_data in enumerate(google_fonts):
                # Проверяем поддержку кириллицы
                subsets = font_data.get('subsets', [])
                has_cyrillic = any(subset in ['cyrillic', 'cyrillic-ext'] for subset in subsets)
                
                if has_cyrillic:  # ТОЛЬКО кириллические шрифты!
                    font_info = self.convert_to_font_info(font_data, font_id)
                    if font_info:
                        results.append(font_info)
                        font_id += 1
                        cyrillic_count += 1
                
                # Логируем прогресс каждые 500 шрифтов
                if (i + 1) % 500 == 0:
                    logger.info(f"🇷🇺 Обработано {i + 1}/{total} шрифтов, найдено {cyrillic_count} кириллических")
            
            # Кэшируем результат
            self._all_fonts_cache = results
            self._all_fonts_cache_time = datetime.now()
            
            logger.info(f"✅ Конвертировано и закэшировано {len(results)} Google Fonts для сопоставления")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка получения всех Google Fonts: {str(e)}")
            return self._all_fonts_cache if self._all_fonts_cache else []
    
    async def get_popular_fonts(self, limit: int = 100) -> List[FontInfo]:
        """Получение популярных шрифтов"""
        try:
            google_fonts = await self.get_fonts()
            
            results = []
            font_id = 10000
            
            # Google Fonts уже отсортированы по популярности
            for font_data in google_fonts[:limit]:
                font_info = self.convert_to_font_info(font_data, font_id)
                if font_info:
                    results.append(font_info)
                    font_id += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка получения популярных Google Fonts: {str(e)}")
            return []

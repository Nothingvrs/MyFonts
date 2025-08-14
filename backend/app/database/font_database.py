"""
База данных шрифтов
"""

import logging
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..models.font_models import FontInfo, FontCategory, FontCharacteristics, CyrillicFeatures
from ..services.google_fonts_service import GoogleFontsService

logger = logging.getLogger(__name__)


class FontDatabase:
    """Управление базой данных шрифтов"""
    
    def __init__(self):
        self.fonts: List[FontInfo] = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.google_fonts_service = GoogleFontsService(api_key="AIzaSyBGG0iqkjWIr8SlH8au0vQbmfojz7wtrKs")
        self._google_fonts_cache: List[FontInfo] = []
        self._initialize_fonts()
    
    async def initialize(self):
        """Асинхронная инициализация базы данных"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._initialize_fonts)
        
        # Загружаем популярные Google Fonts с API ключом
        try:
            logger.info("🚀 Загружаем Google Fonts с API ключом...")
            google_fonts = await self.google_fonts_service.get_popular_fonts(limit=200)
            
            if google_fonts:
                self._google_fonts_cache = google_fonts
                logger.info(f"✅ Загружено {len(google_fonts)} Google Fonts")
            else:
                logger.warning("⚠️ Google Fonts API не вернул шрифты - используем локальные")
                self._google_fonts_cache = []
                self._add_popular_fonts_to_local_database()
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Google Fonts: {str(e)}")
            logger.info("🔄 Используем расширенную локальную базу")
            self._google_fonts_cache = []
            self._add_popular_fonts_to_local_database()
        
        total_fonts = len(self.fonts) + len(self._google_fonts_cache)
        logger.info(f"📚 База данных инициализирована: {len(self.fonts)} локальных + {len(self._google_fonts_cache)} Google Fonts = {total_fonts} всего")
    
    def _initialize_fonts(self):
        """Инициализация базы данных шрифтов"""
        self.fonts = [
            FontInfo(
                id=1,
                name="Times New Roman",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.15,
                    contrast=0.7,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.7,
                        fi_shape=0.9,
                        shcha_shape=0.8,
                        yery_shape=0.7
                    ),
                    x_height=50.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=2.0,
                    word_spacing=6.0,
                    density=0.4
                ),
                popularity=0.9,
                cyrillic_support=True,
                designer="Stanley Morison",
                year=1932,
                foundry="Monotype",
                description="Классический переходный антиквенный шрифт",
                license="Commercial"
            ),
            
            FontInfo(
                id=2,
                name="Arial",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.18,
                    contrast=0.3,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.6,
                        zh_shape=0.8,
                        fi_shape=0.7,
                        shcha_shape=0.9,
                        yery_shape=0.8
                    ),
                    x_height=55.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.5,
                    word_spacing=5.0,
                    density=0.45
                ),
                popularity=0.95,
                cyrillic_support=True,
                designer="Robin Nicholas, Patricia Saunders",
                year=1982,
                foundry="Monotype",
                description="Неогротесковый шрифт без засечек",
                license="Commercial"
            ),
            
            FontInfo(
                id=3,
                name="PT Sans",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.25,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9,
                        zh_shape=0.9,
                        fi_shape=0.8,
                        shcha_shape=0.95,
                        yery_shape=0.9
                    ),
                    x_height=52.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.8,
                    word_spacing=5.5,
                    density=0.42
                ),
                popularity=0.7,
                cyrillic_support=True,
                designer="Alexandra Korolkova, Olga Umpeleva, Vladimir Yefimov",
                year=2009,
                foundry="ParaType",
                description="Российский шрифт семейства PT, оптимизированный для кириллицы",
                license="OFL"
            ),
            
            FontInfo(
                id=4,
                name="PT Serif",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.14,
                    contrast=0.8,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85,
                        zh_shape=0.8,
                        fi_shape=0.9,
                        shcha_shape=0.9,
                        yery_shape=0.8
                    ),
                    x_height=48.0,
                    cap_height=68.0,
                    ascender=82.0,
                    descender=22.0,
                    letter_spacing=2.2,
                    word_spacing=6.5,
                    density=0.38
                ),
                popularity=0.6,
                cyrillic_support=True,
                designer="Alexandra Korolkova, Olga Umpeleva, Vladimir Yefimov",
                year=2009,
                foundry="ParaType",
                description="Переходная антиква семейства PT для кириллицы",
                license="OFL"
            ),
            
            FontInfo(
                id=5,
                name="Georgia",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.17,
                    contrast=0.6,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.7,
                        zh_shape=0.75,
                        fi_shape=0.8,
                        shcha_shape=0.8,
                        yery_shape=0.75
                    ),
                    x_height=53.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=2.0,
                    word_spacing=6.0,
                    density=0.43
                ),
                popularity=0.75,
                cyrillic_support=True,
                designer="Matthew Carter",
                year=1993,
                foundry="Microsoft",
                description="Переходная антиква, оптимизированная для экранов",
                license="Commercial"
            ),
            
            FontInfo(
                id=6,
                name="Open Sans",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.15,
                    contrast=0.2,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.7,
                        zh_shape=0.8,
                        fi_shape=0.75,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=54.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.6,
                    word_spacing=5.2,
                    density=0.44
                ),
                popularity=0.8,
                cyrillic_support=True,
                designer="Steve Matteson",
                year=2011,
                foundry="Ascender Corp",
                description="Гуманистический гротеск с дружелюбным характером",
                license="OFL"
            ),
            
            FontInfo(
                id=7,
                name="Roboto",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.15,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.75,
                        zh_shape=0.85,
                        fi_shape=0.8,
                        shcha_shape=0.9,
                        yery_shape=0.85
                    ),
                    x_height=53.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.4,
                    word_spacing=4.8,
                    density=0.46
                ),
                popularity=0.85,
                cyrillic_support=True,
                designer="Christian Robertson",
                year=2011,
                foundry="Google",
                description="Неогротеск, созданный для Android",
                license="Apache 2.0"
            ),
            
            FontInfo(
                id=8,
                name="Liberation Serif",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.15,
                    contrast=0.65,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.7,
                        fi_shape=0.85,
                        shcha_shape=0.8,
                        yery_shape=0.7
                    ),
                    x_height=50.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=2.1,
                    word_spacing=6.2,
                    density=0.39
                ),
                popularity=0.4,
                cyrillic_support=True,
                designer="Steve Matteson",
                year=2007,
                foundry="Red Hat",
                description="Свободная альтернатива Times New Roman",
                license="OFL"
            ),
            
            FontInfo(
                id=9,
                name="DejaVu Sans",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.17,
                    contrast=0.25,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.65,
                        zh_shape=0.8,
                        fi_shape=0.7,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=54.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.7,
                    word_spacing=5.3,
                    density=0.43
                ),
                popularity=0.5,
                cyrillic_support=True,
                designer="DejaVu Team",
                year=2004,
                foundry="DejaVu",
                description="Расширенная версия Vera Sans с поддержкой Unicode",
                license="Bitstream Vera License"
            ),
            
            FontInfo(
                id=10,
                name="Source Sans Pro",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.14,
                    contrast=0.2,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.7,
                        zh_shape=0.8,
                        fi_shape=0.75,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=52.0,
                    cap_height=69.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.5,
                    word_spacing=5.0,
                    density=0.41
                ),
                popularity=0.6,
                cyrillic_support=True,
                designer="Paul D. Hunt",
                year=2012,
                foundry="Adobe",
                description="Гуманистический гротеск от Adobe",
                license="OFL"
            ),
            
            # === РОССИЙСКИЕ/УКРАИНСКИЕ ШРИФТЫ ===
            
            FontInfo(
                id=11,
                name="Pragmatica",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.2,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.95,
                        zh_shape=0.9,
                        fi_shape=0.9,
                        shcha_shape=0.95,
                        yery_shape=0.9
                    ),
                    x_height=54.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.6,
                    word_spacing=5.2,
                    density=0.44
                ),
                popularity=0.3,
                cyrillic_support=True,
                designer="Vladimir Yefimov",
                year=1989,
                foundry="ParaType",
                description="Классический российский гротеск советского периода",
                license="Commercial"
            ),
            
            FontInfo(
                id=12,
                name="Minion Pro Cyrillic",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.13,
                    contrast=0.75,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9,
                        zh_shape=0.85,
                        fi_shape=0.95,
                        shcha_shape=0.9,
                        yery_shape=0.85
                    ),
                    x_height=47.0,
                    cap_height=68.0,
                    ascender=82.0,
                    descender=22.0,
                    letter_spacing=2.4,
                    word_spacing=6.8,
                    density=0.36
                ),
                popularity=0.25,
                cyrillic_support=True,
                designer="Robert Slimbach",
                year=1990,
                foundry="Adobe",
                description="Элегантная антиква эпохи Возрождения с кириллицей",
                license="Commercial"
            ),
            
            FontInfo(
                id=13,
                name="Officina Serif Cyrillic",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.18,
                    contrast=0.4,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.85,
                        fi_shape=0.8,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=55.0,
                    cap_height=70.0,
                    ascender=78.0,
                    descender=18.0,
                    letter_spacing=1.8,
                    word_spacing=5.5,
                    density=0.48
                ),
                popularity=0.2,
                cyrillic_support=True,
                designer="Erik Spiekermann",
                year=1990,
                foundry="FontFont",
                description="Брусковая антиква для деловой корреспонденции",
                license="Commercial"
            ),
            
            FontInfo(
                id=14,
                name="Myriad Pro Cyrillic",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.15,
                    contrast=0.25,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85,
                        zh_shape=0.9,
                        fi_shape=0.85,
                        shcha_shape=0.9,
                        yery_shape=0.85
                    ),
                    x_height=53.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.5,
                    word_spacing=5.0,
                    density=0.43
                ),
                popularity=0.35,
                cyrillic_support=True,
                designer="Robert Slimbach, Carol Twombly",
                year=1992,
                foundry="Adobe",
                description="Гуманистический гротеск с кириллическим дополнением",
                license="Commercial"
            ),
            
            FontInfo(
                id=15,
                name="Futura PT",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.18,
                    contrast=0.1,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.7,
                        zh_shape=0.8,
                        fi_shape=0.75,
                        shcha_shape=0.8,
                        yery_shape=0.75
                    ),
                    x_height=50.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=2.0,
                    word_spacing=6.0,
                    density=0.4
                ),
                popularity=0.3,
                cyrillic_support=True,
                designer="Paul Renner, Vladimir Yefimov",
                year=1927,
                foundry="ParaType",
                description="Геометрический гротеск с кириллическим дополнением",
                license="Commercial"
            ),
            
            # === ДЕКОРАТИВНЫЕ И DISPLAY ШРИФТЫ ===
            
            FontInfo(
                id=16,
                name="Impact",
                category=FontCategory.DISPLAY,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.25,
                    contrast=0.1,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.6,
                        zh_shape=0.7,
                        fi_shape=0.65,
                        shcha_shape=0.7,
                        yery_shape=0.65
                    ),
                    x_height=60.0,
                    cap_height=70.0,
                    ascender=75.0,
                    descender=15.0,
                    letter_spacing=0.5,
                    word_spacing=3.0,
                    density=0.65
                ),
                popularity=0.4,
                cyrillic_support=True,
                designer="Geoffrey Lee",
                year=1965,
                foundry="Stephenson Blake",
                description="Сверхжирный гротеск для заголовков и акцентов",
                license="Commercial"
            ),
            
            FontInfo(
                id=17,
                name="Cooper Black",
                category=FontCategory.DISPLAY,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.3,
                    contrast=0.2,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.75,
                        fi_shape=0.85,
                        shcha_shape=0.8,
                        yery_shape=0.75
                    ),
                    x_height=58.0,
                    cap_height=70.0,
                    ascender=75.0,
                    descender=15.0,
                    letter_spacing=1.0,
                    word_spacing=4.0,
                    density=0.7
                ),
                popularity=0.15,
                cyrillic_support=False,
                designer="Oswald Bruce Cooper",
                year=1922,
                foundry="Barnhart Brothers & Spindler",
                description="Жирная декоративная антиква американского стиля",
                license="Commercial"
            ),
            
            FontInfo(
                id=18,
                name="Bebas Neue",
                category=FontCategory.DISPLAY,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.22,
                    contrast=0.05,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.7,
                        zh_shape=0.8,
                        fi_shape=0.75,
                        shcha_shape=0.8,
                        yery_shape=0.75
                    ),
                    x_height=70.0,
                    cap_height=70.0,
                    ascender=70.0,
                    descender=0.0,
                    letter_spacing=2.5,
                    word_spacing=7.0,
                    density=0.5
                ),
                popularity=0.5,
                cyrillic_support=True,
                designer="Ryoichi Tsunekawa",
                year=2010,
                foundry="Dharma Type",
                description="Современный конденсированный дисплейный шрифт",
                license="OFL"
            ),
            
            FontInfo(
                id=19,
                name="Lobster",
                category=FontCategory.SCRIPT,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.2,
                    contrast=0.6,
                    slant=15.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9,
                        zh_shape=0.85,
                        fi_shape=0.9,
                        shcha_shape=0.9,
                        yery_shape=0.85
                    ),
                    x_height=45.0,
                    cap_height=65.0,
                    ascender=80.0,
                    descender=25.0,
                    letter_spacing=1.0,
                    word_spacing=4.0,
                    density=0.35
                ),
                popularity=0.3,
                cyrillic_support=True,
                designer="Pablo Impallari",
                year=2010,
                foundry="Pablo Impallari",
                description="Жирный рукописный шрифт с ретро-характером",
                license="OFL"
            ),
            
            FontInfo(
                id=20,
                name="Pacifico",
                category=FontCategory.SCRIPT,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.18,
                    contrast=0.4,
                    slant=10.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85,
                        zh_shape=0.8,
                        fi_shape=0.85,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=50.0,
                    cap_height=70.0,
                    ascender=85.0,
                    descender=30.0,
                    letter_spacing=0.8,
                    word_spacing=3.5,
                    density=0.32
                ),
                popularity=0.25,
                cyrillic_support=False,
                designer="Vernon Adams",
                year=2011,
                foundry="Google Fonts",
                description="Непринужденный рукописный шрифт в стиле серфинга",
                license="OFL"
            ),
            
            # === МОНОШИРИННЫЕ ШРИФТЫ ===
            
            FontInfo(
                id=21,
                name="Courier New",
                category=FontCategory.MONOSPACE,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.14,
                    contrast=0.3,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.75,
                        zh_shape=0.8,
                        fi_shape=0.8,
                        shcha_shape=0.8,
                        yery_shape=0.75
                    ),
                    x_height=50.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=0.0,
                    word_spacing=10.0,
                    density=0.35
                ),
                popularity=0.6,
                cyrillic_support=True,
                designer="Howard Kettler",
                year=1955,
                foundry="IBM",
                description="Классический моноширинный шрифт пишущих машинок",
                license="Commercial"
            ),
            
            FontInfo(
                id=22,
                name="Fira Code",
                category=FontCategory.MONOSPACE,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.2,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.85,
                        fi_shape=0.8,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=52.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=0.0,
                    word_spacing=10.0,
                    density=0.4
                ),
                popularity=0.3,
                cyrillic_support=True,
                designer="Nikita Prokopov",
                year=2014,
                foundry="Mozilla",
                description="Моноширинный шрифт для программирования с лигатурами",
                license="OFL"
            ),
            
            # === РЕДКИЕ И СПЕЦИАЛЬНЫЕ ШРИФТЫ ===
            
            FontInfo(
                id=23,
                name="Trajan Pro",
                category=FontCategory.DISPLAY,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.12,
                    contrast=0.8,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9,
                        zh_shape=0.85,
                        fi_shape=0.9,
                        shcha_shape=0.9,
                        yery_shape=0.85
                    ),
                    x_height=0.0,
                    cap_height=70.0,
                    ascender=70.0,
                    descender=0.0,
                    letter_spacing=3.0,
                    word_spacing=8.0,
                    density=0.3
                ),
                popularity=0.1,
                cyrillic_support=False,
                designer="Carol Twombly",
                year=1989,
                foundry="Adobe",
                description="Капительный шрифт на основе колонны Траяна",
                license="Commercial"
            ),
            
            FontInfo(
                id=24,
                name="Optima",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.15,
                    contrast=0.4,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.85,
                        fi_shape=0.8,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=53.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.8,
                    word_spacing=5.5,
                    density=0.42
                ),
                popularity=0.2,
                cyrillic_support=True,
                designer="Hermann Zapf",
                year=1958,
                foundry="Stempel",
                description="Гуманистический гротеск с каллиграфическими чертами",
                license="Commercial"
            ),
            
            FontInfo(
                id=25,
                name="Univers",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.17,
                    contrast=0.15,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.75,
                        zh_shape=0.8,
                        fi_shape=0.75,
                        shcha_shape=0.8,
                        yery_shape=0.75
                    ),
                    x_height=54.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.5,
                    word_spacing=5.0,
                    density=0.44
                ),
                popularity=0.25,
                cyrillic_support=True,
                designer="Adrian Frutiger",
                year=1957,
                foundry="Deberny & Peignot",
                description="Неогротеск швейцарской школы типографики",
                license="Commercial"
            ),
            
            FontInfo(
                id=26,
                name="Frutiger",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.25,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.85,
                        fi_shape=0.8,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=53.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.6,
                    word_spacing=5.2,
                    density=0.43
                ),
                popularity=0.15,
                cyrillic_support=True,
                designer="Adrian Frutiger",
                year=1976,
                foundry="Linotype",
                description="Гуманистический гротеск для аэропорта Шарля де Голля",
                license="Commercial"
            ),
            
            FontInfo(
                id=27,
                name="Caslon",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.13,
                    contrast=0.7,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85,
                        zh_shape=0.8,
                        fi_shape=0.9,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=48.0,
                    cap_height=68.0,
                    ascender=82.0,
                    descender=22.0,
                    letter_spacing=2.2,
                    word_spacing=6.5,
                    density=0.37
                ),
                popularity=0.1,
                cyrillic_support=False,
                designer="William Caslon",
                year=1722,
                foundry="Caslon",
                description="Старостильная антиква XVIII века",
                license="Public Domain"
            ),
            
            FontInfo(
                id=28,
                name="Garamond",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.12,
                    contrast=0.75,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9,
                        zh_shape=0.85,
                        fi_shape=0.95,
                        shcha_shape=0.9,
                        yery_shape=0.85
                    ),
                    x_height=46.0,
                    cap_height=67.0,
                    ascender=83.0,
                    descender=23.0,
                    letter_spacing=2.5,
                    word_spacing=7.0,
                    density=0.34
                ),
                popularity=0.2,
                cyrillic_support=True,
                designer="Claude Garamond",
                year=1530,
                foundry="Various",
                description="Классическая старостильная антиква эпохи Возрождения",
                license="Various"
            ),
            
            FontInfo(
                id=29,
                name="Bodoni",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.1,
                    contrast=0.9,
                    slant=0.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85,
                        zh_shape=0.8,
                        fi_shape=0.9,
                        shcha_shape=0.85,
                        yery_shape=0.8
                    ),
                    x_height=45.0,
                    cap_height=70.0,
                    ascender=85.0,
                    descender=25.0,
                    letter_spacing=2.8,
                    word_spacing=7.5,
                    density=0.32
                ),
                popularity=0.15,
                cyrillic_support=True,
                designer="Giambattista Bodoni",
                year=1798,
                foundry="Various",
                description="Классицистическая антиква с высоким контрастом",
                license="Various"
            ),
            
            FontInfo(
                id=30,
                name="Brush Script MT",
                category=FontCategory.SCRIPT,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.25,
                    contrast=0.5,
                    slant=20.0,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8,
                        zh_shape=0.75,
                        fi_shape=0.8,
                        shcha_shape=0.8,
                        yery_shape=0.75
                    ),
                    x_height=40.0,
                    cap_height=60.0,
                    ascender=85.0,
                    descender=35.0,
                    letter_spacing=0.5,
                    word_spacing=3.0,
                    density=0.28
                ),
                popularity=0.2,
                cyrillic_support=False,
                designer="Robert E. Smith",
                year=1942,
                foundry="American Type Founders",
                description="Имитация письма кистью",
                license="Commercial"
            )
        ]
    
    async def get_fonts(self, category: Optional[str] = None) -> List[FontInfo]:
        """Получение списка шрифтов"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._get_fonts_sync, 
            category
        )
    
    def _get_fonts_sync(self, category: Optional[str] = None) -> List[FontInfo]:
        """Синхронное получение шрифтов из всех источников"""
        # Объединяем локальные и Google Fonts
        all_fonts = self.fonts.copy() + self._google_fonts_cache.copy()
        
        if category:
            return [font for font in all_fonts if font.category.value == category]
        return all_fonts
    
    def get_all_fonts_sync(self) -> List[FontInfo]:
        """Синхронное получение всех шрифтов из всех источников"""
        return self.fonts.copy() + self._google_fonts_cache.copy()
    
    async def get_font_by_id(self, font_id: int) -> Optional[FontInfo]:
        """Получение шрифта по ID"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_font_by_id_sync,
            font_id
        )
    
    def _get_font_by_id_sync(self, font_id: int) -> Optional[FontInfo]:
        """Синхронное получение шрифта по ID из всех источников"""
        # Сначала ищем в локальных шрифтах
        for font in self.fonts:
            if font.id == font_id:
                return font
        
        # Потом в Google Fonts
        for font in self._google_fonts_cache:
            if font.id == font_id:
                return font
        
        return None
    
    async def search_fonts(self, query: str) -> List[FontInfo]:
        """Поиск шрифтов по названию"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._search_fonts_sync,
            query
        )
    
    def _search_fonts_sync(self, query: str) -> List[FontInfo]:
        """Синхронный поиск шрифтов во всех источниках"""
        query_lower = query.lower()
        results = []
        
        # Объединяем все шрифты
        all_fonts = self.fonts + self._google_fonts_cache
        
        for font in all_fonts:
            if (query_lower in font.name.lower() or
                (font.designer and query_lower in font.designer.lower()) or
                (font.foundry and query_lower in font.foundry.lower())):
                results.append(font)
        
        return results
    
    def get_popular_fonts(self, min_popularity: float = 0.7) -> List[FontInfo]:
        """Получение популярных шрифтов из всех источников"""
        all_fonts = self.fonts + self._google_fonts_cache
        return sorted(
            [font for font in all_fonts if font.popularity >= min_popularity],
            key=lambda x: x.popularity,
            reverse=True
        )
    
    def get_fonts_by_category(self, category: FontCategory) -> List[FontInfo]:
        """Получение шрифтов по категории из всех источников"""
        all_fonts = self.fonts + self._google_fonts_cache
        return [font for font in all_fonts if font.category == category]
    
    async def refresh_google_fonts(self) -> bool:
        """Принудительное обновление Google Fonts"""
        try:
            logger.info("Принудительное обновление Google Fonts...")
            google_fonts = await self.google_fonts_service.get_popular_fonts(limit=200)
            self._google_fonts_cache = google_fonts
            logger.info(f"Google Fonts обновлены: {len(google_fonts)} шрифтов")
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления Google Fonts: {str(e)}")
            return False
    
    def _add_popular_fonts_to_local_database(self):
        """Добавляем популярные шрифты в локальную базу (имитация Google Fonts)"""
        
        # Популярные шрифты которые часто встречаются в дизайне
        popular_fonts = [
            # Sans-Serif популярные
            FontInfo(
                id=1001,
                name="Roboto",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.3,
                    slant=0.0,
                    x_height=52.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.5,
                    word_spacing=5.0,
                    density=0.42,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9, zh_shape=0.9, fi_shape=0.9, shcha_shape=0.95, yery_shape=0.9
                    )
                ),
                popularity=0.95,
                cyrillic_support=True,
                designer="Christian Robertson",
                year=2011,
                foundry="Google",
                description="Современный геометрический гротеск, один из самых популярных веб-шрифтов",
                license="Apache 2.0"
            ),
            
            FontInfo(
                id=1002,
                name="Open Sans",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.15,
                    contrast=0.25,
                    slant=0.0,
                    x_height=54.0,
                    cap_height=72.0,
                    ascender=82.0,
                    descender=22.0,
                    letter_spacing=1.6,
                    word_spacing=5.2,
                    density=0.44,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85, zh_shape=0.85, fi_shape=0.85, shcha_shape=0.9, yery_shape=0.85
                    )
                ),
                popularity=0.92,
                cyrillic_support=True,
                designer="Steve Matteson",
                year=2011,
                foundry="Ascender Corporation",
                description="Гуманистический гротеск с отличной читаемостью",
                license="Apache 2.0"
            ),
            
            FontInfo(
                id=1003,
                name="Lato",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.14,
                    contrast=0.2,
                    slant=0.0,
                    x_height=50.0,
                    cap_height=68.0,
                    ascender=78.0,
                    descender=18.0,
                    letter_spacing=1.4,
                    word_spacing=4.8,
                    density=0.40,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.8, zh_shape=0.8, fi_shape=0.8, shcha_shape=0.85, yery_shape=0.8
                    )
                ),
                popularity=0.88,
                cyrillic_support=True,
                designer="Łukasz Dziedzic",
                year=2010,
                foundry="tyPoland",
                description="Элегантный гротеск с полукруглыми деталями",
                license="OFL"
            ),
            
            # Serif популярные
            FontInfo(
                id=1004,
                name="Playfair Display",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.18,
                    contrast=0.8,
                    slant=0.0,
                    x_height=45.0,
                    cap_height=72.0,
                    ascender=85.0,
                    descender=25.0,
                    letter_spacing=2.0,
                    word_spacing=6.0,
                    density=0.35,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9, zh_shape=0.9, fi_shape=0.95, shcha_shape=0.95, yery_shape=0.9
                    )
                ),
                popularity=0.85,
                cyrillic_support=True,
                designer="Claus Eggers Sørensen",
                year=2011,
                foundry="Google",
                description="Элегантный заголовочный антикв с высоким контрастом",
                license="OFL"
            ),
            
            FontInfo(
                id=1005,
                name="Merriweather",
                category=FontCategory.SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=True,
                    stroke_width=0.16,
                    contrast=0.6,
                    slant=0.0,
                    x_height=48.0,
                    cap_height=70.0,
                    ascender=82.0,
                    descender=24.0,
                    letter_spacing=2.2,
                    word_spacing=6.5,
                    density=0.38,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85, zh_shape=0.85, fi_shape=0.9, shcha_shape=0.9, yery_shape=0.85
                    )
                ),
                popularity=0.82,
                cyrillic_support=True,
                designer="Eben Sorkin",
                year=2010,
                foundry="Sorkin Type",
                description="Читабельный антикв для длинных текстов",
                license="OFL"
            ),
            
            # Monospace популярные
            FontInfo(
                id=1006,
                name="JetBrains Mono",
                category=FontCategory.MONOSPACE,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.15,
                    contrast=0.2,
                    slant=0.0,
                    x_height=52.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=0.0,
                    word_spacing=10.0,
                    density=0.4,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9, zh_shape=0.9, fi_shape=0.9, shcha_shape=0.9, yery_shape=0.9
                    )
                ),
                popularity=0.75,
                cyrillic_support=True,
                designer="JetBrains",
                year=2020,
                foundry="JetBrains",
                description="Современный моноширинный шрифт для программирования",
                license="Apache 2.0"
            ),
            
            # Display популярные
            FontInfo(
                id=1007,
                name="Montserrat",
                category=FontCategory.DISPLAY,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.2,
                    contrast=0.1,
                    slant=0.0,
                    x_height=58.0,
                    cap_height=72.0,
                    ascender=78.0,
                    descender=18.0,
                    letter_spacing=2.5,
                    word_spacing=6.5,
                    density=0.48,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85, zh_shape=0.85, fi_shape=0.85, shcha_shape=0.9, yery_shape=0.85
                    )
                ),
                popularity=0.90,
                cyrillic_support=True,
                designer="Julieta Ulanovsky",
                year=2011,
                foundry="Google",
                description="Геометрический гротеск вдохновленный вывесками Монсеррата",
                license="OFL"
            ),
            
            # Кириллические специальные
            FontInfo(
                id=1008,
                name="Yandex Sans",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.25,
                    slant=0.0,
                    x_height=53.0,
                    cap_height=71.0,
                    ascender=81.0,
                    descender=21.0,
                    letter_spacing=1.5,
                    word_spacing=5.0,
                    density=0.43,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.95, zh_shape=0.95, fi_shape=0.95, shcha_shape=0.98, yery_shape=0.95
                    )
                ),
                popularity=0.70,
                cyrillic_support=True,
                designer="Yandex Design",
                year=2019,
                foundry="Yandex",
                description="Корпоративный шрифт Яндекса с превосходной кириллицей",
                license="Proprietary"
            ),
            
            FontInfo(
                id=1009,
                name="PT Sans",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.15,
                    contrast=0.3,
                    slant=0.0,
                    x_height=51.0,
                    cap_height=69.0,
                    ascender=79.0,
                    descender=19.0,
                    letter_spacing=1.4,
                    word_spacing=4.9,
                    density=0.41,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.9, zh_shape=0.9, fi_shape=0.9, shcha_shape=0.95, yery_shape=0.9
                    )
                ),
                popularity=0.78,
                cyrillic_support=True,
                designer="Alexandra Korolkova",
                year=2009,
                foundry="ParaType",
                description="Российский гротеск из проекта общедоступных шрифтов",
                license="OFL"
            ),
            
            FontInfo(
                id=1010,
                name="Fira Sans",
                category=FontCategory.SANS_SERIF,
                characteristics=FontCharacteristics(
                    has_serifs=False,
                    stroke_width=0.16,
                    contrast=0.2,
                    slant=0.0,
                    x_height=52.0,
                    cap_height=70.0,
                    ascender=80.0,
                    descender=20.0,
                    letter_spacing=1.6,
                    word_spacing=5.1,
                    density=0.42,
                    cyrillic_features=CyrillicFeatures(
                        ya_shape=0.85, zh_shape=0.85, fi_shape=0.85, shcha_shape=0.9, yery_shape=0.85
                    )
                ),
                popularity=0.73,
                cyrillic_support=True,
                designer="Erik Spiekermann",
                year=2013,
                foundry="Mozilla",
                description="Гуманистический гротеск от Mozilla с отличной кириллицей",
                license="OFL"
            )
        ]
        
        # Добавляем популярные шрифты к основной базе
        self.fonts.extend(popular_fonts)
        logger.info(f"➕ Добавлено {len(popular_fonts)} популярных шрифтов в локальную базу")


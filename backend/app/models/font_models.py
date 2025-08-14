"""
Pydantic модели для работы с шрифтами
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class FontCategory(str, Enum):
    """Категории шрифтов"""
    SERIF = "serif"
    SANS_SERIF = "sans-serif"
    MONOSPACE = "monospace"
    DISPLAY = "display"
    HANDWRITING = "handwriting"
    SCRIPT = "script"


class CyrillicFeatures(BaseModel):
    """Характеристики кириллических букв"""
    ya_shape: float = Field(..., ge=0.0, le=1.0, description="Форма буквы Я")
    zh_shape: float = Field(..., ge=0.0, le=1.0, description="Форма буквы Ж")
    fi_shape: float = Field(..., ge=0.0, le=1.0, description="Форма буквы Ф")
    shcha_shape: float = Field(..., ge=0.0, le=1.0, description="Форма буквы Щ")
    yery_shape: float = Field(..., ge=0.0, le=1.0, description="Форма буквы Ы")


class FontCharacteristics(BaseModel):
    """Характеристики шрифта, извлеченные из изображения"""
    
    # Основные характеристики
    has_serifs: bool = Field(..., description="Наличие засечек")
    stroke_width: float = Field(..., ge=0.0, le=1.0, description="Толщина штрихов")
    contrast: float = Field(..., ge=0.0, le=1.0, description="Контраст между толстыми и тонкими штрихами")
    slant: float = Field(..., ge=-45.0, le=45.0, description="Наклон в градусах")
    
    # Кириллические особенности
    cyrillic_features: CyrillicFeatures = Field(..., description="Характеристики кириллических букв")
    
    # Геометрические характеристики
    x_height: float = Field(..., ge=0.0, description="Высота строчных букв")
    cap_height: float = Field(..., ge=0.0, description="Высота заглавных букв")
    ascender: float = Field(..., ge=0.0, description="Выносные элементы вверх")
    descender: float = Field(..., ge=0.0, description="Выносные элементы вниз")
    
    # Интервалы
    letter_spacing: float = Field(..., ge=0.0, description="Межбуквенное расстояние")
    word_spacing: float = Field(..., ge=0.0, description="Межсловное расстояние")
    density: float = Field(..., ge=0.0, le=1.0, description="Плотность текста")


class FontInfo(BaseModel):
    """Информация о шрифте"""
    id: Optional[int] = Field(None, description="ID шрифта в базе данных")
    name: str = Field(..., description="Название шрифта")
    category: FontCategory = Field(..., description="Категория шрифта")
    characteristics: FontCharacteristics = Field(..., description="Характеристики шрифта")
    popularity: float = Field(..., ge=0.0, le=1.0, description="Популярность шрифта (0-1)")
    cyrillic_support: bool = Field(..., description="Поддержка кириллицы")
    designer: Optional[str] = Field(None, description="Дизайнер шрифта")
    year: Optional[int] = Field(None, description="Год создания")
    foundry: Optional[str] = Field(None, description="Шрифтовая студия")
    description: Optional[str] = Field(None, description="Описание шрифта")
    license: Optional[str] = Field(None, description="Лицензия")
    download_url: Optional[str] = Field(None, description="Ссылка для скачивания")


class FontMatch(BaseModel):
    """Результат сопоставления шрифта"""
    font_info: FontInfo = Field(..., description="Информация о шрифте")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в совпадении")
    match_details: Dict[str, float] = Field(..., description="Детали сопоставления")


class FontAnalysisResult(BaseModel):
    """Результат анализа шрифта"""
    success: bool = Field(..., description="Успешность анализа")
    message: str = Field(..., description="Сообщение о результате")
    matches: List[FontMatch] = Field(default=[], description="Найденные совпадения")
    characteristics: Optional[FontCharacteristics] = Field(None, description="Извлеченные характеристики")
    error: Optional[str] = Field(None, description="Код ошибки")
    processing_time: Optional[float] = Field(None, description="Время обработки в секундах")


class AnalysisRequest(BaseModel):
    """Запрос на анализ шрифта"""
    max_results: int = Field(10, ge=1, le=50, description="Максимальное количество результатов")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Минимальная уверенность")
    categories: Optional[List[FontCategory]] = Field(None, description="Фильтр по категориям")


class ErrorResponse(BaseModel):
    """Модель ошибки"""
    error: str = Field(..., description="Тип ошибки")
    message: str = Field(..., description="Сообщение об ошибке")
    details: Optional[Dict[str, Any]] = Field(None, description="Дополнительные детали")


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""
    status: str = Field(..., description="Статус API")
    message: str = Field(..., description="Сообщение о состоянии")
    uptime: Optional[float] = Field(None, description="Время работы в секундах")
    version: Optional[str] = Field(None, description="Версия API")


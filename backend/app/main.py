"""
MyFonts Backend API
Определение кириллических шрифтов по изображениям
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import logging
import os
from pathlib import Path

from .services.font_analyzer import FontAnalyzer
from .services.font_matcher import FontMatcher
from .models.font_models import FontAnalysisResult, FontInfo, AnalysisRequest
from .database.font_database import FontDatabase

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="MyFonts API",
    description="API для определения кириллических шрифтов по изображениям",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS для работы с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite и React dev серверы
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализируем сервисы
font_analyzer = FontAnalyzer()
font_matcher = FontMatcher()
font_database = FontDatabase()

# Создаем папку для временных файлов
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    logger.info("Запуск MyFonts API...")
    await font_database.initialize()
    logger.info("База данных шрифтов инициализирована")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке приложения"""
    logger.info("Остановка MyFonts API...")


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "MyFonts API - Определение кириллических шрифтов",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Проверка здоровья API"""
    return {
        "status": "healthy",
        "message": "API работает нормально"
    }


@app.post("/api/refresh-fonts")
async def refresh_google_fonts():
    """Принудительное обновление Google Fonts"""
    try:
        success = await font_database.refresh_google_fonts()
        if success:
            total_fonts = len(font_database.get_all_fonts_sync())
            return {
                "success": True,
                "message": "Google Fonts успешно обновлены",
                "total_fonts": total_fonts
            }
        else:
            return {
                "success": False,
                "message": "Ошибка обновления Google Fonts"
            }
    except Exception as e:
        logger.error(f"Ошибка обновления шрифтов: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обновления шрифтов: {str(e)}"
        )


@app.post("/api/analyze-font", response_model=FontAnalysisResult)
async def analyze_font(file: UploadFile = File(...)):
    """
    Анализ шрифта по загруженному изображению
    """
    try:
        # Проверяем тип файла
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Файл должен быть изображением (PNG, JPG, JPEG, GIF, BMP, WebP)"
            )
        
        # Читаем содержимое файла
        contents = await file.read()
        
        logger.info(f"Получен файл для анализа: {file.filename}, размер: {len(contents)} байт")
        
        # Анализируем изображение
        characteristics = await font_analyzer.analyze_image(contents)
        logger.info("Характеристики извлечены успешно")
        
        # Находим похожие шрифты
        matches = font_matcher.find_matches(characteristics)
        logger.info(f"Найдено {len(matches)} совпадений")
        
        return FontAnalysisResult(
            success=True,
            message="Анализ завершен успешно",
            matches=matches,
            characteristics=characteristics
        )
        
    except Exception as e:
        logger.error(f"Ошибка при анализе шрифта: {str(e)}")
        
        # Если это ошибка отсутствия текста, возвращаем специальное сообщение
        if "не обнаружен текст" in str(e).lower():
            return FontAnalysisResult(
                success=False,
                message="На изображении не обнаружен текст для анализа",
                matches=[],
                error="NO_TEXT_DETECTED"
            )
        
        # Если это ошибка множественных шрифтов, возвращаем специальное сообщение
        if "несколько разных шрифтов" in str(e).lower():
            return FontAnalysisResult(
                success=False,
                message="На изображении обнаружено несколько разных шрифтов. Для точного анализа загрузите изображение с текстом одного шрифта.",
                matches=[],
                error="MULTIPLE_FONTS_DETECTED"
            )
        
        # Для других ошибок возвращаем общее сообщение
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при анализе изображения: {str(e)}"
        )


@app.get("/api/fonts", response_model=List[FontInfo])
async def get_fonts(category: Optional[str] = None):
    """
    Получение списка всех шрифтов или по категории
    """
    try:
        fonts = await font_database.get_fonts(category=category)
        return fonts
    except Exception as e:
        logger.error(f"Ошибка при получении списка шрифтов: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при получении списка шрифтов")


@app.get("/api/fonts/{font_id}", response_model=FontInfo)
async def get_font_by_id(font_id: int):
    """
    Получение информации о конкретном шрифте
    """
    try:
        font = await font_database.get_font_by_id(font_id)
        if not font:
            raise HTTPException(status_code=404, detail="Шрифт не найден")
        return font
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении шрифта {font_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при получении информации о шрифте")


@app.get("/api/fonts/search/{query}")
async def search_fonts(query: str):
    """
    Поиск шрифтов по названию
    """
    try:
        fonts = await font_database.search_fonts(query)
        return fonts
    except Exception as e:
        logger.error(f"Ошибка при поиске шрифтов '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка при поиске шрифтов")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


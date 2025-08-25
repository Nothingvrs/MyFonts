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
import asyncio

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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],  # dev серверы (Vite/React)
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
    try:
        await font_database.initialize()
        logger.info("База данных шрифтов инициализирована")
    except asyncio.CancelledError:
        # Перезагрузка uvicorn/моделей может отменить startup — не считаем это фатальной ошибкой
        logger.warning("Инициализация базы шрифтов отменена (reload), продолжаем без ожидания")
    except Exception as e:
        logger.warning(f"База данных шрифтов не инициализирована: {str(e)}")
        logger.info("Продолжаем работу без базы данных шрифтов")


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
    try:
        # Проверяем статус PaddleOCR
        paddleocr_status = "available" if font_analyzer.paddleocr_service.is_available() else "unavailable"
        
        # Проверяем базу данных
        db_status = "available"
        try:
            await font_database.get_fonts(limit=1)
        except:
            db_status = "unavailable"
        
        overall_status = "healthy" if paddleocr_status == "available" and db_status == "available" else "degraded"
        
        return {
            "status": overall_status,
            "message": "API работает" if overall_status == "healthy" else "API работает с ограничениями",
            "services": {
                "paddleocr": paddleocr_status,
                "database": db_status
            },
            "paddleocr_available": paddleocr_status == "available"
        }
    except Exception as e:
        logger.error(f"Ошибка health check: {str(e)}")
        return {
            "status": "unhealthy",
            "message": "API недоступен",
            "error": str(e)
        }


@app.get("/api/paddleocr-status")
async def paddleocr_status():
    """Проверка статуса PaddleOCR"""
    try:
        is_available = font_analyzer.paddleocr_service.is_available()
        
        if is_available:
            return {
                "status": "available",
                "message": "PaddleOCR доступен и готов к работе",
                "can_analyze": True
            }
        else:
            return {
                "status": "unavailable",
                "message": "PaddleOCR недоступен - анализ шрифтов невозможен",
                "can_analyze": False,
                "suggestion": "Попробуйте позже или обратитесь к администратору"
            }
    except Exception as e:
        logger.error(f"Ошибка проверки статуса PaddleOCR: {str(e)}")
        return {
            "status": "error",
            "message": "Ошибка проверки статуса PaddleOCR",
            "can_analyze": False,
            "error": str(e)
        }


@app.post("/api/paddleocr-reinit")
async def paddleocr_reinit():
    """Принудительная переинициализация PaddleOCR"""
    try:
        logger.info("🔄 Принудительная переинициализация PaddleOCR...")
        
        # Переинициализируем PaddleOCR
        success = font_analyzer.paddleocr_service.reinitialize()
        
        if success:
            return {
                "success": True,
                "message": "PaddleOCR успешно переинициализирован",
                "status": "available"
            }
        else:
            return {
                "success": False,
                "message": "PaddleOCR не удалось переинициализировать",
                "status": "unavailable"
            }
    except Exception as e:
        logger.error(f"Ошибка переинициализации PaddleOCR: {str(e)}")
        return {
            "success": False,
            "message": f"Ошибка переинициализации: {str(e)}",
            "status": "error"
        }


@app.post("/api/paddleocr-test")
async def paddleocr_test(file: UploadFile = File(...)):
    """Тестирование PaddleOCR на конкретном изображении"""
    try:
        logger.info("🧪 Тестирование PaddleOCR на изображении...")
        
        # Проверяем тип файла
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Файл должен быть изображением"
            )
        
        # Читаем содержимое файла
        contents = await file.read()
        logger.info(f"📁 Получен файл: {file.filename}, размер: {len(contents)} байт")
        
        # Загружаем изображение
        import cv2
        import numpy as np
        from PIL import Image
        import io
        
        # Конвертируем bytes в numpy array
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        logger.info(f"🖼️ Изображение загружено: {image_np.shape}")
        
        # Тестируем PaddleOCR
        result = await font_analyzer.paddleocr_service.detect_and_analyze_text(image_np)
        
        return {
            "success": True,
            "message": "Тест PaddleOCR завершен",
            "result": result,
            "image_info": {
                "shape": image_np.shape,
                "filename": file.filename,
                "size_bytes": len(contents)
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка тестирования PaddleOCR: {str(e)}")
        return {
            "success": False,
            "message": f"Ошибка тестирования: {str(e)}",
            "error": str(e)
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
        
        logger.info(f"📁 Получен файл: {file.filename}, размер: {len(contents)} байт")
        
        # Анализируем изображение
        logger.info("🧠 Запускаем анализ изображения...")
        characteristics = await font_analyzer.analyze_image(contents)
        logger.info("✅ Характеристики извлечены успешно")
        
        # Находим похожие шрифты (теперь асинхронно с полной базой Google Fonts)
        logger.info("🔎 Ищем похожие шрифты...")
        matches = await font_matcher.find_matches(characteristics)
        logger.info(f"✅ Найдено {len(matches)} совпадений")
        
        return FontAnalysisResult(
            success=True,
            message="Анализ завершен успешно",
            matches=matches,
            characteristics=characteristics
        )
        
    except Exception as e:
        logger.error(f"💥 КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        logger.error(f"📊 Тип ошибки: {type(e)}")
        import traceback
        logger.error(f"🔍 Трассировка: {traceback.format_exc()}")
        
        # Если это ошибка отсутствия текста, возвращаем специальное сообщение
        if "не обнаружен текст" in str(e).lower() or "не содержит читаемый текст" in str(e).lower():
            return FontAnalysisResult(
                success=False,
                message="На изображении не обнаружен текст для анализа. Попробуйте загрузить изображение с четким, читаемым текстом.",
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
        
        # Если это ошибка PaddleOCR
        if "paddleocr" in str(e).lower() or "ocr" in str(e).lower() or "ocr не смог найти текст" in str(e).lower():
            return FontAnalysisResult(
                success=False,
                message="OCR не смог найти текст на изображении. Попробуйте загрузить изображение с более четким текстом.",
                matches=[],
                error="OCR_ERROR"
            )
        
        # Если это ошибка недоступности ИИ
        if "ии для анализа шрифтов временно недоступен" in str(e).lower() or "сервис анализа шрифтов временно недоступен" in str(e).lower():
            return FontAnalysisResult(
                success=False,
                message="ИИ для анализа шрифтов временно недоступен. Попробуйте позже или обратитесь к администратору.",
                matches=[],
                error="AI_SERVICE_UNAVAILABLE"
            )
        
        # Если это ошибка инициализации PaddleOCR
        if "не инициализирован" in str(e).lower() or "недоступен" in str(e).lower():
            return FontAnalysisResult(
                success=False,
                message="Сервис анализа шрифтов временно недоступен. Попробуйте позже или обратитесь к администратору.",
                matches=[],
                error="SERVICE_UNAVAILABLE"
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


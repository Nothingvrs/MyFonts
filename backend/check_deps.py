"""
Проверка зависимостей
"""

print("Проверяем зависимости...")

# Базовые зависимости
try:
    import fastapi
    print("✅ FastAPI:", fastapi.__version__)
except ImportError as e:
    print("❌ FastAPI:", e)

try:
    import uvicorn
    print("✅ Uvicorn:", uvicorn.__version__)
except ImportError as e:
    print("❌ Uvicorn:", e)

try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV:", e)

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError as e:
    print("❌ NumPy:", e)

try:
    from PIL import Image
    print("✅ Pillow: OK")
except ImportError as e:
    print("❌ Pillow:", e)

# PaddleOCR
try:
    import paddle
    print("✅ PaddlePaddle:", paddle.__version__)
except ImportError as e:
    print("❌ PaddlePaddle:", e)

try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR: OK")
except ImportError as e:
    print("❌ PaddleOCR:", e)

# Попытка создать PaddleOCR
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='ru')
    print("✅ PaddleOCR инициализация: OK")
except Exception as e:
    print("❌ PaddleOCR инициализация:", e)

print("\nПроверка завершена!")


#!/usr/bin/env python3
"""
Базовый тест PaddleOCR - минимальный код для проверки
"""

print("🧪 Тестируем PaddleOCR...")

try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR импортирован")
except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    exit(1)

try:
    print("🚀 Инициализация PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='ru')
    print("✅ PaddleOCR инициализирован")
except Exception as e:
    print(f"❌ Ошибка инициализации: {e}")
    exit(1)

# Создаем простое изображение
import numpy as np
import cv2

# Белое изображение с черным текстом
img = np.ones((100, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, 'TEST', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

print(f"🖼️ Создано изображение: {img.shape}")

try:
    print("🔍 Запуск OCR...")
    result = ocr.ocr(img)
    print(f"📊 Результат: {result}")
    
    if result and result[0]:
        print(f"🎯 Найдено {len(result[0])} областей текста")
        for i, line in enumerate(result[0]):
            print(f"  {i+1}. {line}")
    else:
        print("⚠️ Текст не найден")
        
except Exception as e:
    print(f"❌ Ошибка OCR: {e}")

print("🏁 Тест завершен")

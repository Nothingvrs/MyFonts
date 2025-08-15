#!/usr/bin/env python3
"""
Простой тест PaddleOCR согласно официальной документации
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np

# Инициализация согласно документации
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Сначала английский для простоты

# Создаем простое изображение с текстом
img = np.ones((100, 300, 3), dtype=np.uint8) * 255
cv2.putText(img, 'Hello World', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

print("Тестируем PaddleOCR...")
print(f"Размер изображения: {img.shape}")

# Запускаем OCR
result = ocr.ocr(img, cls=True)

print(f"Результат OCR: {result}")
print("Тест завершен!")

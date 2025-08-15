#!/usr/bin/env python3
"""
Простой тест PaddleOCR для диагностики проблемы
"""

import numpy as np
import cv2
from PIL import Image
import io

try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR импортирован успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта PaddleOCR: {e}")
    exit(1)

def test_paddleocr():
    """Тестируем PaddleOCR с простым изображением"""
    print("🚀 Инициализация PaddleOCR...")
    
    try:
        # Инициализируем PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='ru')
        print("✅ PaddleOCR инициализирован")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        return False
    
    # Создаем простое тестовое изображение с текстом
    print("🖼️ Создаем тестовое изображение...")
    
    # Белый фон 400x200
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    
    # Добавляем черный текст
    cv2.putText(img, 'Test Text', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    print(f"📊 Размер изображения: {img.shape}, тип: {img.dtype}")
    
    # Тестируем OCR
    try:
        print("🔍 Запуск OCR...")
        result = ocr.ocr(img)
        
        print(f"✅ OCR выполнен!")
        print(f"📊 Тип результата: {type(result)}")
        print(f"📝 Результат: {result}")
        
        if result and len(result) > 0 and result[0]:
            print(f"🎯 Найдено {len(result[0])} текстовых областей")
            for i, line in enumerate(result[0]):
                if len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    print(f"  {i+1}. bbox: {bbox}")
                    print(f"     text: {text_info}")
        else:
            print("⚠️ Текст не найден")
            
        return True
        
    except Exception as e:
        print(f"❌ Ошибка OCR: {e}")
        print(f"📊 Тип ошибки: {type(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Тестирование PaddleOCR...")
    success = test_paddleocr()
    
    if success:
        print("🎉 Тест успешен!")
    else:
        print("💥 Тест провален!")

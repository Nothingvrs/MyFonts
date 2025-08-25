#!/usr/bin/env python3
"""
Простой тест PaddleOCR без запуска сервера
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.paddleocr_service import PaddleOCRService
import numpy as np
import cv2

def test_paddleocr():
    """Тестируем PaddleOCR"""
    print("🚀 Тестируем PaddleOCR...")
    
    # Создаем тестовое изображение с текстом
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Test Text", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    
    print(f"📸 Создано тестовое изображение: {img.shape}")
    
    # Инициализируем PaddleOCR
    service = PaddleOCRService()
    
    if not service.is_available():
        print("❌ PaddleOCR недоступен")
        return False
    
    print("✅ PaddleOCR доступен")
    
    # Тестируем анализ
    try:
        result = service._run_ocr_sync(img)
        print(f"🔍 Результат анализа: {result}")
        
        if result.get('has_text'):
            print("✅ Текст найден!")
            print(f"📝 Содержимое: {result.get('text_content')}")
            print(f"🎯 Уверенность: {result.get('confidence')}")
            print(f"🔤 Множественные шрифты: {result.get('multiple_fonts')}")
        else:
            print("❌ Текст не найден")
            print(f"💡 Ошибка: {result.get('error')}")
        
        return result.get('has_text', False)
        
    except Exception as e:
        print(f"💥 Ошибка анализа: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_paddleocr()
    if success:
        print("🎉 Тест PaddleOCR прошел успешно!")
    else:
        print("💥 Тест PaddleOCR не прошел!")

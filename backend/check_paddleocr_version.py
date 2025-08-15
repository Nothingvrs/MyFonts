"""
Проверка версии PaddleOCR и диагностика
"""

print("=== ДИАГНОСТИКА PADDLEOCR ===")

try:
    import paddleocr
    print(f"✅ PaddleOCR версия: {paddleocr.__version__}")
except Exception as e:
    print(f"❌ Ошибка импорта PaddleOCR: {e}")

try:
    import paddle
    print(f"✅ PaddlePaddle версия: {paddle.__version__}")
except Exception as e:
    print(f"❌ Ошибка импорта PaddlePaddle: {e}")

# Тестовая инициализация
try:
    from paddleocr import PaddleOCR
    print("📊 Пробуем инициализировать PaddleOCR...")
    
    # Согласно документации PaddleOCR 3.0
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Английский для начала
    print("✅ PaddleOCR успешно инициализирован")
    
    # Тестируем на простом тексте
    import numpy as np
    
    # Создаем простое тестовое изображение
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255  # Белый фон
    
    print("📊 Тестируем OCR на простом изображении...")
    result = ocr.ocr(test_image)
    print(f"📊 Результат теста: {result}")
    print(f"📊 Тип результата: {type(result)}")
    
except Exception as e:
    print(f"❌ Ошибка тестирования: {e}")
    import traceback
    traceback.print_exc()

print("\n=== РЕКОМЕНДАЦИИ ===")
print("Если версия PaddleOCR < 2.7, обновите:")
print("pip install --upgrade paddleocr")
print("\nЕсли версия PaddlePaddle < 2.5, обновите:")
print("pip install --upgrade paddlepaddle")

#!/usr/bin/env python3
"""
Простейший тест PaddleOCR для диагностики
"""
import sys
import os
import numpy as np
import cv2

print("🔍 Диагностика PaddleOCR")
print("=" * 50)

# 1. Проверим версии библиотек
print("📦 Версии библиотек:")
try:
    import paddlepaddle
    print(f"  - PaddlePaddle: {paddlepaddle.__version__}")
except:
    print("  - PaddlePaddle: НЕ УСТАНОВЛЕН")

try:
    import paddleocr
    print(f"  - PaddleOCR: {paddleocr.__version__}")
except:
    print("  - PaddleOCR: НЕ УСТАНОВЛЕН")

print(f"  - OpenCV: {cv2.__version__}")
print(f"  - NumPy: {np.__version__}")

# 2. Попробуем импортировать PaddleOCR
print("\n🔄 Импорт PaddleOCR...")
try:
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR импортирован успешно")
except Exception as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

# 3. Создадим простое тестовое изображение
print("\n🖼️ Создание тестового изображения...")
test_image = np.ones((100, 400, 3), dtype=np.uint8) * 255  # Белый фон
cv2.putText(test_image, 'HELLO WORLD', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Сохраним для проверки
cv2.imwrite('test_image.png', test_image)
print("✅ Тестовое изображение создано: test_image.png")

# 4. Попробуем создать PaddleOCR объект
print("\n🚀 Создание PaddleOCR объекта...")
try:
    # Самая простая конфигурация
    ocr = PaddleOCR(lang='en', use_angle_cls=False)
    print("✅ PaddleOCR объект создан успешно")
except Exception as e:
    print(f"❌ Ошибка создания PaddleOCR: {e}")
    print(f"💡 Тип ошибки: {type(e).__name__}")
    sys.exit(1)

# 5. Попробуем распознать текст
print("\n📖 Тест распознавания...")
try:
    result = ocr.ocr(test_image)
    print(f"✅ OCR выполнен, результат: {result}")
    
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            print(f"  📝 Найден текст: '{text}' (уверенность: {confidence:.2f})")
    else:
        print("⚠️ Текст не найден")
        
except Exception as e:
    print(f"❌ Ошибка OCR: {e}")
    print(f"💡 Тип ошибки: {type(e).__name__}")

# 6. Попробуем с русским языком
print("\n🇷🇺 Тест с русским языком...")
try:
    # Создадим изображение с русским текстом
    russian_image = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(russian_image, 'PRIVET MIR', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite('test_russian.png', russian_image)
    
    # Создадим OCR для русского
    ru_ocr = PaddleOCR(lang='ru', use_angle_cls=False)
    result = ru_ocr.ocr(russian_image)
    print(f"✅ Русский OCR выполнен, результат: {result}")
    
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            print(f"  📝 Найден текст: '{text}' (уверенность: {confidence:.2f})")
    else:
        print("⚠️ Русский текст не найден")
        
except Exception as e:
    print(f"❌ Ошибка русского OCR: {e}")
    print(f"💡 Тип ошибки: {type(e).__name__}")

print("\n🏁 Диагностика завершена")

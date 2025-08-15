"""
Упрощенный запуск сервера для тестирования
"""

import uvicorn
import sys
import os

print("Проверяем простой сервер...")

# Добавляем текущую папку в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from test_server import app
    print("✅ Тестовый сервер импортирован успешно")
    
    print("Запускаем тестовый сервер на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
except Exception as e:
    print(f"❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()


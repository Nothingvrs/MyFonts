"""
Стабильный запуск сервера для разработки
"""

import uvicorn
import sys
import os

if __name__ == "__main__":
    try:
        print("🚀 Запуск MyFonts сервера...")
        print("📂 Рабочая папка:", os.getcwd())
        print("🔄 Автоперезагрузка включена")
        print("🌐 Сервер: http://localhost:8000")
        print("📖 Документация: http://localhost:8000/docs")
        print("⚠️  Для остановки: Ctrl+C")
        print("-" * 50)
        
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",  # Локальный хост вместо 0.0.0.0
            port=8000,
            reload=True,
            reload_dirs=["app"],  # Отслеживаем только папку app
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка запуска сервера: {e}")
        sys.exit(1)

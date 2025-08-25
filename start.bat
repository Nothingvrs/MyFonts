@echo off
echo 🚀 Запуск MyFonts проекта...
echo.
echo 📱 Фронтенд будет доступен на: http://localhost:5173
echo 🔧 Бэкенд будет доступен на: http://localhost:8000
echo.
echo ⚠️  Для остановки нажмите Ctrl+C
echo.

REM Устанавливаем базовый URL backend для фронтенда по умолчанию
IF NOT DEFINED VITE_API_BASE_URL SET VITE_API_BASE_URL=http://localhost:8000

npm start

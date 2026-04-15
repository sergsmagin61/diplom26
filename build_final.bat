@echo off
chcp 65001 >nul
echo.
echo ========================================
echo   Crayfish AI Studio Pro - СБОРКА
echo ========================================
echo.

REM Проверка наличия иконки
if not exist icon.ico (
    echo [ВНИМАНИЕ] Файл icon.ico не найден! Иконка не будет добавлена.
    set "ICON_PARAM="
) else (
    echo [OK] Иконка найдена: icon.ico
    set "ICON_PARAM=--icon=icon.ico"
)
echo.

REM Очистка старых сборок
echo [1/4] Очистка старых файлов...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec 2>nul
echo [OK] Очистка завершена
echo.

REM Сборка приложения
echo [2/4] Запуск PyInstaller...
echo.

python -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "CrayfishAIStudio" ^
    %ICON_PARAM% ^
    --add-data "app.py;." ^
    --add-data "database.py;." ^
    --add-data "telegram_bot.py;." ^
    --add-data "models_manager.py;." ^
    --add-data "utils.py;." ^
    --add-data "config.py;." ^
    --hidden-import=tkinter ^
    --hidden-import=tkinter.ttk ^
    --hidden-import=PIL ^
    --hidden-import=PIL._tkinter_finder ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=pandas ^
    --hidden-import=matplotlib ^
    --hidden-import=matplotlib.backends.backend_tkagg ^
    --hidden-import=telebot ^
    --hidden-import=yaml ^
    --hidden-import=sqlite3 ^
    --hidden-import=json ^
    --hidden-import=threading ^
    --hidden-import=datetime ^
    --hidden-import=time ^
    --hidden-import=pathlib ^
    --hidden-import=collections ^
    --hidden-import=math ^
    --hidden-import=warnings ^
    --hidden-import=scipy ^
    --hidden-import=scipy.interpolate ^
    --hidden-import=scipy.stats ^
    --hidden-import=openpyxl ^
    --hidden-import=ultralytics ^
    --hidden-import=ultralytics.utils ^
    --hidden-import=torch ^
    --hidden-import=torchvision ^
    --collect-all=ultralytics ^
    --collect-all=cv2 ^
    --collect-all=PIL ^
    --collect-all=matplotlib ^
    --collect-all=numpy ^
    --collect-all=pandas ^
    --collect-all=scipy ^
    --collect-all=torch ^
    --collect-all=torchvision ^
    --exclude-module=IPython ^
    --exclude-module=jupyter ^
    --exclude-module=notebook ^
    --exclude-module=tensorflow ^
    --exclude-module=keras ^
    --exclude-module=PyQt5 ^
    --exclude-module=PyQt6 ^
    main.py

if %errorlevel% neq 0 (
    echo.
    echo [ОШИБКА] Сборка не удалась! Код ошибки: %errorlevel%
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo [3/4] Сборка завершена успешно!
echo.

REM Копирование дополнительных файлов (если нужно)
echo [4/4] Копирование дополнительных файлов...
if exist "dist\CrayfishAIStudio" (
    if exist "icon.ico" copy "icon.ico" "dist\CrayfishAIStudio\" >nul 2>nul
    echo [OK] Дополнительные файлы скопированы
) else (
    echo [ВНИМАНИЕ] Папка с собранным приложением не найдена
)

echo.
echo ========================================
echo   ГОТОВО!
echo   Приложение: dist\CrayfishAIStudio\CrayfishAIStudio.exe
echo ========================================
echo.
pause
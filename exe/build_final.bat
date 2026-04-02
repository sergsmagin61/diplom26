@echo off
chcp 65001 >nul
echo ========================================
echo    СБОРКА CRAYFISH AI STUDIO 64-bit
echo    (Режим с папкой _internal)
echo ========================================
echo.

REM Проверка наличия иконки
if not exist icon.ico (
    echo [ВНИМАНИЕ] Файл icon.ico не найден!
    set "ICON_PARAM="
) else (
    echo [OK] Иконка найдена: icon.ico
    set "ICON_PARAM=--icon=icon.ico"
    echo.
)

REM Удаляем старые сборки
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec

echo.
echo Запускаю сборку (64-bit, режим с папкой)...
echo Это займет 10-20 минут...
echo.

REM Важно: используем --onedir, а не --onefile
python -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "CrayfishAIStudio" ^
    %ICON_PARAM% ^
    --add-data "app.py;." ^
    --add-data "database.py;." ^
    --add-data "telegram_bot.py;." ^
    --add-data "models_manager.py;." ^
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
    echo [ОШИБКА] Сборка не удалась!
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo ========================================
echo    ✅ СБОРКА УСПЕШНО ЗАВЕРШЕНА!
echo ========================================
echo.
echo Файлы находятся в папке:
echo    %CD%\dist\CrayfishAIStudio\
echo.
echo Запускаемый файл:
echo    %CD%\dist\CrayfishAIStudio\CrayfishAIStudio.exe
echo.
echo Размер папки:
dir /s dist\CrayfishAIStudio

echo.
echo ========================================
echo    ИНСТРУКЦИЯ:
echo    1. Скопируйте всю папку CrayfishAIStudio
echo    2. Положите модели .pt в эту папку
echo    3. Запустите CrayfishAIStudio.exe
echo.
echo    ПРЕИМУЩЕСТВА:
echo    - Нет ограничения на размер
echo    - Быстрый запуск
echo    - Легче отлаживать
echo    - Меньше проблем с путями
echo ========================================
echo.
pause
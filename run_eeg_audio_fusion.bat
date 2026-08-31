@echo off
chcp 65001 >nul
echo ============================================
echo EEG音频融合分类 - 单人处理
echo ============================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python and add it to PATH.
    echo You can download Python from: https://www.python.org
    pause
    exit /b 1
)

echo Running EEG-Audio Fusion Classification...
echo.

REM 运行Python脚本
python eeg_audio_fusion_individual.py

echo.
echo ============================================
echo Processing completed!
echo ============================================
pause
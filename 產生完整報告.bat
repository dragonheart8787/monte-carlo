@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo 蒙地卡羅通訊效能評估系統 - 完整報告產生
echo ============================================================
echo.

python generate_full_report.py

echo.
pause

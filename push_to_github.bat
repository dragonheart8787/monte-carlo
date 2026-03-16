@echo off
chcp 65001 >nul
cd /d "%~dp0"

set /p GITHUB_USER=請輸入你的 GitHub 帳號 (例如: myusername): 
if "%GITHUB_USER%"=="" (
    echo 未輸入帳號，結束。
    pause
    exit /b 1
)

set REPO_NAME=monte-carlo-comm
set REPO_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git

echo.
echo 將推送到: %REPO_URL%
echo.
echo 請先到 https://github.com/new 建立儲存庫 "%REPO_NAME%"
echo 建立時請勿勾選 Add README
echo.
pause

git remote remove origin 2>nul
git remote add origin %REPO_URL%
git push -u origin main

echo.
pause

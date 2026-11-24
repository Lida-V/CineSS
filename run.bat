@echo off
chcp 65001 > nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [エラー] .venv がありません。先に setup.bat を実行してください。
  pause
  exit /b 1
)

set "PORT=%CINESS_PORT%"
if "%PORT%"=="" set "PORT=8780"

rem 既に起動済みならブラウザを開くだけ
netstat -an | findstr /c:"127.0.0.1:%PORT%" | findstr /c:"LISTENING" > nul
if not errorlevel 1 (
  echo 既に CineSS が起動しています。ブラウザを開きます...
  start "" http://127.0.0.1:%PORT%
  exit /b 0
)

echo ============================================
echo   CineSS を起動しています...
echo   URL: http://127.0.0.1:%PORT%
echo   準備ができたら自動でブラウザが開きます
echo   このウィンドウは閉じないでください
echo ============================================

rem サーバーが応答したらブラウザを開く監視をバックグラウンドで開始
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 240;$i++){try{$c=New-Object Net.Sockets.TcpClient('127.0.0.1',%PORT%);$c.Close();Start-Process 'http://127.0.0.1:%PORT%';exit}catch{Start-Sleep -Milliseconds 500}}"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -X utf8 -m uvicorn app.server:app --host 127.0.0.1 --port %PORT%
echo.
echo サーバーが終了しました。エラーが表示されている場合は内容を確認してください。
pause

@echo off
REM Start local review server for the elevator human visual review task.
REM Opens reviewer.html in the default browser at http://127.0.0.1:8765/
setlocal
set "REVIEW_DIR=%~dp0..\..\outputs\elevator_review"
cd /d "%REVIEW_DIR%"
echo Serving %REVIEW_DIR% at http://127.0.0.1:8765/reviewer.html
start "elevator-review-server" /min python -m http.server 8765 --bind 127.0.0.1
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:8765/reviewer.html"
endlocal
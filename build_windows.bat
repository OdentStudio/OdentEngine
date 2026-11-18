@echo off
rem odent_engine for Windows: ITK, VTK and the engine, then the smoke test.
rem See BUILD_WINDOWS.md. Arguments go to build_windows.ps1 (e.g. -Deps D:\odent_deps).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
if errorlevel 1 (echo. & echo The build failed: see the messages above.) else (echo. & echo The build succeeded.)
pause

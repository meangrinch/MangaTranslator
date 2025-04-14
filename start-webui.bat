@echo off

echo Launching MangaTranslator...
venv\scripts\python.exe app.py %* --open-browser
pause
@echo off

echo Launching Manga Translator...
venv\scripts\python.exe app.py %* --open-browser
pause
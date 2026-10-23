@echo off
title Benchmark Metalinguistico y Estilometria LLM
cd /d "%~dp0"

set "PYTHON_EXE=python"
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)

echo ============================================================================
echo   BENCHMARK EXPERIMENTAL: EVALUACION METALINGUISTICA Y ESTILOMETRIA - LLMs
echo ============================================================================
echo.

echo [1/3] Verificando dependencias de Python...
"%PYTHON_EXE%" -c "import streamlit, pandas, spacy" >nul 2>nul
if errorlevel 1 (
    echo       Instalando dependencias desde requirements.txt...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
)

echo [2/3] Verificando modelo linguistico es_core_news_lg de spaCy...
"%PYTHON_EXE%" -c "import spacy; spacy.load('es_core_news_lg')" >nul 2>nul
if errorlevel 1 (
    echo       Descargando modelo es_core_news_lg de spaCy por primera vez...
    "%PYTHON_EXE%" -m spacy download es_core_news_lg
)

echo [3/3] Iniciando interfaz grafica en Streamlit...
echo       La aplicacion se abrira automaticamente en tu navegador.
echo.
"%PYTHON_EXE%" -m streamlit run app.py
pause

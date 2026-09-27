@echo off
title ChurnGuard AI - Customer Churn Prediction System
echo =====================================================================
echo           Launching ChurnGuard AI Retention Platform
echo =====================================================================
echo.
echo Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.9+.
    pause
    exit /b 1
)

echo Starting Streamlit Web Application on default browser...
echo Local URL: http://localhost:8501
echo.
streamlit run app.py
pause

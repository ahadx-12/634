@echo off
setlocal
cd /d %~dp0

if not exist .venv (
  python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

pytest
if errorlevel 1 (
  echo Tests failed. Fix before running validation.
  exit /b 1
)

python run_validation.py
endlocal

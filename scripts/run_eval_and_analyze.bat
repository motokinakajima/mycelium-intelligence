@echo off
setlocal enabledelayedexpansion

set "TRIALS=%~1"
if "%TRIALS%"=="" set "TRIALS=512"

set "STEPS=%~2"
if "%STEPS%"=="" set "STEPS=1200"

set "COLS=%~3"
if "%COLS%"=="" set "COLS=5"

set "ROWS=%~4"
if "%ROWS%"=="" set "ROWS=5"

set "SEED_START=%~5"
if "%SEED_START%"=="" set "SEED_START=0"

set "THREADS=%~6"
if "%THREADS%"=="" set "THREADS=24"

set "BUILD_DIR=%~7"
if "%BUILD_DIR%"=="" set "BUILD_DIR=cmake-build-debug"

set "OUTPUT_CSV_NAME=%~8"
if "%OUTPUT_CSV_NAME%"=="" set "OUTPUT_CSV_NAME=seed_step_metrics_512.csv"

pushd "%~dp0\.." >nul
if errorlevel 1 (
  echo Failed to move to repository root.
  exit /b 1
)

echo [1/3] Building eval_seed_trials in %BUILD_DIR% ...
cmake --build ".\%BUILD_DIR%" --target eval_seed_trials -j
if errorlevel 1 goto :fail

set "EXE_PATH=.\%BUILD_DIR%\eval_seed_trials.exe"
set "OUTPUT_CSV_PATH=.\%BUILD_DIR%\%OUTPUT_CSV_NAME%"

echo [2/3] Running evaluator (trials=%TRIALS%, steps=%STEPS%, threads=%THREADS%) ...
"%EXE_PATH%" ".\hyperparameters.txt" "%OUTPUT_CSV_PATH%" %TRIALS% %STEPS% %COLS% %ROWS% %SEED_START% 0 %THREADS%
if errorlevel 1 goto :fail

echo [3/3] Running Python analysis and overwriting results/connected_metrics ...
python ".\results\analyze_connected_metrics.py" "%OUTPUT_CSV_PATH%" --out-dir ".\results\connected_metrics"
if errorlevel 1 goto :fail

echo Done.
echo - CSV: %OUTPUT_CSV_PATH%
echo - Metrics: results\connected_metrics\metrics_overview.json
popd >nul
exit /b 0

:fail
echo Failed with exit code %errorlevel%.
popd >nul
exit /b %errorlevel%

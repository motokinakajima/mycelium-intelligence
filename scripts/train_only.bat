@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\.." >nul

set "EPOCHS=%~1"
if "%EPOCHS%"=="" set "EPOCHS=400"

set "MODEL=%~2"
if "%MODEL%"=="" set "MODEL=node_nn_model.nn"

set "TEST_RATIO=%~3"
if "%TEST_RATIO%"=="" set "TEST_RATIO=0.2"

set "CSV=%~4"
if "%CSV%"=="" set "CSV=heuristics\training_data.csv"

if not exist "%CSV%" (
    echo [ERROR] CSV not found: %CSV%
    echo Usage: bat\train_only.bat [epochs] [model] [test_ratio] [csv]
    popd >nul
    exit /b 1
)

echo [INFO] Building train_nn target...
cmake --build cmake-build-debug --target train_nn
if errorlevel 1 goto :error

echo [INFO] Training model...
echo        CSV=%CSV%
echo        EPOCHS=%EPOCHS%
echo        MODEL=%MODEL%
echo        TEST_RATIO=%TEST_RATIO%

.\cmake-build-debug\train_nn.exe "%CSV%" %EPOCHS% "%MODEL%" %TEST_RATIO%
if errorlevel 1 goto :error

echo [OK] Training finished. Model saved to %MODEL%
popd >nul
exit /b 0

:error
echo [ERROR] Training failed.
popd >nul
exit /b 1

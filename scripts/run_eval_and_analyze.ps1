param(
    [int]$Trials = 512,
    [int]$Steps = 1200,
    [int]$Cols = 5,
    [int]$Rows = 5,
    [uint32]$SeedStart = 0,
    [int]$Threads = 24,
    [string]$BuildDir = "cmake-build-debug",
    [string]$OutputCsvName = "seed_step_metrics_512.csv"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot

try {
    Write-Host "[1/3] Building eval_seed_trials in $BuildDir ..."
    cmake --build ".\$BuildDir" --target eval_seed_trials -j
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }

    $exePath = Join-Path $BuildDir "eval_seed_trials.exe"
    $outputCsvPath = Join-Path $BuildDir $OutputCsvName

    Write-Host "[2/3] Running evaluator (trials=$Trials, steps=$Steps, threads=$Threads) ..."
    & $exePath ".\hyperparameters.txt" $outputCsvPath $Trials $Steps $Cols $Rows $SeedStart 0 $Threads
    if ($LASTEXITCODE -ne 0) {
        throw "eval_seed_trials failed with exit code $LASTEXITCODE"
    }

    Write-Host "[3/3] Running Python analysis and overwriting results/connected_metrics ..."
    python ".\results\analyze_connected_metrics.py" $outputCsvPath --out-dir ".\results\connected_metrics"
    if ($LASTEXITCODE -ne 0) {
        throw "analyze_connected_metrics.py failed with exit code $LASTEXITCODE"
    }

    Write-Host "Done."
    Write-Host "- CSV: $outputCsvPath"
    Write-Host "- Metrics: results\\connected_metrics\\metrics_overview.json"
}
finally {
    Pop-Location
}

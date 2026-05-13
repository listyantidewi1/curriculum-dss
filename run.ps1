# run.ps1 — wrapper for the v2 D-side venv.
#
# Usage:
#   .\run.ps1 scripts\run_full_v2_pipeline.py --tag latest
#   .\run.ps1 -m streamlit run dashboard_v2\app.py
#   .\run.ps1 -m pip install <pkg>
#
# Sets HF_HOME to D:\hf_cache if not already set, and dispatches everything
# through D:\Projects\skill-extraction\.venv-ml\Scripts\python.exe so no
# package or model spills onto C: again.

$ErrorActionPreference = "Stop"
$venvPy = Join-Path $PSScriptRoot ".venv-ml\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Error "venv not found at $venvPy. See PYTHON_ENV.md to create it."
    exit 1
}

# Belt-and-braces: ensure HF cache stays off C even if the User env var is missing
if (-not $env:HF_HOME) {
    $env:HF_HOME = "D:\hf_cache"
}

& $venvPy @args
exit $LASTEXITCODE

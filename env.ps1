# Dot-source from the repo root:  . .\env.ps1
# Reuses the already-downloaded Hugging Face models; never fetches from the network.
$root = (Get-Location).Path
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PSScriptRoot "cache\huggingface" }
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$sub = "mechanism","cost","scheduling","audit","reporting" | ForEach-Object { Join-Path $root "experiments\$_" }
$env:PYTHONPATH = (@((Join-Path $root "src")) + $sub) -join ";"
$tcc = Join-Path (python -c "import site;print(site.getusersitepackages())") "triton\runtime\tcc\tcc.exe"
if (Test-Path $tcc) { $env:CC = $tcc }
Write-Host "HF_HOME=$env:HF_HOME (offline); PYTHONPATH set; CC=$env:CC"

# shinsen_sim セットアップ
# - Python(3.10+) が必要です
# - 可能なら Python Launcher (py) で作成します

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

function Fail($msg) {
  Write-Host "[ERROR] $msg" -ForegroundColor Red
  exit 1
}

# Python コマンド検出
$pyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  $pyCmd = "py -3"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
  $pyCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
  $pyCmd = "python3"
}

if (-not $pyCmd) {
  Fail "Python が見つかりません。Python 3.10+ をインストールしてから再実行してください。（インストール時は 'Add Python to PATH' 推奨）"
}

Write-Host "Using: $pyCmd" -ForegroundColor Cyan

# venv 作成（既存が壊れてる可能性があるので、無ければ作成、あればそのまま）
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Write-Host "Creating venv (.venv)..." -ForegroundColor Cyan
  & $pyCmd -m venv .venv
}

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Fail "venv の作成に失敗しました。.venv\\Scripts\\python.exe が見つかりません。Python のインストール状態を確認してください。"
}

# pip 更新 & 依存導入
Write-Host "Upgrading pip..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -U pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Setup complete." -ForegroundColor Green

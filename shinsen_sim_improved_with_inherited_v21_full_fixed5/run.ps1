# shinsen_sim 起動
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

function Fail($msg) {
  Write-Host "[ERROR] $msg" -ForegroundColor Red
  exit 1
}

# venv が無い/壊れてる場合はセットアップを促す
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  Fail "venv (.venv) が見つかりません。先に setup.ps1 を実行してください。"
}

# streamlit が無い場合は追加インストール
if (-not (Test-Path .\.venv\Scripts\streamlit.exe)) {
  Write-Host "streamlit が見つからないため、requirements を再インストールします..." -ForegroundColor Yellow
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

if (-not (Test-Path .\.venv\Scripts\streamlit.exe)) {
  Fail "streamlit.exe が見つかりません。setup.ps1 を再実行してください。"
}

Write-Host "Starting Streamlit..." -ForegroundColor Cyan
.\.venv\Scripts\streamlit.exe run app.py

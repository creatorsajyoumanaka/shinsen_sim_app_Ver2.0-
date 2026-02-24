# 信長真戦シミュレーター（改良版 / v2.1 修復版）

## 必要なもの
- **Python 3.10 以上**（おすすめは 3.11 / 3.12）
  - インストール時に **「Add Python to PATH」** にチェック推奨
  - `py` コマンド（Python Launcher）が入っているとより確実です

## 起動（PowerShell）
このフォルダで以下を実行します。

### 1) 初回セットアップ（1回だけ）
```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\setup.ps1
```

### 2) 起動（毎回）
```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run.ps1
```

---

## 固有戦法の追加/修正
- `unique_skills.json` に追記します。
- 形式は `id`（例: `UNQ_風林火山`）で紐づきます。`units.json` の `unique_skill_id` と同じ文字列にしてください。

### 自動生成し直したい場合
`units.json` を更新した後に、下記を実行：
```powershell
python .\tools\rebuild_unique_skills_from_units.py
```

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固有戦法一覧（テキスト貼り付け）から「戦法名→所持武将」を抽出し、
unique_skills.json の owner / id を上書きします。

使い方（PowerShell例）:
  python tools\fix_unique_owners_from_text.py --source unique_skills_source.txt

入力:
  - unique_skills_source.txt : ユーザーが貼ってくれた「固有戦法一覧」のテキスト
  - unique_skills.json       : 本体データ（同じフォルダにある前提）

出力:
  - unique_skills.json を更新（バックアップ unique_skills.json.bak も作成）
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAT = re.compile(
    r"(?P<skill>[^\n\r【】]{1,50})\s*\r?\n"
    r"【適性兵種】[\s\S]*?"
    r"【所持武将】\s*\r?\n"
    r"(?P<owner>.+?)\s+の固有戦法",
    re.MULTILINE,
)

def parse_mapping(text: str) -> dict[str, str]:
    m: dict[str, str] = {}
    for mm in PAT.finditer(text):
        skill = mm.group("skill").strip()
        owner = mm.group("owner").strip()
        # 先に出てきたもの優先（重複があれば後で手で直せる）
        m.setdefault(skill, owner)
    return m

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="固有戦法一覧のテキストファイルパス")
    ap.add_argument("--unique", default="unique_skills.json", help="unique_skills.json のパス（既定: unique_skills.json）")
    args = ap.parse_args()

    src_path = Path(args.source)
    uniq_path = Path(args.unique)

    if not src_path.exists():
        raise SystemExit(f"source not found: {src_path}")
    if not uniq_path.exists():
        raise SystemExit(f"unique_skills.json not found: {uniq_path}")

    text = src_path.read_text(encoding="utf-8", errors="ignore")
    mapping = parse_mapping(text)
    if not mapping:
        raise SystemExit("戦法名→所持武将 の抽出に失敗しました。テキストの形式を確認してください。")

    data = json.loads(uniq_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("unique_skills.json の形式が不正です（listではありません）")

    changed = 0
    missing = 0

    for s in data:
        name = (s or {}).get("name")
        if not name:
            continue
        owner = mapping.get(name)
        if owner:
            if s.get("owner") != owner:
                s["owner"] = owner
                s["id"] = f"UNQ_{owner}"
                changed += 1
        else:
            missing += 1

    # backup
    bak = uniq_path.with_suffix(uniq_path.suffix + ".bak")
    orig = uniq_path.read_text(encoding="utf-8")
    bak.write_text(orig, encoding="utf-8")
    uniq_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done. updated owners: {changed}, skills without mapping: {missing}")
    print(f"backup: {bak.name} （※バックアップにも同内容を書いています。必要なら元ファイルを先にコピーして下さい）")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

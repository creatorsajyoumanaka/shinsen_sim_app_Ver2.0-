# -*- coding: utf-8 -*-
"""
unique_skills_raw.txt（ユーザーがコピペした固有戦法一覧）から unique_skills.json を生成する簡易スクリプト。

使い方:
  1) unique_skills_raw.txt に、チャットで送った「戦法名 ... 【所持武将】 ...」の本文を丸ごと貼る
  2) venv有効化後に実行:
        python tools/import_unique_skills.py unique_skills_raw.txt unique_skills.json
"""
from __future__ import annotations
import re
import json
import sys
from pathlib import Path

def parse_blocks(text: str) -> list[dict]:
    # normalize
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    # Split by "戦法名" markers (keep name inside each block)
    parts = re.split(r"(?:^|\n)\s*戦法名\s+", t)
    skills = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # name: up to first bracket section
        m_name = re.search(r"^(.+?)\s+【", p)
        if not m_name:
            continue
        name = m_name.group(1).strip()
        # owner (optional)
        owner = ""
        m_owner = re.search(r"【所持武将】\s*([^\n【]+)", p)
        if m_owner:
            owner = m_owner.group(1).strip()
        raw = "戦法名 " + p  # restore marker for readability
        skills.append({
            "id": f"UNQ_{name}",
            "name": name,
            "owner": owner,
            "raw": raw.strip(),
        })
    # de-dup by name (last wins)
    dedup = {}
    for s in skills:
        dedup[s["name"]] = s
    return list(dedup.values())

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/import_unique_skills.py <unique_skills_raw.txt> <unique_skills.json>")
        sys.exit(1)
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    text = in_path.read_text(encoding="utf-8")
    skills = parse_blocks(text)
    out_path.write_text(json.dumps(skills, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(skills)} skills -> {out_path}")

if __name__ == "__main__":
    main()

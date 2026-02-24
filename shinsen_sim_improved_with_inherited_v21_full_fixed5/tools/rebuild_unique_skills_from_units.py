import json
from pathlib import Path

units = json.load(open("units.json", "r", encoding="utf-8"))
owners = {}
for u in units:
    usid = u.get("unique_skill_id")
    if usid and usid not in owners:
        owners[usid] = u.get("name", "")
skills = []
for usid, owner in sorted(owners.items()):
    name = usid[4:] if isinstance(usid, str) and usid.startswith("UNQ_") else usid
    skills.append({
        "id": usid,
        "name": name,
        "owner": owner,
        "activation_chance": 0,
        "effects_text": "（効果未入力：ここに追記してください）",
        "notes": "units.json の unique_skill_id から自動生成。効果は手動で追記可。"
    })
Path("unique_skills.json").write_text(json.dumps(skills, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"unique_skills.json regenerated: {len(skills)} skills")

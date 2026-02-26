
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
import streamlit as st
import engine
import pandas as pd
import streamlit.components.v1 as components

_LAST_KEY = "shinsen_sim:last_comp_v1"

def _ls_set(key: str, value: dict):
    payload = json.dumps(value, ensure_ascii=False)
    components.html(
        f"<script>localStorage.setItem({json.dumps(key)}, {json.dumps(payload)});</script>",
        height=0,
    )


def _ls_get(key: str):
    components.html(
        f"""
        <script>
        const v = localStorage.getItem({json.dumps(key)}) || "";
        const url = new URL(window.location.href);
        if (v) url.searchParams.set("ls_load", encodeURIComponent(v));
        else url.searchParams.delete("ls_load");
        window.history.replaceState(null, "", url.toString());
        </script>
        """,
        height=0,
    )

    qp = st.query_params
    v = qp.get("ls_load")
    if not v:
        return None
    if isinstance(v, list):
        v = v[0]

    try:
        import urllib.parse
        decoded = urllib.parse.unquote(v)
        return json.loads(decoded)
    except Exception:
        return None


def _build_comp_state() -> dict:
    """
    session_state から「編成に関係するキー」だけ拾って保存する（汎用版）
    """
    keep = {}
    for k, v in st.session_state.items():
        ks = str(k)
        # ally/enemy のユニット選択・伝授・覚醒っぽいキーだけ拾う
        if any(x in ks for x in ["ally", "enemy"]) and any(
            x in ks for x in ["unit", "name", "skill", "inh", "us_", "awake"]
        ):
            try:
                json.dumps(v, ensure_ascii=False)
                keep[ks] = v
            except Exception:
                pass

    # 全体設定も保存したいなら（存在するものだけ）
    for opt_key in ["seed", "TROOP_SCALE", "troop_scale"]:
        if opt_key in st.session_state:
            keep[opt_key] = st.session_state[opt_key]

    return keep


def _apply_comp_state(data: dict):
    if not isinstance(data, dict):
        return
    for k, v in data.items():
        st.session_state[k] = v
from engine import Unit, Skill, simulate_battle, extract_max_from_arrow, parse_probability_max

APP_TITLE = "信長真戦シミュレーター（Ver2.0.1）"

DATA_DIR = Path(__file__).parent
UNITS_PATH = DATA_DIR / "units.json"
UNIQUE_SKILLS_PATH = DATA_DIR / "unique_skills.json"
INHERITED_SKILLS_PATH = DATA_DIR / "inherited_skills.json"



def make_skill_from_raw(name, raw, kind="unknown", default_prob=35.0):
    prob = parse_probability_max(raw)
    if prob is None:
        prob = float(default_prob)
    return Skill(name=name, raw=raw or "", kind=kind, base_prob=float(prob), level=20, awaken=True)

def skill_display(entry):
    raw = entry.get("raw", "") or ""
    prob = entry.get("base_prob", None)
    if raw:
        pmax = parse_probability_max(raw)
        if pmax is not None:
            prob = pmax
    if prob is None:
        prob = "—"
    raw_max = extract_max_from_arrow(raw) if raw else ""
    return prob, raw_max

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def _normalize_unit(u: dict) -> dict:
    # units.json は base_stats に能力値が入っている想定（str/int/lea/spd）
    bs = u.get("base_stats") or {}
    # 表示・計算で使うキー（wu/int/lea/spd/max_soldiers）を保証
    if "wu" not in u:
        u["wu"] = bs.get("wu", bs.get("str", bs.get("武勇", 0)))
    if "int" not in u:
        u["int"] = bs.get("int", bs.get("知略", 0))
    if "lea" not in u:
        u["lea"] = bs.get("lea", bs.get("統率", 0))
    if "spd" not in u:
        u["spd"] = bs.get("spd", bs.get("速度", 0))
    if "max_soldiers" not in u:
        u["max_soldiers"] = u.get("soldiers", u.get("兵力", 0))

    # 固有戦法キーの揺れ対策（units.json 側が "UNQ_戦法名" 形式のことがある）
    us_key = u.get("unique_skill_id") or u.get("unique_skill") or ""
    if isinstance(us_key, str):
        if us_key.startswith("UNQ_"):
            u["unique_skill_name"] = us_key.split("_", 1)[1]
        else:
            u["unique_skill_name"] = us_key
    else:
        u["unique_skill_name"] = ""
    return u

def load_units():
    raw = json.loads(UNITS_PATH.read_text(encoding="utf-8"))
    return [_normalize_unit(u) for u in raw]
def load_unique_skill_list():
    """Load unique skills from json.
    Supports both the legacy list-of-dicts schema (expects 'name'/'raw')
    and the converted schema (expects 'id'/'name'/'owner'/'raw_max')."""
    data = json.loads(UNIQUE_SKILLS_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # allow dict-of-skills too
        data = list(data.values())
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        sid = item.get("id") or item.get("skill_id") or item.get("name")
        name = item.get("name") or sid or "UNKNOWN"
        owner = item.get("owner") or item.get("holder") or item.get("character")
        kind = item.get("kind") or ""
        target = item.get("target") or ""
        raw = item.get("raw_max") or item.get("raw") or item.get("raw_text") or ""
        # allow providing already-parsed prob
        prob = item.get("prob_max") if item.get("prob_max") is not None else item.get("prob")
        prob = float(prob) if isinstance(prob, (int, float)) else None

        # NOTE: unique_skills.json may contain a separate "target" field for UI.
        # The battle engine parses behavior from the free-form "effect" text (raw),
        # so we don't pass target into Skill (engine.Skill has no target attribute).
        s = make_skill_from_raw(name=name, kind=kind, raw=raw)
        if sid:
            s["id"] = sid
        if owner:
            s["owner"] = owner
        if prob is not None:
            s["base_prob"] = prob / 100.0 if prob > 1.0 else prob  # accept 0-1 or %
        out.append(s)
    return out


def load_unique_skills():
    skills = load_unique_skill_list()
    m = {}
    for s in skills:
        name = s.get("name")
        if name:
            m[name] = s
        sid = s.get("id")
        if sid:
            m[sid] = s
    return m

@st.cache_data
def load_inherited_skills():
    if INHERITED_SKILLS_PATH.exists():
        return json.loads(INHERITED_SKILLS_PATH.read_text(encoding="utf-8"))
    return []

unique_skill_list = load_unique_skill_list()
unique_skill_map = load_unique_skills()
units = load_units()

# Attach correct unique skill ids by owner (units.json may be missing or wrong)
_owner_to_us = {s.get('owner'): s for s in unique_skill_list if s.get('owner')}
for u in units:
    nm = u.get('name')
    if nm and nm in _owner_to_us:
        us = _owner_to_us[nm]
        u['unique_skill_id'] = us.get('id') or us.get('name')
        u['unique_skill_name'] = us.get('name')
base_inherited = load_inherited_skills()

# session custom skills
if "custom_inherited" not in st.session_state:
    st.session_state.custom_inherited = []
if "custom_unique" not in st.session_state:
    st.session_state.custom_unique = []


def merged_inherited():
    # merge by name (custom overrides base if same name)
    merged = {s["name"]: s for s in base_inherited}
    for s in st.session_state.custom_inherited:
        merged[s["name"]] = s
    return list(merged.values())

# -----------------------------
# Helpers
# -----------------------------
def merged_unique_skill_map(base_map, units_list):
    """Merge base unique skill map with session custom skills and placeholders from units."""
    merged = dict(base_map) if base_map else {}
    # apply custom unique skills (override by name)
    for s in st.session_state.get("custom_unique", []):
        nm = (s.get("name") or "").strip()
        if not nm:
            continue
        merged[nm] = {
            "name": nm,
            "owner": (s.get("owner") or "").strip(),
            "raw": (s.get("raw") or "").strip(),
        }
    # placeholders so UI never shows 'データなし' just because details are missing
    for u in units_list or []:
        us_name = (u.get("unique_skill") or "").strip()
        if us_name and us_name not in merged:
            merged[us_name] = {"name": us_name, "raw": "", "owner": u.get("name", "")}
    return merged
unique_skill_map = merged_unique_skill_map(unique_skill_map, units)


def unit_label(u):
    return f'{u["name"]}｜武{u["wu"]} 知{u["int"]} 統{u["lea"]} 速{u["spd"]} 兵{u["max_soldiers"]}'

def normalize_unit(raw: dict) -> dict:
    """Normalize a unit record from units.json into the flat keys used by the UI/engine."""
    u = dict(raw) if raw else {}
    bs = u.get('base_stats') or {}
    # Accept both 'new' flat keys and the units.json base_stats schema.
    def pick(*keys, default=0):
        for k in keys:
            if k in u and u[k] is not None:
                return u[k]
            if k in bs and bs[k] is not None:
                return bs[k]
        return default
    u['wu'] = int(pick('wu', 'str', default=0))
    u['int'] = int(pick('int', 'intel', default=0))
    u['lea'] = int(pick('lea', 'cmd', default=0))
    u['spd'] = int(pick('spd', default=0))
    u['max_soldiers'] = int(pick('max_soldiers', 'troops', default=0))
    # Keep a stable unique-skill reference
    if 'unique_skill_id' in u and isinstance(u['unique_skill_id'], str):
        u['unique_skill_id'] = u['unique_skill_id'].strip()
    return u

def get_unit_by_name(name):
    for u in units:
        if u["name"] == name:
            return normalize_unit(u)
    return None



def style_log(df: pd.DataFrame):
    """ログ表の見た目（色付け）を安全に適用する。

    途中で列名を日本語にリネームするため、英語/日本語どちらの列名でも動くようにしている。
    """
    if df is None or len(df) == 0:
        return df.style

    # どの列名で来ても対応
    side_col = "side" if "side" in df.columns else ("陣営" if "陣営" in df.columns else None)
    action_col = "action_name" if "action_name" in df.columns else ("行動" if "行動" in df.columns else None)

    def _color_side(v):
        # 自軍:青 / 敵軍:赤 / その他:黒
        s = str(v)
        if s in ("ally", "自軍"):
            return "color: #1f77b4;"
        if s in ("enemy", "敵軍"):
            return "color: #d62728;"
        return ""

    def _color_action(v):
        # 通常攻撃（黒）／戦法発動（黄）／不発（灰）
        s = str(v)
        if s in ("通常攻撃", "Normal"):
            return ""
        if s in ("戦法発動", "Skill"):
            return "color: #b58900; font-weight: 700;"
        if s in ("不発", "Miss"):
            return "color: #777;"
        return ""

    sty = df.style
    if side_col is not None:
        sty = sty.map(_color_side, subset=[side_col])
    if action_col is not None:
        sty = sty.map(_color_action, subset=[action_col])
    return sty

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("データ追加（任意）")

    st.markdown("### 編成の保存（このPC/ブラウザ）")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("💾保存", key="save_comp"):
            _ls_set(_LAST_KEY, _build_comp_state())
            st.success("保存しました")

    with c2:
        if st.button("↩復元", key="load_comp"):
            loaded = _ls_get(_LAST_KEY)
            if loaded:
                _apply_comp_state(loaded)
                st.rerun()
            else:
                st.warning("保存がありません")

    with c3:
        if st.button("🧹削除", key="clear_comp"):
            _ls_set(_LAST_KEY, {})
            st.success("削除しました")

    st.divider()

    st.caption("伝授戦法の効果テキスト（raw）が未登録のものは、発動ログのみ出して効果は反映しません。ここで追加すると反映できます。")
    with st.expander("伝授戦法（custom）を追加"):
        cname = st.text_input("戦法名", key="custom_skill_name")
        cprob = st.number_input("発動確率（%）", min_value=0.0, max_value=100.0, value=35.0, step=1.0, key="custom_skill_prob")
        ckind = st.selectbox("種別", ["unknown", "active", "charge", "command", "passive", "troop"], index=0, key="custom_skill_kind")
        craw = st.text_area("raw（ゲーム内説明を貼り付け）", height=180, key="custom_skill_raw")
        if st.button("追加/上書き"):
            if cname.strip():
                st.session_state.custom_inherited.append({
                    "name": cname.strip(),
                    "kind": ckind,
                    "base_prob": float(cprob),
                    "raw": craw.strip(),
                })
                st.success("追加しました（同名は上書きされます）")
            else:
                st.warning("戦法名を入力してください。")

    st.divider()
    seed = st.number_input("乱数シード（同じなら再現）", min_value=0, max_value=999999, value=42, step=1)
    max_turns = st.slider("最大ターン数", min_value=1, max_value=8, value=8, step=1)
troop_scale = st.slider("ダメージスケール(TROOP_SCALE)", min_value=1000, max_value=50000, value=10000, step=1000)
engine.TROOP_SCALE = int(troop_scale)


st.subheader("編成")

colA, colB = st.columns(2)

# 表示する武将は units.json の内容に限定（追加で「固有戦法データありのみ」フィルタ可能）
only_with_unique = st.sidebar.checkbox("固有戦法データがある武将のみ表示", value=False)

if only_with_unique:
    def _has_unique_data(u: dict) -> bool:
        us = (u.get("unique_skill_id") or u.get("unique_skill") or "").strip()
        if us.startswith("UNQ_"):
            us = us[4:]
        return bool(us) and (us in unique_skill_map or f"UNQ_{us}" in unique_skill_map)
    unit_names = [u["name"] for u in units if _has_unique_data(u)]
else:
    unit_names = [u["name"] for u in units]

# keep original order...uniqueness list
unit_options = list(dict.fromkeys(unit_names))

with colA:
    st.markdown("### 自軍（青）")
    ally_sel = st.multiselect("武将を3名選択", options=unit_options, default=unit_options[:3], max_selections=3, key="ally_sel")
with colB:
    st.markdown("### 敵軍（赤）")
    enemy_sel = st.multiselect("武将を3名選択", options=unit_options, default=unit_options[3:6] if len(unit_options) >= 6 else unit_options[:3], max_selections=3, key="enemy_sel")

if len(ally_sel) != 3 or len(enemy_sel) != 3:
    st.info("自軍3名・敵軍3名を選択すると、下に設定とシミュレーションが表示されます。")
    st.stop()

inherited_db = merged_inherited()
inherited_names = [s["name"] for s in inherited_db]
# Search keyword (global)
st.markdown("### 伝授戦法検索")
search_kw = st.text_input("戦法名で検索（例：回天）", value="", key="search_kw")
filtered_inherited = [s for s in inherited_db if search_kw.strip() in s["name"]]
filtered_names = [s["name"] for s in filtered_inherited]

st.caption(f"候補数: {len(filtered_names)}（全{len(inherited_names)}）")

def render_unit_panel(side: str, name: str, idx: int):
    u = get_unit_by_name(name)
    if not u:
        st.error(f"{name} のデータが見つかりません")
        return None, []

    # units.json 側に固有戦法ID/名称が入っている場合に備えて先に取り出す
    # （例: "UNQ_XXXX" / "豊臣秀吉" など、キー揺れがあり得るため）
    us_name = (u.get("unique_skill_id") or u.get("unique_skill") or "").strip() or None
    us_name_no_prefix = None
    if us_name and isinstance(us_name, str) and us_name.startswith("UNQ_"):
        us_name_no_prefix = us_name.replace("UNQ_", "", 1)

    # Unique skill (owned by this unit name)
    us_entry = (
        unique_skill_map.get(name)
        or (unique_skill_map.get(us_name) if us_name else None)
        or (unique_skill_map.get(us_name_no_prefix) if us_name_no_prefix else None)
        or (unique_skill_map.get(f"UNQ_{us_name}") if us_name and not us_name.startswith("UNQ_") else None)
    )
    if us_entry:
        us_name = us_entry.get("name", us_name)
        us_raw = us_entry.get("raw", "")
        us_kind = us_entry.get("kind") or us_entry.get("type") or "unique"
        us_prob = parse_probability_max(us_raw, default=35.0)
        with st.expander(f"固有戦法（シミュ反映）: {us_name}"):
            st.markdown(f"- 種別: `{us_kind}` / 発動確率（最大）: **{us_prob:.0f}%**")
            st.text(us_raw)
        unique_skill_obj = make_skill_from_raw(name=us_name, raw=us_raw, kind=us_kind)
    else:
        unique_skill_obj = None
        with st.expander("固有戦法: なし/不明"):
            st.caption("この武将の固有戦法が unique_skills.json に見つかりません。")

    st.markdown("**伝授戦法（入力可 / 最大Lv10固定 & 覚醒）**")
    inh_kw = st.text_input("伝授戦法検索（この武将）", value="", placeholder="例：回天 / 火計 / 無策", key=f"{side}_{idx}_inh_kw")
    local_names = filtered_names
    if inh_kw.strip():
        kw = inh_kw.strip()
        local_names = [n for n in filtered_names if kw in n]
    if not local_names:
        st.caption("該当なし → 全件表示に戻しています")
        local_names = filtered_names

    c1, c2 = st.columns(2)
    with c1:
        sk1 = st.selectbox(f"伝授1（{name}）", options=["—"] + local_names, key=f"{side}_{idx}_sk1")
        lv1 = 10  # 伝授戦法Lvは最大10固定（真戦仕様）
        aw1 = st.checkbox("覚醒", value=True, key=f"{side}_{idx}_aw1")
    with c2:
        sk2 = st.selectbox(f"伝授2（{name}）", options=["—"] + local_names, key=f"{side}_{idx}_sk2")
        lv2 = 10  # 伝授戦法Lvは最大10固定（真戦仕様）
        aw2 = st.checkbox("覚醒 ", value=True, key=f"{side}_{idx}_aw2")

    chosen = []
    for nm, lv, aw in [(sk1, lv1, aw1), (sk2, lv2, aw2)]:
        if nm != "—":
            entry = next((s for s in inherited_db if s["name"] == nm), None)
            if entry is None:
                continue
            prob, raw_max = skill_display(entry)
            st.caption(f"{nm}｜発動確率: {prob}％｜効果: {'登録あり' if (entry.get('raw') or '').strip() else '未登録'}")
            chosen.append((entry, lv, aw))

    unit_obj = Unit(
        name=name,
        side=side,
        wu=int(u["wu"]),
        int_=int(u["int"]),
        lea=int(u["lea"]),
        spd=int(u["spd"]),
        max_soldiers=int(u["max_soldiers"]),
        soldiers=int(u["max_soldiers"]),
        unique_skill=unique_skill_obj,
        inherited=[]
    )

    inh_skills = []
    for entry, lv, aw in chosen:
        inh_skills.append(Skill(
            name=entry["name"],
            raw=entry.get("raw", "") or "",
            kind=entry.get("kind", "unknown"),
            base_prob=float(entry.get("base_prob", 35.0)),
            level=int(lv),
            awaken=bool(aw),
        ))
    unit_obj.inherited = inh_skills
    return unit_obj, chosen

col1, col2 = st.columns(2)

allies_units = []
enemies_units = []

with col1:
    st.markdown("## 自軍の詳細設定")
    for i, nm in enumerate(ally_sel):
        uo, _ = render_unit_panel("ally", nm, i)
        if uo:
            allies_units.append(uo)

with col2:
    st.markdown("## 敵軍の詳細設定")
    for i, nm in enumerate(enemy_sel):
        uo, _ = render_unit_panel("enemy", nm, i)
        if uo:
            enemies_units.append(uo)

st.divider()

st.subheader("シミュレーション")

if st.button("シミュ実行", type="primary"):
    logs, summary = simulate_battle(allies_units, enemies_units, turns=max_turns, seed=int(seed))

    st.markdown("### 結果サマリー")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("自軍 生存", summary["ally_alive"])
    c2.metric("敵軍 生存", summary["enemy_alive"])
    c3.metric("自軍 残兵力", summary["ally_soldiers"])
    c4.metric("敵軍 残兵力", summary["enemy_soldiers"])

    st.markdown("### 毎ターンログ（最大8ターン）")
    df = pd.DataFrame([
        {
            "turn": r.turn,
            "order": r.order,
            "side": r.side,
            "unit": r.unit,
            "action_type": r.action_type,
            "action_name": r.action_name,
            "detail": r.detail,
            "actor_hp": r.actor_hp,
            "target_hp": r.target_hp,
        }
        for r in logs
    ])

    if df.empty:
        st.info("ログがありません（全滅またはデータ不足の可能性）")
    else:
        # order per turn quick view
        with st.expander("行動順（ターンごと）"):
            for t in sorted(df["turn"].unique()):
                sub = df[df["turn"] == t].sort_values(["order"])
                order_txt = []
                for _, row in sub.iterrows():
                    color = "#1f77b4" if row["side"] == "ally" else "#d62728"
                    order_txt.append(f"<span style='color:{color}'>{row['unit']}</span>")
                st.markdown(f"**Turn {t}**: " + " → ".join(order_txt), unsafe_allow_html=True)

        # styled table
        show = df[["turn", "order", "side", "unit", "action_name", "detail", "actor_hp", "target_hp"]].copy()
        show.rename(
            columns={
                "turn": "ターン",
                "order": "順番",
                "side": "陣営",
                "unit": "行動者",
                "action_name": "行動",
                "detail": "詳細",
                "actor_hp": "行動者 残兵",
                "target_hp": "対象 残兵",
            },
            inplace=True,
        )
        show["陣営"] = show["陣営"].map({"ally": "自軍", "enemy": "敵軍"}).fillna(show["陣営"])
        st.dataframe(style_log(show), use_container_width=True, height=520)

    st.markdown("### 注意")
    st.write("- 固有戦法はシミュレーションに反映されます（※発動判定あり）")
    st.write("- 伝授戦法は raw が未登録だと **発動ログのみ** になります。必要な戦法から順に raw を追加すれば、効果も反映されます。")

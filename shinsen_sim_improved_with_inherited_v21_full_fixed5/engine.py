
# -*- coding: utf-8 -*-
# FIXED_ENGINE_2026-02-28 (tabs->spaces, passive/command handled, seal_attack & double_attack)
"""
Lightweight battle simulator engine for Nobunaga Shinsen-style tactics.

Design goals:
- Never crash on unknown tactics (treat as no-op, still log).
- Data-driven skill parsing (best-effort) from the "raw" text found in unique_skills/custom skills.
- Deterministic when seed is fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import random
import re

# Damage scaling: divide base damage by this value to keep numbers in a sensible range.
TROOP_SCALE = 10000  # damage scaling; adjust from app sidebar.

# -----------------------------
# Utility: parsing helpers
# -----------------------------

_ARROW_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*→\s*(\d+(?:\.\d+)?)\s*%?")
_RATE_RE = re.compile(r"(?:ダメージ率|回復率)\s*(\d+(?:\.\d+)?)\s*%")
_PROB_RE = re.compile(r"(?:発動確率)\s*(\d+(?:\.\d+)?)\s*%")

def _interp(min_v: float, max_v: float, level: int) -> float:
    """Linear interpolation across levels 1..20."""
    level = max(1, min(20, int(level)))
    t = (level - 1) / 19.0
    return min_v + (max_v - min_v) * t

def extract_max_from_arrow(text: str) -> str:
    """Replace 'a→b' with 'b' for display purposes."""
    def repl(m: re.Match) -> str:
        return m.group(2)
    return _ARROW_RE.sub(repl, text or "")

def parse_probability_max(raw: str, default: Optional[float] = None) -> Optional[float]:
    """Parse activation probability (max, if arrow exists). Returns percent (0-100)."""
    try:
        if raw is None:
            return default
        m = _PROB_RE.search(raw)
        if not m:
            return default
        vicinity = raw[m.start(): m.start() + 60]
        am = _ARROW_RE.search(vicinity)
        if am:
            return float(am.group(2))
        return float(m.group(1))
    except Exception:
        return default

def parse_first_rate(raw: str, key: str = "ダメージ率", level: int = 20, awaken: bool = True) -> Optional[float]:
    """
    Parse first (damage/heal) rate found in raw.
    If arrow exists near it, interpolate by level (or max if awaken).
    Returns decimal multiplier, e.g. 104% -> 1.04
    """
    if not raw:
        return None
    m = _RATE_RE.search(raw)
    if not m:
        return None
    start = max(0, m.start() - 30)
    end = min(len(raw), m.end() + 30)
    vicinity = raw[start:end]
    am = _ARROW_RE.search(vicinity)
    if am:
        if awaken:
            v = float(am.group(2))
        else:
            v = _interp(float(am.group(1)), float(am.group(2)), level)
        return v / 100.0
    return float(m.group(1)) / 100.0

def detect_damage_type(raw: str) -> Optional[str]:
    if not raw:
        return None
    if "計略ダメージ" in raw:
        return "strategy"
    if "兵刃ダメージ" in raw:
        return "physical"
    if "ダメージタイプは武勇と知略" in raw:
        return "hybrid"
    return None

def detect_targets(raw: str) -> str:
    """Best-effort target detector."""
    if not raw:
        return "enemy_single"
    if "敵軍全体" in raw:
        return "enemy_all"
    if "敵軍複数" in raw:
        return "enemy_multi"
    if "敵軍単体" in raw or "敵軍大将" in raw:
        return "enemy_single"
    if "自軍全体" in raw:
        return "ally_all"
    if "自軍複数" in raw:
        return "ally_multi"
    if "自軍単体" in raw or "友軍単体" in raw:
        return "ally_single"
    if "自分" in raw or "自身" in raw:
        return "self"
    return "enemy_single"

# Status keywords (best-effort) for ON-HIT apply in damage skills.
# (Passive/command effects like 連撃 or 気炎万丈 are handled separately.)
STATUS_KEYWORDS: Dict[str, str] = {
    "威圧": "stun",
    "麻痺": "paralyze",
    "混乱": "confuse",
    "無策": "silence",
    "封撃": "disarm",
    "火傷": "burn",
    "水攻め": "flood",
    "潰走": "collapse",
    "消沈": "depress",
}

def detect_statuses(raw: str) -> List[str]:
    found: List[str] = []
    if not raw:
        return found
    for jp, code in STATUS_KEYWORDS.items():
        if jp in raw:
            found.append(code)
    return found

def _parse_kien_params(raw: str) -> Tuple[float, float, int]:
    """
    気炎万丈など:
      毎ターン70%の確率で通常攻撃不可（毎ターン発動確率が14%減少）
    Returns (p, decay, turns).
    """
    p = 0.70
    decay = 0.14
    turns = 3

    m = re.search(r"毎ターン\s*(\d+(?:\.\d+)?)\s*%.*通常攻撃不可", raw)
    if m:
        try:
            p = float(m.group(1)) / 100.0
        except Exception:
            pass
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*%.*減少", raw)
    if m2:
        try:
            decay = float(m2.group(1)) / 100.0
        except Exception:
            pass
    # "3ターン目まで" etc.
    m3 = re.search(r"(\d+)\s*ターン目まで", raw)
    if m3:
        try:
            turns = int(m3.group(1))
        except Exception:
            pass
    return p, decay, turns

# -----------------------------
# Core data structures
# -----------------------------

@dataclass
class Skill:
    name: str
    raw: str = ""
    kind: str = "unknown"  # active / charge / command / passive / troop / unknown
    base_prob: float = 0.0  # percent
    level: int = 20
    awaken: bool = True

    # dict-like helpers (for backward compatibility with older UI code)
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

@dataclass
class Unit:
    name: str
    side: str  # "ally" or "enemy"
    wu: int
    int_: int
    lea: int
    spd: int
    max_soldiers: int
    soldiers: int
    unique_skill: Optional[Skill] = None
    inherited: List[Skill] = field(default_factory=list)

    # dynamic modifiers
    dmg_bonus: float = 0.0       # +% as decimal e.g. 0.20
    dmg_reduction: float = 0.0   # -% as decimal
    strat_bonus: float = 0.0
    phys_bonus: float = 0.0
    heal_bonus: float = 0.0
    statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # code -> {turns, ...}

    def alive(self) -> bool:
        return self.soldiers > 0

# -----------------------------
# Damage / heal formulas
# -----------------------------

def _base_damage(stat_atk: float, stat_def: float, soldiers_atk: int) -> float:
    # Base scaling for damage/heal.
    BASE_CONST = 500.0
    return (((stat_atk - stat_def) * 1.4 + (-0.05 / 10000.0 * soldiers_atk) + 0.1 + BASE_CONST) * soldiers_atk) / TROOP_SCALE

def damage_strategy(attacker: Unit, defender: Unit, dmg_rate: float, matchup: float, rng: random.Random) -> int:
    base = _base_damage(attacker.int_, defender.int_, attacker.soldiers)
    mult = (1.0 + attacker.dmg_bonus + attacker.strat_bonus) * (1.0 - defender.dmg_reduction)
    mult *= (1.0 + matchup)
    jitter = rng.uniform(0.95, 1.05)
    dmg = max(0.0, base * mult * dmg_rate * jitter)
    return int(dmg)

def damage_physical(attacker: Unit, defender: Unit, dmg_rate: float, matchup: float, rng: random.Random) -> int:
    base = _base_damage(attacker.wu, defender.lea, attacker.soldiers)
    mult = (1.0 + attacker.dmg_bonus + attacker.phys_bonus) * (1.0 - defender.dmg_reduction)
    mult *= (1.0 + matchup)
    jitter = rng.uniform(0.95, 1.05)
    dmg = max(0.0, base * mult * dmg_rate * jitter)
    return int(dmg)

def heal_amount(healer: Unit, target: Unit, heal_rate: float, rng: random.Random) -> int:
    base = max(0.0, _base_damage(healer.int_, 0, healer.soldiers) * 0.6)
    mult = (1.0 + healer.heal_bonus)
    jitter = rng.uniform(0.95, 1.05)
    heal = max(0.0, base * mult * heal_rate * jitter)
    return int(heal)

# -----------------------------
# Battle simulation
# -----------------------------

@dataclass
class LogRow:
    turn: int
    order: int
    side: str
    unit: str
    action_type: str  # "normal" "skill" "fail"
    action_name: str
    detail: str
    actor_hp: int
    target_hp: Optional[int] = None

def _matchup(attacker: Unit, defender: Unit) -> float:
    # Placeholder: without explicit troop types, keep neutral (0.0).
    return 0.0

def _tick_statuses_once_per_turn(unit: Unit):
    """Decrement status durations ONCE per turn (end of turn)."""
    to_del: List[str] = []
    for k, v in list(unit.statuses.items()):
        # seal_attack: decay probability each turn before decrementing
        if k == "seal_attack":
            p = float(v.get("p", 0.70))
            decay = float(v.get("decay", 0.14))
            v["p"] = max(0.0, p - decay)

        if "turns" in v:
            v["turns"] -= 1
            if v["turns"] <= 0:
                to_del.append(k)
    for k in to_del:
        unit.statuses.pop(k, None)

def _apply_dot(unit: Unit, code: str, attacker: Optional[Unit], rng: random.Random) -> Optional[int]:
    st = unit.statuses.get(code)
    if not st or not attacker:
        return None
    rate = float(st.get("rate", 0.07))
    dtype = st.get("dtype", "strategy")
    matchup = 0.0
    if dtype == "physical":
        dmg = damage_physical(attacker, unit, rate, matchup, rng)
    else:
        dmg = damage_strategy(attacker, unit, rate, matchup, rng)
    unit.soldiers = max(0, unit.soldiers - dmg)
    return dmg

def _can_act(unit: Unit, rng: random.Random) -> Tuple[bool, str]:
    if "stun" in unit.statuses:
        return False, "威圧（行動不能）"
    if "paralyze" in unit.statuses:
        p = unit.statuses["paralyze"].get("p", 0.30)
        if rng.random() < p:
            return False, "麻痺（行動不能）"
    return True, ""

def _choose_target(attacker: Unit, allies: List[Unit], enemies: List[Unit], target_mode: str, rng: random.Random) -> List[Unit]:
    alive_enemies = [u for u in enemies if u.alive()]
    alive_allies = [u for u in allies if u.alive()]

    if target_mode == "enemy_all":
        return alive_enemies[:]
    if target_mode == "enemy_multi":
        rng.shuffle(alive_enemies)
        return alive_enemies[:2] if len(alive_enemies) >= 2 else alive_enemies
    if target_mode == "enemy_single":
        if not alive_enemies:
            return []
        return [rng.choice(alive_enemies)]
    if target_mode == "ally_all":
        return alive_allies[:]
    if target_mode == "ally_multi":
        rng.shuffle(alive_allies)
        return alive_allies[:2] if len(alive_allies) >= 2 else alive_allies
    if target_mode == "ally_single":
        if not alive_allies:
            return []
        return [min(alive_allies, key=lambda u: u.soldiers)]
    if target_mode == "self":
        return [attacker]
    return []

def _apply_opening_effects(allies: List[Unit], enemies: List[Unit], rng: random.Random, logs: List[LogRow]):
    """Apply opening-only effects for passive/command skills (best-effort).

    - Passive: 連撃（通常攻撃2回）, 武勇+X
    - Command: 気炎万丈系の 封撃（通常攻撃不可, 確率+減衰）
    """
    for unit in allies + enemies:
        if not unit.alive():
            continue

        actor_allies = allies if unit.side == "ally" else enemies
        actor_enemies = enemies if unit.side == "ally" else allies

        for sk in [unit.unique_skill] + unit.inherited:
            if sk is None:
                continue
            raw = sk.raw or ""

            # --- Passive: 連撃 / 武勇+X（成田固有など想定） ---
            if sk.kind == "passive":
                applied = []
                if "連撃" in raw:
                    unit.statuses.setdefault("double_attack", {"turns": 999999})
                    applied.append("連撃")
                m = re.search(r"武勇が\s*(\d+)\s*増加", raw)
                if m and "bonus_wu" not in unit.statuses:
                    unit.wu += int(m.group(1))
                    unit.statuses["bonus_wu"] = {"turns": 999999}
                    applied.append(f"武勇+{m.group(1)}")

                if applied:
                    logs.append(
                        LogRow(
                            0,
                            0,
                            unit.side,
                            unit.name,
                            "skill",
                            sk.name,
                            "開幕効果: " + " / ".join(applied),
                            actor_hp=unit.soldiers,
                        )
                    )
                continue  # passiveはここで終わり

            # --- Command: 封撃付与（気炎万丈想定） ---
            if (sk.kind == "command") and ("封撃" in raw) and ("通常攻撃不可" in raw):
                targets_mode = detect_targets(raw)
                targets = _choose_target(unit, actor_allies, actor_enemies, targets_mode, rng)

                p0, decay, turns = _parse_kien_params(raw)
                for t in targets:
                    t.statuses["seal_attack"] = {"turns": turns, "p": p0, "decay": decay}

                if targets:
                    names = ", ".join(t.name for t in targets)
                    logs.append(
                        LogRow(
                            0,
                            0,
                            unit.side,
                            unit.name,
                            "skill",
                            sk.name,
                            f"開幕効果: 封撃付与 → {names}（{int(p0*100)}%/減衰{int(decay*100)}%）",
                            actor_hp=unit.soldiers,
                        )
                    )

def _try_cast_skill(
    unit: Unit,
    skill: Skill,
    allies: List[Unit],
    enemies: List[Unit],
    rng: random.Random,
) -> Tuple[bool, str, List[Tuple[str, int, int]]]:
    """
    Returns (casted, detail, list of (target_name, delta_soldiers(neg for dmg, pos for heal))).
    """
    bp = float(getattr(skill, 'base_prob', 0.0) or 0.0)
    # base_prob may be stored as percent (35.0) or fraction (0.35); support both.
    prob = (bp / 100.0) if bp > 1.0 else bp
    if prob <= 0:
        return False, "発動率0%（未設定）", []
    if rng.random() > prob:
        return False, "不発", []

    raw = skill.raw or ""

    # Treat passive/command skills as non-consuming (they are applied at opening effects).
    if (skill.kind or '').lower() in ('passive', 'command') or ('受動' in raw) or ('指揮' in raw):
        return False, '（自動効果）', []
    dmg_type = detect_damage_type(raw)
    targets_mode = detect_targets(raw)
    rate = parse_first_rate(raw, level=skill.level, awaken=skill.awaken)

    # Unknown skill: cast but no effect
    if not raw or (dmg_type is None and "回復" not in raw and "回復率" not in raw and rate is None):
        return True, "発動（効果未登録）", []

    targets = _choose_target(unit, allies, enemies, targets_mode, rng)
    results: List[Tuple[str, int, int]] = []

    # Heal skill
    if "回復" in raw or "回復率" in raw:
        if rate is None:
            rate = 1.0
        for t in targets:
            h = heal_amount(unit, t, rate, rng)
            t.soldiers = min(t.max_soldiers, t.soldiers + h)
            results.append((t.name, +h, t.soldiers))
        return True, f"回復 {int(rate*100)}%", results

    # Damage skill
    if rate is None:
        rate = 1.0
    for t in targets:
        matchup = _matchup(unit, t)
        if dmg_type in ("strategy", "hybrid") and (dmg_type == "strategy" or unit.int_ >= unit.wu):
            d = damage_strategy(unit, t, rate, matchup, rng)
        else:
            d = damage_physical(unit, t, rate, matchup, rng)

        t.soldiers = max(0, t.soldiers - d)
        results.append((t.name, -d, t.soldiers))

        # status apply (best-effort)
        sts = detect_statuses(raw)
        if sts:
            for code in sts:
                if code in ("burn", "flood", "collapse", "depress"):
                    t.statuses[code] = {"turns": 2, "rate": 0.07, "dtype": "strategy", "source": unit.name}
                    break
                if code in ("stun", "confuse", "silence", "disarm", "paralyze"):
                    t.statuses[code] = {"turns": 1}
                    break

    return True, f"ダメージ {int(rate*100)}%", results

def simulate_battle(
    allies: List[Unit],
    enemies: List[Unit],
    turns: int = 8,
    seed: int = 42,
) -> Tuple[List[LogRow], Dict[str, Any]]:
    rng = random.Random(seed)
    logs: List[LogRow] = []

    # Apply opening passive/command effects
    _apply_opening_effects(allies, enemies, rng, logs)

    def get_unit_by_name(name: str) -> Optional[Unit]:
        for u in allies + enemies:
            if u.name == name:
                return u
        return None

    for turn in range(1, turns + 1):
        # DOT tick at turn start
        for u in allies + enemies:
            if not u.alive():
                continue
            for code in ("burn", "flood", "collapse", "depress"):
                if code in u.statuses:
                    src = get_unit_by_name(u.statuses[code].get("source", ""))
                    dmg = _apply_dot(u, code, src, rng)
                    if dmg:
                        logs.append(
                            LogRow(
                                turn,
                                0,
                                u.side,
                                u.name,
                                "skill",
                                f"{code.upper()}(DOT)",
                                f"継続ダメージ {dmg}",
                                actor_hp=u.soldiers,
                                target_hp=u.soldiers,
                            )
                        )

        # Determine action order
        living = [u for u in allies + enemies if u.alive()]
        rng.shuffle(living)
        living.sort(key=lambda u: u.spd, reverse=True)

        # Execute actions
        for idx, actor in enumerate(living, start=1):
            if not actor.alive():
                continue

            # End early if one side wiped
            if not any(u.alive() for u in allies) or not any(u.alive() for u in enemies):
                break

            can_act, reason = _can_act(actor, rng)
            if not can_act:
                logs.append(LogRow(turn, idx, actor.side, actor.name, "fail", "行動不能", reason, actor_hp=actor.soldiers))
                continue

            actor_allies = allies if actor.side == "ally" else enemies
            actor_enemies = enemies if actor.side == "ally" else allies
            confused = "confuse" in actor.statuses

            # Try skills (unique then inherited)
            casted = False
            for sk in [actor.unique_skill] + actor.inherited:
                if sk is None:
                    continue

                # passive/command are handled at opening (avoid consuming action)
                sk_raw = (sk.raw or "")
                if (sk.kind or "").lower() in ("passive", "command") or ("受動" in sk_raw) or ("指揮" in sk_raw):
                    continue

                # silence blocks active/charge skills best-effort
                if "silence" in actor.statuses and (sk.kind or "").lower() in ("active", "charge"):
                    continue

                ok, detail, results = _try_cast_skill(actor, sk, actor_allies, actor_enemies, rng)
                if ok:
                    casted = True
                    if results:
                        res_txt = ", ".join([f"{t}{'+' if d>0 else ''}{d}（残兵 {remain}）" for t, d, remain in results])
                    else:
                        res_txt = "—"
                    logs.append(
                        LogRow(
                            turn,
                            idx,
                            actor.side,
                            actor.name,
                            "skill",
                            sk.name,
                            f"{detail} / {res_txt}",
                            actor_hp=actor.soldiers,
                            target_hp=results[0][2] if len(results) == 1 else None,
                        )
                    )
                    break

            if casted:
                continue

            # Otherwise normal attack
            # disarm prevents normal attack (always)
            if "disarm" in actor.statuses:
                logs.append(LogRow(turn, idx, actor.side, actor.name, "fail", "通常攻撃不可", "封撃", actor_hp=actor.soldiers))
                continue

            # seal_attack: chance-based normal attack block（気炎万丈）
            if "seal_attack" in actor.statuses:
                stt = actor.statuses["seal_attack"]
                p = float(stt.get("p", 0.70))
                if rng.random() < p:
                    logs.append(LogRow(turn, idx, actor.side, actor.name, "fail", "通常攻撃不可", f"封撃（{int(p*100)}%）", actor_hp=actor.soldiers))
                    continue

            targets = [u for u in actor_enemies if u.alive()]
            if not targets:
                break

            hits = 2 if "double_attack" in actor.statuses else 1
            target = rng.choice(targets) if confused else targets[0]

            for h in range(hits):
                if not target.alive():
                    targets = [u for u in actor_enemies if u.alive()]
                    if not targets:
                        break
                    target = rng.choice(targets) if confused else targets[0]

                matchup = _matchup(actor, target)
                dmg = damage_physical(actor, target, 1.0, matchup, rng)
                target.soldiers = max(0, target.soldiers - dmg)

                atk_name = "通常攻撃" if hits == 1 else f"通常攻撃({h+1}/{hits})"
                logs.append(
                    LogRow(
                        turn,
                        idx,
                        actor.side,
                        actor.name,
                        "normal",
                        atk_name,
                        f"{target.name} -{dmg}（残兵 {target.soldiers}）",
                        actor_hp=actor.soldiers,
                        target_hp=target.soldiers,
                    )
                )

        # End-of-turn status tick (once per turn)
        for u in allies + enemies:
            if u.alive():
                _tick_statuses_once_per_turn(u)

        if not any(u.alive() for u in allies) or not any(u.alive() for u in enemies):
            break

    summary = {
        "ally_alive": sum(1 for u in allies if u.alive()),
        "enemy_alive": sum(1 for u in enemies if u.alive()),
        "ally_soldiers": sum(u.soldiers for u in allies),
        "enemy_soldiers": sum(u.soldiers for u in enemies),
    }
    return logs, summary

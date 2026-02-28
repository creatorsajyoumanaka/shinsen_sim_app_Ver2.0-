# -*- coding: utf-8 -*-
"""
Lightweight battle simulator engine for Nobunaga Shinsen-style tactics.

Goals:
- Never crash on unknown tactics (treat as no-op, still log if it "fires").
- Best-effort parsing from "raw" text.
- Deterministic when seed is fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import random
import re

# Damage scaling: divide base damage by this value to keep numbers in a sensible range.
TROOP_SCALE = 10000

# -----------------------------
# Parsing helpers
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
    """Parse activation probability percent (max if arrow exists), e.g. '35%→45%' -> 45.0."""
    try:
        if not raw:
            return default
        m = _PROB_RE.search(raw)
        if not m:
            return default
        vicinity = raw[m.start(): m.start() + 40]
        am = _ARROW_RE.search(vicinity)
        if am:
            return float(am.group(2))
        return float(m.group(1))
    except Exception:
        return default

def parse_first_rate(raw: str, level: int = 20, awaken: bool = True) -> Optional[float]:
    """Parse first damage/heal rate. Returns decimal (e.g., 104% -> 1.04)."""
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
    """enemy_single / enemy_multi / enemy_all / ally_single / ally_multi / ally_all / self"""
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

# Best-effort status keywords (avoid duplicate keys!)
STATUS_KEYWORDS: Dict[str, str] = {
    "威圧": "stun",
    "麻痺": "paralyze",
    "混乱": "confuse",
    "無策": "silence",
    "封撃": "disarm",     # hard block normal attack
    "火傷": "burn",
    "水攻め": "flood",
    "潰走": "collapse",
    "消沈": "depress",
}

def detect_statuses(raw: str) -> List[str]:
    if not raw:
        return []
    found: List[str] = []
    for jp, code in STATUS_KEYWORDS.items():
        if jp in raw:
            found.append(code)
    return found

# -----------------------------
# Core data structures
# -----------------------------

@dataclass
class Skill:
    name: str
    raw: str = ""
    kind: str = "unknown"     # active / charge / command / passive / troop / unknown
    base_prob: float = 0.0    # percent (0..100)
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

    dmg_bonus: float = 0.0
    dmg_reduction: float = 0.0
    strat_bonus: float = 0.0
    phys_bonus: float = 0.0
    heal_bonus: float = 0.0

    # statuses: code -> dict(turns=..., other params...)
    statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def alive(self) -> bool:
        return self.soldiers > 0

# -----------------------------
# Formulas
# -----------------------------

def _base_damage(stat_atk: float, stat_def: float, soldiers_atk: int) -> float:
    BASE_CONST = 500.0
    return (((stat_atk - stat_def) * 1.4 + (-0.05 / 10000.0 * soldiers_atk) + 0.1 + BASE_CONST) * soldiers_atk) / TROOP_SCALE

def damage_strategy(attacker: Unit, defender: Unit, rate: float, matchup: float, rng: random.Random) -> int:
    base = _base_damage(attacker.int_, defender.int_, attacker.soldiers)
    mult = (1.0 + attacker.dmg_bonus + attacker.strat_bonus) * (1.0 - defender.dmg_reduction)
    mult *= (1.0 + matchup)
    jitter = rng.uniform(0.95, 1.05)
    return int(max(0.0, base * mult * rate * jitter))

def damage_physical(attacker: Unit, defender: Unit, rate: float, matchup: float, rng: random.Random) -> int:
    base = _base_damage(attacker.wu, defender.lea, attacker.soldiers)
    mult = (1.0 + attacker.dmg_bonus + attacker.phys_bonus) * (1.0 - defender.dmg_reduction)
    mult *= (1.0 + matchup)
    jitter = rng.uniform(0.95, 1.05)
    return int(max(0.0, base * mult * rate * jitter))

def heal_amount(healer: Unit, target: Unit, rate: float, rng: random.Random) -> int:
    base = max(0.0, _base_damage(healer.int_, 0, healer.soldiers) * 0.6)
    mult = (1.0 + healer.heal_bonus)
    jitter = rng.uniform(0.95, 1.05)
    return int(max(0.0, base * mult * rate * jitter))

# -----------------------------
# Battle simulation
# -----------------------------

@dataclass
class LogRow:
    turn: int
    order: int
    side: str
    unit: str
    action_type: str      # "normal" / "skill" / "fail"
    action_name: str
    detail: str
    actor_hp: int
    target_hp: Optional[int] = None

def _matchup(attacker: Unit, defender: Unit) -> float:
    return 0.0

def _tick_statuses(unit: Unit) -> None:
    """Tick once per turn (NOT per action)."""
    to_del: List[str] = []
    for k, v in unit.statuses.items():
        if k == "seal_attack":
            p = float(v.get("p", 0.70))
            decay = float(v.get("decay", 0.14))
            v["p"] = max(0.0, p - decay)

        v["turns"] = int(v.get("turns", 0)) - 1
        if v["turns"] <= 0:
            to_del.append(k)

    for k in to_del:
        unit.statuses.pop(k, None)

def _can_act(unit: Unit, rng: random.Random) -> Tuple[bool, str]:
    if "stun" in unit.statuses:
        return False, "威圧（行動不能）"
    if "paralyze" in unit.statuses:
        p = float(unit.statuses["paralyze"].get("p", 0.30))
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
        return [rng.choice(alive_enemies)] if alive_enemies else []

    if target_mode == "ally_all":
        return alive_allies[:]
    if target_mode == "ally_multi":
        rng.shuffle(alive_allies)
        return alive_allies[:2] if len(alive_allies) >= 2 else alive_allies
    if target_mode == "ally_single":
        return [min(alive_allies, key=lambda u: u.soldiers)] if alive_allies else []

    if target_mode == "self":
        return [attacker]
    return []

def _apply_passive(unit: Unit, raw: str) -> List[str]:
    """Apply passive effects and return notes."""
    notes: List[str] = []
    if "連撃" in raw:
        if "double_attack" not in unit.statuses:
            unit.statuses["double_attack"] = {"turns": 999999}
            notes.append("連撃付与")
    m = re.search(r"武勇が\s*(\d+)\s*増加", raw)
    if m and "bonus_wu" not in unit.statuses:
        inc = int(m.group(1))
        unit.wu += inc
        unit.statuses["bonus_wu"] = {"turns": 999999, "amount": inc}
        notes.append(f"武勇+{inc}")
    return notes

def _apply_command_seal_attack(
    caster: Unit,
    raw: str,
    allies: List[Unit],
    enemies: List[Unit],
    rng: random.Random,
) -> List[Unit]:
    """気炎万丈想定: 敵軍複数(2名)に seal_attack を3ターン付与（毎ターン発動率減衰）。"""
    targets_mode = detect_targets(raw)
    targets = _choose_target(caster, allies, enemies, targets_mode, rng)
    for t in targets:
        t.statuses["seal_attack"] = {"turns": 3, "p": 0.70, "decay": 0.14}
    return targets

def _apply_opening_effects(allies: List[Unit], enemies: List[Unit], rng: random.Random, logs: List[LogRow]) -> None:
    """Battle start: apply passive/command effects once (turn=0)."""
    for unit in allies + enemies:
        if not unit.alive():
            continue

        actor_allies = allies if unit.side == "ally" else enemies
        actor_enemies = enemies if unit.side == "ally" else allies

        for sk in [unit.unique_skill] + unit.inherited:
            if sk is None:
                continue
            raw = sk.raw or ""

            # passive
            if sk.kind == "passive" or "受動" in raw:
                notes = _apply_passive(unit, raw)
                if notes:
                    logs.append(LogRow(0, 0, unit.side, unit.name, "skill", sk.name, f"開幕AUTO（受動）: {', '.join(notes)}", actor_hp=unit.soldiers))
                continue

            # command: 封撃（通常攻撃不可）
            if (sk.kind == "command" or "指揮" in raw) and ("封撃" in raw and "通常攻撃不可" in raw):
                if "seal_attack_applied" in unit.statuses:
                    continue
                targets = _apply_command_seal_attack(unit, raw, actor_allies, actor_enemies, rng)
                unit.statuses["seal_attack_applied"] = {"turns": 999999}
                if targets:
                    names = ", ".join(t.name for t in targets)
                    logs.append(LogRow(0, 0, unit.side, unit.name, "skill", sk.name, f"開幕AUTO（指揮）: 封撃付与 → {names}", actor_hp=unit.soldiers))
                continue

def _try_skill(
    unit: Unit,
    skill: Skill,
    allies: List[Unit],
    enemies: List[Unit],
    rng: random.Random,
) -> Tuple[bool, str, List[Tuple[str, int, int]]]:
    """
    Returns (consumes_action, detail, results)
      - consumes_action=False for passive/command auto effects (so unit can still act).
      - results: (target_name, delta(+heal/-dmg), remaining)
    """
    raw = skill.raw or ""

    # passive / command effects are already applied at opening, but keep guard for safety.
    if skill.kind == "passive" or "受動" in raw:
        _apply_passive(unit, raw)
        return False, "AUTO（受動）", []

    if (skill.kind == "command" or "指揮" in raw) and ("封撃" in raw and "通常攻撃不可" in raw):
        if "seal_attack_applied" not in unit.statuses:
            targets = _apply_command_seal_attack(unit, raw, allies, enemies, rng)
            unit.statuses["seal_attack_applied"] = {"turns": 999999}
            return False, "AUTO（指揮）: 封撃付与", [(t.name, 0, t.soldiers) for t in targets]
        return False, "AUTO（指揮）: 封撃（既適用）", []

    # active/charge/unknown: proc check
    prob = float(skill.base_prob) / 100.0 if skill.base_prob else 0.0
    if prob <= 0.0:
        return False, "発動率0%（未設定）", []
    if rng.random() > prob:
        return True, "不発", []

    dmg_type = detect_damage_type(raw)
    targets_mode = detect_targets(raw)
    rate = parse_first_rate(raw, level=skill.level, awaken=skill.awaken)

    if not raw or (dmg_type is None and "回復" not in raw and "回復率" not in raw and rate is None):
        return True, "発動（効果未登録）", []

    targets = _choose_target(unit, allies, enemies, targets_mode, rng)
    results: List[Tuple[str, int, int]] = []

    # heal
    if "回復" in raw or "回復率" in raw:
        if rate is None:
            rate = 1.0
        for t in targets:
            h = heal_amount(unit, t, rate, rng)
            t.soldiers = min(t.max_soldiers, t.soldiers + h)
            results.append((t.name, +h, t.soldiers))
        return True, f"回復 {int(rate * 100)}%", results

    # damage
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

        sts = detect_statuses(raw)
        for code in sts:
            if code in ("burn", "flood", "collapse", "depress"):
                t.statuses[code] = {"turns": 2, "rate": 0.07, "dtype": "strategy", "source": unit.name}
                break
            if code in ("stun", "confuse", "silence", "disarm", "paralyze"):
                t.statuses[code] = {"turns": 1}
                break

    return True, f"ダメージ {int(rate * 100)}%", results

def simulate_battle(
    allies: List[Unit],
    enemies: List[Unit],
    turns: int = 8,
    seed: int = 42,
) -> Tuple[List[LogRow], Dict[str, Any]]:
    rng = random.Random(seed)
    logs: List[LogRow] = []

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
                    st = u.statuses.get(code, {})
                    src = get_unit_by_name(st.get("source", ""))
                    if not src:
                        continue
                    rate = float(st.get("rate", 0.07))
                    dtype = st.get("dtype", "strategy")
                    if dtype == "physical":
                        dmg = damage_physical(src, u, rate, 0.0, rng)
                    else:
                        dmg = damage_strategy(src, u, rate, 0.0, rng)
                    u.soldiers = max(0, u.soldiers - dmg)
                    logs.append(
                        LogRow(
                            turn, 0, u.side, u.name, "skill",
                            f"{code.upper()}(DOT)", f"継続ダメージ {dmg}",
                            actor_hp=u.soldiers, target_hp=u.soldiers
                        )
                    )

        # action order
        living = [u for u in allies + enemies if u.alive()]
        rng.shuffle(living)
        living.sort(key=lambda u: u.spd, reverse=True)

        for order, actor in enumerate(living, start=1):
            if not actor.alive():
                continue
            if not any(u.alive() for u in allies) or not any(u.alive() for u in enemies):
                break

            can_act, reason = _can_act(actor, rng)
            if not can_act:
                logs.append(LogRow(turn, order, actor.side, actor.name, "fail", "行動不能", reason, actor_hp=actor.soldiers))
                continue

            actor_allies = allies if actor.side == "ally" else enemies
            actor_enemies = enemies if actor.side == "ally" else allies
            confused = "confuse" in actor.statuses

            # Try skills
            consumed_action = False
            for sk in [actor.unique_skill] + actor.inherited:
                if sk is None:
                    continue
                if "silence" in actor.statuses and sk.kind in ("active", "charge"):
                    continue

                consumes, detail, results = _try_skill(actor, sk, actor_allies, actor_enemies, rng)

                # AUTO effects: ignore (already applied)
                if detail.startswith("AUTO"):
                    continue

                if consumes:
                    if detail == "不発":
                        continue  # try next skill
                    consumed_action = True
                    res_txt = "—"
                    if results:
                        res_txt = ", ".join([f"{t}{'+' if d > 0 else ''}{d}（残兵 {remain}）" for t, d, remain in results])
                    logs.append(
                        LogRow(
                            turn, order, actor.side, actor.name, "skill",
                            sk.name, f"{detail} / {res_txt}",
                            actor_hp=actor.soldiers,
                            target_hp=results[0][2] if len(results) == 1 else None
                        )
                    )
                    break

            if consumed_action:
                continue

            # Normal attack
            if "disarm" in actor.statuses:
                logs.append(LogRow(turn, order, actor.side, actor.name, "fail", "通常攻撃不可", "封撃", actor_hp=actor.soldiers))
                continue

            if "seal_attack" in actor.statuses:
                st = actor.statuses["seal_attack"]
                p = float(st.get("p", 0.70))
                if rng.random() < p:
                    logs.append(LogRow(turn, order, actor.side, actor.name, "fail", "通常攻撃不可", f"封撃（{int(p * 100)}%）", actor_hp=actor.soldiers))
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

                dmg = damage_physical(actor, target, 1.0, _matchup(actor, target), rng)
                target.soldiers = max(0, target.soldiers - dmg)

                atk_name = "通常攻撃" if hits == 1 else f"通常攻撃({h + 1}/{hits})"
                logs.append(
                    LogRow(
                        turn, order, actor.side, actor.name, "normal",
                        atk_name, f"{target.name} -{dmg}（残兵 {target.soldiers}）",
                        actor_hp=actor.soldiers, target_hp=target.soldiers
                    )
                )

        # Tick statuses once per turn
        for u in allies + enemies:
            if u.alive() and u.statuses:
                _tick_statuses(u)

        if not any(u.alive() for u in allies) or not any(u.alive() for u in enemies):
            break

    summary = {
        "ally_alive": sum(1 for u in allies if u.alive()),
        "enemy_alive": sum(1 for u in enemies if u.alive()),
        "ally_soldiers": sum(u.soldiers for u in allies),
        "enemy_soldiers": sum(u.soldiers for u in enemies),
    }
    return logs, summary

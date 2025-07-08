"""
Load *config.yml* and expose a validated list of ``Condition`` objects.

YAML schema
~~~~~~~~~~~
conditions:
  - name: "Straight Bleached"
    slots: "1-3,5,8"         # comma-sep ints or ranges
    control: true            # exactly ONE condition must be control: true
  - name: "Treated"
    slots: "4,6-7,9-10"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


# ───────────────────────── exceptions ──────────────────────────
class ConfigError(Exception):
    """Invalid or inconsistent *config.yml*."""


# ───────────────────────── dataclass ───────────────────────────
@dataclass(slots=True, frozen=True)
class Condition:
    name: str
    slots: List[int]
    is_control: bool = False


# ───────────────────── helper – expand slots ───────────────────
_SLOT_RE = re.compile(r"^\d+(-\d+)?$")


def _expand_slots(spec: str) -> List[int]:
    """
    "1-3,5,8" -> [1,2,3,5,8]
    """
    out: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if not _SLOT_RE.match(token):
            raise ConfigError(f"Bad slot token '{token}'")

        if "-" in token:
            a, b = map(int, token.split("-"))
            if b < a:
                raise ConfigError(f"Range '{token}' is reversed")
            out.extend(range(a, b + 1))
        else:
            out.append(int(token))
    return out


# ───────────────────── public loader ───────────────────────────
def load_config(exp_dir: Path) -> List[Condition]:
    """
    Read ``exp_dir/config.yml`` and return a list of :class:`Condition`.

    Raises
    ------
    FileNotFoundError
        config.yml missing
    ConfigError
        Schema/logic errors (duplicate slots, >1 control, etc.)
    """
    cfg_path = exp_dir / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict) or "conditions" not in data:
        raise ConfigError("Top-level key 'conditions' missing")

    conds: List[Condition] = []
    for obj in data["conditions"]:
        if not isinstance(obj, dict):
            raise ConfigError("Each item in 'conditions' must be a mapping")
        if "name" not in obj or "slots" not in obj:
            raise ConfigError("Condition needs 'name' and 'slots' fields")

        name = str(obj["name"])
        slots = _expand_slots(str(obj["slots"]))
        is_ctrl = bool(obj.get("control", False))
        conds.append(Condition(name, slots, is_ctrl))

    # one – and only one – control
    n_ctrl = sum(c.is_control for c in conds)
    if n_ctrl != 1:
        raise ConfigError("Exactly one condition must have control: true")

    # duplicate slots
    seen: dict[int, str] = {}
    for c in conds:
        for s in c.slots:
            if s in seen:
                raise ConfigError(
                    f"Slot {s} appears in both '{seen[s]}' and '{c.name}'"
                )
            seen[s] = c.name

    return conds

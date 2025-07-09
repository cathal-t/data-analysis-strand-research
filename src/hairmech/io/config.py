"""
Load *config.yml* and expose a validated list of ``Condition`` objects.

YAML schema
~~~~~~~~~~~
conditions:
  - name: "Straight Bleached"
    slots: "1-3,5,8"          # comma-separated ints or ranges
    control: true             # **exactly ONE** condition must be control: true
  - name: "Treated"
    slots: "4,6-7,9-10"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

__all__ = [
    "Condition",
    "ConfigError",
    "load_config",
    "dump_config",
]

# ───────────────────────── exceptions ──────────────────────────
class ConfigError(Exception):
    """Invalid or inconsistent *config.yml*."""


# ───────────────────────── dataclass ───────────────────────────
@dataclass(slots=True, frozen=True)
class Condition:
    """
    One experimental condition (group of fibres).

    Parameters
    ----------
    name
        Friendly name (used in legends, Excel etc.).
    slots
        List of individual tensile-test slot numbers **in ascending order**.
    is_control
        Mark the single reference condition (required by statistics).
    """

    name: str
    slots: Sequence[int] = field(repr=False)
    is_control: bool = False

    # immutable defensive copy
    def __post_init__(self) -> None:  # noqa: D401
        object.__setattr__(self, "slots", tuple(sorted(set(self.slots))))


# ───────────────────── helper – expand & validate slots ───────────────
_SLOT_RE = re.compile(r"^\d+(-\d+)?$")


def _expand_slots(spec: str) -> List[int]:
    """
    `"1-3,5,8"` → ``[1, 2, 3, 5, 8]``

    Raises
    ------
    ConfigError
        Malformed token or reversed range.
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


# ───────────────────── public loader ──────────────────────────────────
def load_config(exp_dir: Path) -> List[Condition]:
    """
    Read ``exp_dir/config.yml`` and return a list of :class:`Condition`.

    Raises
    ------
    FileNotFoundError
        *config.yml* is missing.
    ConfigError
        Schema / logic errors (duplicate slots, ≠1 control, …).
    """
    cfg_path = exp_dir / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    data = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(data, dict) or "conditions" not in data:
        raise ConfigError("Top-level key 'conditions' missing")

    conds: List[Condition] = []
    for obj in data["conditions"]:
        if not isinstance(obj, dict):
            raise ConfigError("Each item in 'conditions' must be a mapping")
        if "name" not in obj or "slots" not in obj:
            raise ConfigError("Condition needs 'name' **and** 'slots' fields")

        name = str(obj["name"])
        slots = _expand_slots(str(obj["slots"]))
        is_ctrl = bool(obj.get("control", False))
        conds.append(Condition(name, slots, is_ctrl))

    # one – and only one – control
    n_ctrl = sum(c.is_control for c in conds)
    if n_ctrl != 1:
        raise ConfigError("Exactly **one** condition must have control: true")

    # duplicate slots
    seen: Dict[int, str] = {}
    for c in conds:
        for s in c.slots:
            if s in seen:
                raise ConfigError(
                    f"Slot {s} appears in both '{seen[s]}' and '{c.name}'"
                )
            seen[s] = c.name

    return conds


# ───────────────────── public dumper (needed by Dash UI) ──────────────
def dump_config(conditions: Sequence[Condition], file: Path | str) -> None:
    """
    Write *conditions* back to YAML (used by the Dash UI “Download config”).

    The written schema matches the one accepted by :func:`load_config`.
    """
    doc = {
        "conditions": [
            {
                "name": c.name,
                "slots": ",".join(
                    # compress consecutive runs into "a-b"
                    _compress_slot_run(run)
                    for run in _runs(c.slots)
                ),
                "control": bool(c.is_control),
            }
            for c in conditions
        ]
    }
    Path(file).write_text(yaml.safe_dump(doc, sort_keys=False))


# ─────────── tiny helpers to serialise slots back to compact ranges ───
def _runs(slots: Sequence[int]) -> List[List[int]]:
    """[[1,2,3],[5],[8,9]] for (1,2,3,5,8,9)"""
    runs: List[List[int]] = []
    for s in sorted(slots):
        if not runs or s != runs[-1][-1] + 1:
            runs.append([s])
        else:
            runs[-1].append(s)
    return runs


def _compress_slot_run(run: Sequence[int]) -> str:
    """[1,2,3] → '1-3'  ;  [7] → '7'"""
    if len(run) == 1:
        return str(run[0])
    return f"{run[0]}-{run[-1]}"

"""
Unit test for the YAML loader in hairmech.io.config
(Option A – config.yml lives inside each experiment folder)
"""
from pathlib import Path

from hairmech.io.config import load_config


def test_load_config():
    # demo_exp contains Dimensional/Tensile txt + config.yml
    root = Path(__file__).parent / "fixtures" / "demo_exp"

    conds = load_config(root)        # → list[Condition]

    # basic sanity checks
    assert len(conds) == 2                         # two conditions in YAML
    assert any(c.is_control for c in conds)        # exactly one control
    assert sum(c.is_control for c in conds) == 1

    # slot expansion worked (example YAML has 1-3)
    slots_all = sorted(s for c in conds for s in c.slots)
    assert slots_all == [1, 2, 3]

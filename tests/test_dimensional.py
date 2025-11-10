from pathlib import Path

import pytest

from hairmech.dimensional import DimensionalData

def test_dummy():
    # point at a very small sample fixture you'll add later
    sample = Path(__file__).parent / "fixtures" / "Dimensional_short.txt"
    if sample.exists():
        d = DimensionalData(sample)
        assert 1 in d.map


def _demo_sample() -> Path:
    return Path(__file__).parent / "fixtures" / "demo_exp" / "Dimensional_Data.txt"


@pytest.mark.skipif(not _demo_sample().exists(), reason="demo fixture missing")
def test_removed_slices_adjust_mean():
    sample = _demo_sample()
    base = DimensionalData(sample)
    cleaned = DimensionalData(sample, removed_slices={1: [2, 4]})

    assert 1 in base.map
    assert 1 in cleaned.map
    assert cleaned.map[1] < base.map[1]
    assert cleaned.removed_applied[1] == [2, 4]


@pytest.mark.skipif(not _demo_sample().exists(), reason="demo fixture missing")
def test_removed_slices_drop_slot_when_empty():
    sample = _demo_sample()
    cleaned = DimensionalData(sample, removed_slices={1: [1, 2, 3, 4, 5]})

    assert 1 not in cleaned.map
    assert 1 in cleaned.removed_empty_slots


@pytest.mark.skipif(not _demo_sample().exists(), reason="demo fixture missing")
def test_removed_slices_report_missing_entries():
    sample = _demo_sample()
    cleaned = DimensionalData(sample, removed_slices={99: [1], 1: [10]})

    assert 99 in cleaned.removed_missing_slots
    assert cleaned.removed_missing_slices[1] == [10]

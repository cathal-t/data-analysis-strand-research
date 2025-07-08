from hairmech.dimensional import DimensionalData
from pathlib import Path

def test_dummy():
    # point at a very small sample fixture you'll add later
    sample = Path(__file__).parent / "fixtures" / "Dimensional_short.txt"
    if sample.exists():
        d = DimensionalData(sample)
        assert 1 in d.map

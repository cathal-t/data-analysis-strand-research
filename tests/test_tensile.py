from hairmech.tensile import TensileTest
from pathlib import Path
import pandas as pd

def test_units_detection():
    # create a *very* small synthetic file in-memory
    txt = (
        "Record\t% Strain\tgmf\n"
        "1\t0\t10\n"
        "1\t0.1\t20\n"
    )
    path = Path(__file__).with_suffix(".tmp.txt")
    path.write_text(txt)
    tt = TensileTest(path)
    assert not tt.is_mpa
    path.unlink()  # clean up

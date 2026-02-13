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


def test_ascii_blocks_record_and_slot_are_stored_separately(tmp_path):
    txt = """ASCII Export File Version:,3
Sample / Slot Number:,2
Record,Index,Position,Strain,Time,Force
,,µm,%,s,N
101,0,0,0,0,0.10
bad_record,1,10,1.5,1,0.20

Sample / Slot Number:,5
Record,Index,Position,Strain,Time,Force
,,µm,%,s,N
200,0,0,0.5,0,0.30
"""
    path = tmp_path / "ascii_blocks.txt"
    path.write_text(txt)

    tt = TensileTest(path)

    assert list(tt.df["Record"]) == [101, 2, 200]
    assert list(tt.df["Slot"]) == [2, 2, 5]

    per_slot = {slot: grp for slot, grp in tt.per_slot()}
    assert sorted(per_slot) == [2, 5]
    assert list(per_slot[2]["Record"]) == [101, 2]
    assert list(per_slot[5]["Record"]) == [200]

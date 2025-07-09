from hairmech.ui.app import _rebalance_rows


def test_rebalance_even_three():
    # three blank rows, pretend total range is 100 (set on first row)
    rows = [
        {"slot_start": "", "slot_end": "", "name": ""},
        {"slot_start": "", "slot_end": "", "name": ""},
        {"slot_start": "", "slot_end": "", "name": ""},
    ]
    rows[0]["slot_end"] = 100

    _rebalance_rows(rows)

    expected = [(1, 33), (34, 66), (67, 100)]
    for r, exp in zip(rows, expected):
        assert (r["slot_start"], r["slot_end"]) == exp
        assert r["name"].startswith("Condition")

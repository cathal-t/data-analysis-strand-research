from hairmech.ui.app import _rebalance_rows


def test_rebalance_even_three():
    rows = [
        {"slot_start": "", "slot_end": "", "name": ""},
        {"slot_start": "", "slot_end": "", "name": ""},
        {"slot_start": "", "slot_end": "", "name": ""},
    ]
    # Pretend total range is 100 by writing it in the first row
    rows[0]["slot_end"] = 100

    _rebalance_rows(rows)

    expected = [(1, 34), (35, 67), (68, 100)]
    for r, exp in zip(rows, expected):
        assert (r["slot_start"], r["slot_end"]) == exp
        assert r["name"].startswith("Condition")

import pandas as pd

from hairmech.ui.app import _remove_initial_force_noise


def _build_group(force_values):
    n = len(force_values)
    return pd.DataFrame(
        {
            "Slot": [1] * n,
            "Record": [1] * n,
            "Index": list(range(n)),
            "Position_um": [0.0] * n,
            "Strain_pct": [0.01 * i for i in range(n)],
            "Time_s": [0.1 * i for i in range(n)],
            "Force_N": force_values,
        }
    )


def test_remove_initial_noise_detects_flat_segment():
    force = [0.0, 0.0, 0.01, -0.02, 0.0, 0.45, 0.52, 0.6, 0.68]
    df = _build_group(force)

    cleaned, trimmed = _remove_initial_force_noise(df)

    assert cleaned.iloc[0]["Force_N"] == force[5]
    assert trimmed[(1, 1)] == 5


def test_remove_initial_noise_leaves_clean_trace():
    force = [0.01, 0.05, 0.08, 0.12, 0.16, 0.22, 0.28]
    df = _build_group(force)

    cleaned, trimmed = _remove_initial_force_noise(df)

    pd.testing.assert_frame_equal(cleaned, df.reset_index(drop=True))
    assert trimmed[(1, 1)] == 0

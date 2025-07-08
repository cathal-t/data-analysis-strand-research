import pandas as pd
import numpy as np

from hairmech.util import rgba, hex_to_rgb01, post_gradient

# ---------- colour helpers -------------------------------------------
def test_rgba_hex_to_rgba():
    assert rgba("#000000", 0.5) == "rgba(0,0,0,0.5)"
    # round-trip: hex -> rgb01 -> rgba
    rgb01 = hex_to_rgb01("2a9d8f")            # (0.165, 0.616, 0.561)
    out   = rgba(rgb01, 1)
    assert out.startswith("rgba(") and out.endswith(",1)")

# ---------- post-gradient helper --------------------------------------
def test_post_gradient_simple():
    # fabricate a *tiny* processed curve with a clear break point
    df = pd.DataFrame({
        "strain":     [0.00, 0.04, 0.08, 0.12],     # 0–1
        "stress_Pa":  [0,    50e6, 90e6, 80e6],     # Pa
    })
    # With delta_pct=8, slope is (90-80) / 8 = 1.25 MPa/%  (90→80 over 8 %)
    grad = post_gradient(df, delta_pct=8.0)
    assert np.isclose(grad, 11.25, atol=1e-2)

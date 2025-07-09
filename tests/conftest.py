"""
Global pytest fixtures / hooks for the UI test-suite.

We return a *ChromeOptions* object – because `dash.testing` passes the
hook’s return-value straight into the **options=…** parameter of
`DashComposite`.  
At the same time we prepend the matching ChromeDriver (downloaded via
*webdriver-manager*) to PATH so that `webdriver.Chrome()` can start
without a version-mismatch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def _chrome_options() -> Options:
    """Headless Chrome options suitable for Dash smoke-tests."""
    opts = Options()
    opts.add_argument("--headless=new")          # modern headless mode
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1280,800")
    # let dash.testing add its own loggingPrefs capability later
    return opts


# ─────────────────────────── dash.testing hook ─────────────────────────
def pytest_setup_options() -> Options:          # noqa: D401  (dash.testing name)
    """
    Called by *dash.testing*.  
    MUST return a *ChromeOptions* instance – **not** a plain dict.
    """
    # 1) Ensure the right chromedriver is first on PATH
    drv_path = Path(ChromeDriverManager().install())
    os.environ["PATH"] = f"{drv_path.parent}{os.pathsep}{os.environ['PATH']}"

    # 2) Hand the options object to dash.testing
    return _chrome_options()

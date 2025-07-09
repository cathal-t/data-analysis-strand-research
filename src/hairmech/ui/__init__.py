"""
Dash UI package bootstrap.
"""
# Ensure chromedriver is discoverable for dash[testing] / selenium
try:                     # no extra cost in production
    import chromedriver_binary   # noqa: F401
except ModuleNotFoundError:
    # dev extra not installed – nothing to do
    pass

from .app import build_dash_app  
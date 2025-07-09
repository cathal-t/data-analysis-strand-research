import pytest
from dash.testing.application_runners import import_app
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


@pytest.mark.parametrize("tab", ["overlay", "violin"])
def test_dash_smoke_and_add_row(dash_duo, tab):
    """
    • App boots with no console errors.
    • After clicking 'Add condition' a second row appears (text contains
      'Condition 2').
    • Switching the view dropdown still works.
    """
    app = import_app("hairmech.ui.app")
    dash_duo.start_server(app)

    # Click the button that adds a row
    dash_duo.find_element("#btn-add").click()

    # Wait ≤8 s until any table cell contains 'Condition 2'
    WebDriverWait(dash_duo.driver, 8, poll_frequency=0.5).until(
        lambda drv: any(
            "Condition 2" in cell.text
            for cell in drv.find_elements(By.CSS_SELECTOR, "#cond-table .dash-cell")
        ),
        message="Condition 2 not rendered in DataTable",
    )

    # Change tab to ensure callbacks wired
    dash_duo.select_dcc_dropdown("#tabs", tab)

    # Browser console must be clean
    assert dash_duo.get_logs() == []

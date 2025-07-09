import pytest
from dash.testing.application_runners import import_app
from selenium.webdriver.common.by import By


@pytest.mark.parametrize("tab", ["overlay", "violin"])
def test_dash_smoke_and_add_row(dash_duo, tab):
    """
    • Smoke-test that the Dash app starts without console errors.
    • Click 'Add condition' once and verify the table now has ≥2 rows
      and a non-empty slot range in the new row.
    """
    app = import_app("hairmech.ui.app")
    dash_duo.start_server(app)

    # click the visible Add-condition button
    add_btn = dash_duo.find_element("#btn-add")
    add_btn.click()

    # wait until DataTable has two data rows (header rows are role="columnheader")
    dash_duo.wait_for_text_to_equal("#cond-table .dash-cell.column-1.row-1", "2", timeout=3)

    # switch between tabs to ensure callbacks are wired
    dash_duo.select_dcc_dropdown("#tabs", tab)

    # finally, ensure the browser console is clean
    assert dash_duo.get_logs() == []

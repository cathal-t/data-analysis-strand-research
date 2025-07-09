import pytest
from dash.testing.application_runners import import_app


@pytest.mark.parametrize("tab", ["overlay", "violin"])
def test_dash_smoke_and_add_row(dash_duo, tab):
    """
    • App boots with no console errors.
    • Click 'Add condition' once → table shows a second row named
      'Condition 2'.
    • Switch tab to ensure callbacks wired.
    """
    app = import_app("hairmech.ui.app")
    dash_duo.start_server(app)

    dash_duo.find_element("#btn-add").click()

    # Wait up to 5 s for the second-row first-column cell to read 'Condition 2'
    dash_duo.wait_for_text_to_equal("#cond-table .dash-cell.column-0.row-1",
                                    "Condition 2", timeout=5)

    dash_duo.select_dcc_dropdown("#tabs", tab)
    assert dash_duo.get_logs() == []

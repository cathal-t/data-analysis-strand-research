import pytest
from dash.testing.application_runners import import_app


@pytest.mark.parametrize("tab", ["overlay", "violin"])
def test_dash_smoke(dash_duo, tab):
    app = import_app("hairmech.ui.app")  # build_dash_app is executed
    dash_duo.start_server(app)
    dash_duo.find_element("#btn-add-row")  # element exists

    # Switch tabs simply to check callback registration
    dash_duo.select_dcc_dropdown("#tabs", tab)
    assert dash_duo.get_logs() == [], "browser console should contain no errors"

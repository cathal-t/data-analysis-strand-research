from hairmech.ui.multi_cassette import _merge_uploaded_files


def test_merge_uploaded_files_preserves_order_and_replaces_duplicates():
    existing = [
        {"name": "first_metrics.xlsx", "records": [{"Condition": "A"}]},
        {"name": "second_metrics.xlsx", "records": [{"Condition": "B"}]},
    ]

    new_files = [
        {"name": "third_metrics.xlsx", "records": [{"Condition": "C"}]},
        {"name": "second_metrics.xlsx", "records": [{"Condition": "B2"}]},
    ]

    merged = _merge_uploaded_files(existing, new_files)

    assert merged == [
        {"name": "first_metrics.xlsx", "records": [{"Condition": "A"}]},
        {"name": "second_metrics.xlsx", "records": [{"Condition": "B2"}]},
        {"name": "third_metrics.xlsx", "records": [{"Condition": "C"}]},
    ]

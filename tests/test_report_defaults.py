from datamining_app.data.preprocessor import ExcludeValidationError, identifier_headers, validate_exclude
from datamining_app.report_defaults import DEFAULT_TABLES, params_for


def test_report_defaults_follow_registry():
    assert DEFAULT_TABLES["apriori"] == "apriori_daily_basket_5tx"
    assert DEFAULT_TABLES["kmeans"] == "kmeans_2d_7pts"
    assert params_for("apriori", "apriori_standard_9tx")["minsup"] == 0.2222
    assert params_for("id3", "id3_weather_play")["decision_attr"] == "Play"
    assert params_for("kmeans", "kmeans_2d_7pts")["init"] == "random"


def test_validate_exclude_pid_ok():
    headers = ["pid", "x", "y"]
    rows = [{"pid": "1", "x": 1, "y": 2}, {"pid": "2", "x": 3, "y": 4}]
    validate_exclude(headers, rows, ["pid"], "kmeans", {"k": 2})


def test_validate_exclude_all_columns_fails():
    headers = ["pid", "x"]
    rows = [{"pid": "1", "x": 1}]
    try:
        validate_exclude(headers, rows, ["pid", "x"], "kmeans")
        assert False
    except ExcludeValidationError:
        pass


def test_validate_exclude_id3_needs_two_columns():
    headers = ["Outlook", "Play"]
    rows = [{"Outlook": "sunny", "Play": "No"}]
    try:
        validate_exclude(headers, rows, ["Outlook"], "id3", {"decision_attr": "Play"})
        assert False
    except ExcludeValidationError:
        pass


def test_validate_exclude_kmeans_needs_numeric():
    headers = ["pid", "name"]
    rows = [{"pid": "1", "name": "a"}]
    try:
        validate_exclude(headers, rows, ["pid"], "kmeans")
        assert False
    except ExcludeValidationError:
        pass


def test_identifier_headers():
    assert identifier_headers(["pid", "tid", "x", "RID"]) == ["pid", "tid", "RID"]

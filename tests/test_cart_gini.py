from datamining_app.algorithms.cart_gini import CARTGiniAlgorithm, gini_impurity
from tests.helpers import load_report_dataset, report_params


def _root_delta(result, feature: str) -> float:
    for step in result.steps:
        data = step.data or {}
        for row in data.get("gains") or []:
            if row.get("feature") == feature:
                return float(row["gain"])
    raise AssertionError(f"feature {feature} not found in gains")


def test_cart_weather_builds_tree(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    result = CARTGiniAlgorithm().run(ds, report_params("cart_gini", "id3_weather_play"))
    assert result.output["tree"]["attribute"]
    assert result.output["gini"] >= 0
    assert result.output["rules"]
    assert any("Gini" in s.title or "ΔGini" in s.title or "gini" in s.title.lower() for s in result.steps)


def test_cart_buy_computer(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_buy_computer")
    result = CARTGiniAlgorithm().run(ds, report_params("cart_gini", "id3_buy_computer"))
    assert result.output["tree"].get("is_leaf") or result.output["tree"].get("attribute")
    assert len(result.output["rules"]) >= 1


def test_cart_predict(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    algo = CARTGiniAlgorithm()
    algo.run(ds, report_params("cart_gini", "id3_weather_play"))
    pred = algo.predict({"Outlook": "sunny", "Temperature": "hot", "Humidity": "high", "Windy": "false"})
    assert pred.label.lower() in {"yes", "no"}


def test_gini_pure_is_zero():
    rows = [{"c": "yes"}, {"c": "yes"}, {"c": "yes"}]
    assert gini_impurity(rows, "c") == 0.0


def test_gini_balanced_two_class():
    rows = [{"c": "yes"}, {"c": "no"}]
    assert abs(gini_impurity(rows, "c") - 0.5) < 1e-9


def test_cart_pedagogical_gini_trace(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    result = CARTGiniAlgorithm().run(ds, report_params("cart_gini", "id3_weather_play"))
    split_step = next(s for s in result.steps if s.data.get("gains") and s.data.get("path") == "Gốc")
    headers = split_step.data["headers"]
    assert "Thuộc tính xem xét\nphân nhánh" in headers
    assert any("G(S)" in h for h in headers)
    assert any("G(A, S)" in h for h in headers)
    assert any("Gini Gain" in h for h in headers)
    assert len(split_step.data["rows"][0]) == 4
    ginis = {row[1] for row in split_step.data["rows"]}
    assert len(ginis) == 1
    desc = split_step.description
    assert "══ Thuộc tính:" in desc
    assert "G(" in desc
    assert "ΔGini(" in desc
    assert "★" in desc
    # No "1 - 0.25000 - 0.25000" or "(4/14) × 0.50000" intermediate expansion.
    assert "1 - 0." not in desc
    assert "× 0." not in desc
    from datamining_app.fmt import fmt_score

    for row in split_step.data["rows"]:
        g_s, g_as, delta_cell = row[1], row[2], row[3].replace(" ★", "")
        assert delta_cell == fmt_score(float(g_s) - float(g_as))


def test_cart_sub_dataset_steps(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    result = CARTGiniAlgorithm().run(ds, report_params("cart_gini", "id3_weather_play"))
    sub_steps = [s for s in result.steps if (s.data or {}).get("sub_dataset")]
    assert sub_steps
    assert any("Dataset for" in s.title for s in sub_steps)
    assert "ID" in sub_steps[0].data["headers"]
    split_step = next(s for s in result.steps if s.data.get("gains") and s.data.get("path") == "Gốc")
    assert "Bước 4" in split_step.description
    assert "Bước 5" in split_step.description
    assert "Bước 4–5" not in split_step.description

from datamining_app.algorithms.id3 import ID3Algorithm
from tests.helpers import load_report_dataset, report_params


def _root_gain(result, feature):
    step = next(s for s in result.steps if s.data.get("gains"))
    gains = {g["feature"]: g["gain"] for g in step.data["gains"]}
    return gains[feature]


def test_id3_buy_computer(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_buy_computer")
    algo = ID3Algorithm()
    result = algo.run(ds, report_params("id3", "id3_buy_computer"))
    assert result.output["tree"]["attribute"] == "age"
    assert abs(_root_gain(result, "age") - 0.246) < 0.02


def test_id3_weather_play(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    algo = ID3Algorithm()
    result = algo.run(ds, report_params("id3", "id3_weather_play"))
    assert result.output["tree"]["attribute"] == "Outlook"
    assert abs(_root_gain(result, "Outlook") - 0.246) < 0.02


def test_id3_binary_simple(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_binary_simple")
    result = ID3Algorithm().run(ds, report_params("id3", "id3_binary_simple"))
    root = result.output["tree"]["attribute"]
    assert root in {"Attr_A", "Attr_B"}
    assert _root_gain(result, "Attr_B") >= 0.4


def test_id3_predict_weather(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    algo = ID3Algorithm()
    algo.run(ds, report_params("id3", "id3_weather_play"))
    pred = algo.predict({"Outlook": "overcast", "Temperature": "hot", "Humidity": "high", "Windy": "false"})
    assert pred.label.lower() == "yes"


def test_id3_predict_buy_computer(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_buy_computer")
    algo = ID3Algorithm()
    algo.run(ds, report_params("id3", "id3_buy_computer"))
    pred = algo.predict({"age": "youth", "income": "medium", "student": "yes", "credit_rating": "fair"})
    assert pred.label.lower() == "yes"


def _real_leaves(node):
    if not node:
        return 0
    if node.get("pending"):
        return sum(_real_leaves(c) for c in (node.get("children") or {}).values())
    if node.get("is_leaf"):
        return 1
    return sum(_real_leaves(c) for c in (node.get("children") or {}).values())


def test_id3_tree_grows_step_by_step(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    result = ID3Algorithm().run(ds, report_params("id3", "id3_weather_play"))
    snaps = [s.data["tree"] for s in result.steps if s.data.get("tree")]
    assert snaps
    leaf_counts = [_real_leaves(t) for t in snaps]
    assert leaf_counts[0] == 0
    assert leaf_counts[-1] >= 2
    assert leaf_counts[-1] == _real_leaves(result.output["tree"])
    assert any(earlier < later for earlier, later in zip(leaf_counts, leaf_counts[1:]))


def test_id3_pedagogical_gain_trace(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    result = ID3Algorithm().run(ds, report_params("id3", "id3_weather_play"))
    gain_step = next(s for s in result.steps if s.data.get("gains") and s.data.get("path") == "Gốc")
    headers = gain_step.data["headers"]
    assert "Thuộc tính xem xét\nphân nhánh" in headers
    assert any("I(S)" in h for h in headers)
    assert any("E(A, S)" in h for h in headers)
    assert any("Information Gain" in h for h in headers)
    assert len(gain_step.data["rows"][0]) == 4
    ents = {row[1] for row in gain_step.data["rows"]}
    assert len(ents) == 1
    desc = gain_step.description
    assert "══ Thuộc tính:" in desc
    assert "I(" in desc
    assert "E(" in desc
    assert "Gain(" in desc
    assert "★" in desc
    # No intermediate decimal expansions that cause rounding mismatches.
    assert "× (" not in desc
    assert "× 0." not in desc  # no "(4/14) × 0.81128" substitution line
    from datamining_app.fmt import fmt_score

    for row in gain_step.data["rows"]:
        i_s, e_as, gain_cell = row[1], row[2], row[3].replace(" ★", "")
        assert gain_cell == fmt_score(float(i_s) - float(e_as))


def test_id3_sub_dataset_steps(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    result = ID3Algorithm().run(ds, report_params("id3", "id3_weather_play"))
    sub_steps = [s for s in result.steps if (s.data or {}).get("sub_dataset")]
    assert sub_steps
    assert any("Dataset for" in s.title for s in sub_steps)
    outlook_branch = next(
        s for s in sub_steps if "Outlook" in s.title and "Humidity" not in s.title
    )
    assert "ID" in outlook_branch.data["headers"]
    assert any(str(row[0]).isdigit() for row in outlook_branch.data["rows"])
    assert any("Nhãn" in h for h in outlook_branch.data["headers"])
    gain_step = next(s for s in result.steps if s.data.get("gains") and s.data.get("path") == "Gốc")
    assert "Bước 4" in gain_step.description
    assert "Bước 5" in gain_step.description
    assert "Bước 4–5" not in gain_step.description

from datamining_app.algorithms.naive_bayes import (
    ClassicNaiveBayesAlgorithm,
    LaplaceBayesAlgorithm,
)
from tests.helpers import load_report_dataset, report_params


def test_nb_classic_buy_computer(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_buy_computer")
    algo = ClassicNaiveBayesAlgorithm()
    result = algo.run(ds, report_params("naive_bayes", "id3_buy_computer"))
    assert abs(result.output["priors"]["yes"] - 9 / 14) < 1e-9
    assert result.output["smoothing"] == "none"
    assert "laplace_alpha" not in result.parameters
    pred = algo.predict({"age": "youth", "income": "medium", "student": "yes", "credit_rating": "fair"})
    assert pred.label.lower() in {"yes", "no"}
    assert pred.details["posteriors"]


def test_nb_laplace_buy_computer(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_buy_computer")
    algo = LaplaceBayesAlgorithm()
    result = algo.run(ds, report_params("naive_bayes_laplace", "id3_buy_computer"))
    assert abs(result.output["priors"]["yes"] - 9 / 14) < 1e-9
    assert result.parameters["laplace_alpha"] == 1.0
    pred = algo.predict({"age": "youth", "income": "medium", "student": "yes", "credit_rating": "fair"})
    assert pred.label.lower() == "yes"
    posts = pred.details["posteriors"]
    assert posts["yes"] > posts["no"]


def test_nb_laplace_weather_play(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    algo = LaplaceBayesAlgorithm()
    algo.run(ds, report_params("naive_bayes_laplace", "id3_weather_play"))
    pred = algo.predict({"Outlook": "sunny", "Temperature": "cool", "Humidity": "high", "Windy": "true"})
    assert pred.label.lower() == "no"


def test_nb_classic_buy_mobile(seeded_db):
    ds = load_report_dataset(seeded_db, "nb_buy_mobile")
    algo = ClassicNaiveBayesAlgorithm()
    algo.run(ds, report_params("naive_bayes", "nb_buy_mobile"))
    pred = algo.predict({"age": "<20", "income": "medium", "Region": "USA", "credit_rating": "High"})
    assert pred.label.lower() in {"yes", "no"}
    assert pred.details["posteriors"]


def test_nb_classic_likelihood_is_count_over_nc(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    result = ClassicNaiveBayesAlgorithm().run(ds, report_params("naive_bayes", "id3_weather_play"))
    # P(Outlook=sunny | No) classic = 3/5 = 0.6 (Play=No has 5 samples, 3 sunny)
    p = result.output["likelihoods"]["No"]["Outlook"]["sunny"]
    assert abs(p - 0.6) < 1e-9


def test_nb_predict_explanation_is_stepwise(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    algo = ClassicNaiveBayesAlgorithm()
    algo.run(ds, report_params("naive_bayes", "id3_weather_play"))
    pred = algo.predict(
        {"Outlook": "overcast", "Temperature": "cool", "Humidity": "high", "Windy": "false"}
    )
    text = pred.explanation
    assert "① Mẫu cần dự đoán:" in text
    assert "Outlook = overcast" in text
    assert "── Lớp No ──" in text
    assert "P(Outlook=overcast | Play=No) = 0/5" in text
    assert "bị loại" in text
    assert "④ Kết luận:" in text
    assert "Chuẩn hóa" not in text
    assert pred.label.lower() == "yes"
    assert "{'Outlook'" not in text


def test_nb_console_cells_embed_formula(seeded_db):
    from datamining_app.console.formatter import ConsoleFormatter

    ds = load_report_dataset(seeded_db, "id3_weather_play")
    classic = ClassicNaiveBayesAlgorithm().run(ds, report_params("naive_bayes", "id3_weather_play"))
    prior = classic.steps[0]
    assert prior.data["headers"] == ["Lớp", "Count", "P(C) = |D_C| / |D|"]
    assert any("P(Play=No) = 5/14" in str(cell) and "35.71%" in str(cell) for row in prior.data["rows"] for cell in row)
    like = next(s for s in classic.steps if s.data.get("feature") == "Outlook")
    assert like.data["headers"] == ["Lớp", "Giá trị", "Count", "P(Outlook=v | Play=C) = count / n_C"]
    assert "n_No = 5" in like.description
    assert "Công thức" not in like.data["headers"]
    assert any("P(Outlook=overcast | Play=No) = 0/5 =" in str(cell) for row in like.data["rows"] for cell in row)
    assert "⚠" in str(like.data.get("conclusion", ""))
    assert "①" in classic.summary and "②" in classic.summary and "③" in classic.summary
    assert "Zero-Frequency" in classic.summary or "tần suất bằng 0" in classic.summary
    assert "P(Play=C)" in classic.summary

    laplace = LaplaceBayesAlgorithm().run(
        ds, report_params("naive_bayes_laplace", "id3_weather_play")
    )
    like_l = next(s for s in laplace.steps if s.data.get("feature") == "Outlook")
    assert "|V| = 3" in like_l.description
    assert any(
        "P(Outlook=overcast | Play=No) = (0+1)/(5+1×3) = 1/8" in str(cell)
        for row in like_l.data["rows"]
        for cell in row
    )
    assert "Laplace" in laplace.summary
    assert "①" in laplace.summary and "không có Zero-Frequency" in laplace.summary
    rendered = ConsoleFormatter().render_result(classic)
    assert classic.summary.strip() in rendered
    assert "  → KẾT LUẬN" not in rendered

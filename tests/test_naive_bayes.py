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

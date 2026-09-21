from datamining_app.algorithms.naive_bayes import NaiveBayesAlgorithm
from tests.helpers import load_report_dataset, report_params


def test_nb_buy_computer(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_buy_computer")
    algo = NaiveBayesAlgorithm()
    result = algo.run(ds, report_params("naive_bayes", "id3_buy_computer"))
    assert abs(result.output["priors"]["yes"] - 9 / 14) < 1e-9
    pred = algo.predict({"age": "youth", "income": "medium", "student": "yes", "credit_rating": "fair"})
    assert pred.label.lower() == "yes"
    posts = pred.details["posteriors"]
    assert posts["yes"] > posts["no"]


def test_nb_weather_play(seeded_db):
    ds = load_report_dataset(seeded_db, "id3_weather_play")
    algo = NaiveBayesAlgorithm()
    algo.run(ds, report_params("naive_bayes", "id3_weather_play"))
    pred = algo.predict({"Outlook": "sunny", "Temperature": "cool", "Humidity": "high", "Windy": "true"})
    assert pred.label.lower() == "no"


def test_nb_buy_mobile(seeded_db):
    ds = load_report_dataset(seeded_db, "nb_buy_mobile")
    algo = NaiveBayesAlgorithm()
    algo.run(ds, report_params("naive_bayes", "nb_buy_mobile"))
    pred = algo.predict({"age": "<20", "income": "medium", "Region": "USA", "credit_rating": "High"})
    assert pred.label.lower() in {"yes", "no"}
    assert pred.details["posteriors"]

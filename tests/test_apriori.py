from datamining_app.algorithms.apriori import AprioriAlgorithm
from tests.helpers import load_report_dataset, report_params


def _itemsets(levels, k):
    return {tuple(x) for x in levels.get(str(k), [])}


def _run(seeded_db, table, algo="apriori"):
    ds = load_report_dataset(seeded_db, table)
    return AprioriAlgorithm().run(ds, report_params(algo, table))


def test_apriori_standard_9tx(seeded_db):
    levels = _run(seeded_db, "apriori_standard_9tx").output["levels"]
    assert _itemsets(levels, 1) == {("I1",), ("I2",), ("I3",), ("I4",), ("I5",)}


def test_apriori_grocery_4tx(seeded_db):
    levels = _run(seeded_db, "apriori_grocery_4tx").output["levels"]
    assert _itemsets(levels, 2)


def test_apriori_daily_basket_5tx(seeded_db):
    levels = _run(seeded_db, "apriori_daily_basket_5tx").output["levels"]
    l1 = {t[0] for t in _itemsets(levels, 1)}
    assert {"Bread", "Jam"} <= l1


def test_apriori_numeric_4tx(seeded_db):
    result = _run(seeded_db, "apriori_numeric_4tx")
    frequent = {tuple(sorted(s)) for level in result.output["levels"].values() for s in level}
    assert ("2", "3", "5") in frequent or {("2",), ("3",), ("5")} <= frequent


def test_apriori_grocery_25tx(seeded_db):
    levels = _run(seeded_db, "apriori_grocery_25tx").output["levels"]
    l1 = {t[0] for t in _itemsets(levels, 1)}
    assert "Bournvita" in l1


def test_apriori_butter_jam_5tx(seeded_db):
    result = _run(seeded_db, "apriori_butter_jam_5tx")
    frequent = {tuple(sorted(s)) for level in result.output["levels"].values() for s in level}
    assert ("Butter", "Jam") in frequent


def test_apriori_letters_4tx(seeded_db):
    levels = _run(seeded_db, "apriori_letters_4tx").output["levels"]
    assert ("B", "C", "E") in _itemsets(levels, 3)


def test_apriori_theory_8tx(seeded_db):
    result = _run(seeded_db, "apriori_theory_8tx")
    frequent = {tuple(sorted(s)) for level in result.output["levels"].values() for s in level}
    assert ("c", "d", "e") in frequent


def test_apriori_pharmacy_7tx(seeded_db):
    result = _run(seeded_db, "apriori_pharmacy_7tx")
    frequent = {tuple(sorted(s)) for level in result.output["levels"].values() for s in level}
    assert ("Aspirin", "VitaminC") in frequent


def test_apriori_math_5tx(seeded_db):
    levels = _run(seeded_db, "apriori_math_5tx").output["levels"]
    l2 = _itemsets(levels, 2)
    assert {("a", "b"), ("a", "c"), ("b", "c")} <= l2


def test_apriori_math_10tx(seeded_db):
    result = _run(seeded_db, "apriori_math_10tx")
    supports = result.output["supports"]
    ones = {k: v for k, v in supports.items() if k.count(",") == 0}
    top = max(ones, key=ones.get)
    assert "i2" in top

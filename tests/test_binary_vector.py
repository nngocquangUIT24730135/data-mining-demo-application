from datamining_app.algorithms.apriori import AprioriAlgorithm
from datamining_app.algorithms.binary_vector import BinaryVectorAlgorithm, mine_binary_vectors
from tests.helpers import load_report_dataset, report_params


def _sets(levels, k):
    return {tuple(x) for x in levels[str(k)]}


def test_binary_vector_daily_basket(seeded_db):
    ds = load_report_dataset(seeded_db, "apriori_daily_basket_5tx")
    params = report_params("binary_vector", "apriori_daily_basket_5tx")
    result = BinaryVectorAlgorithm().run(ds, params)
    l1 = {t[0] for t in _sets(result.output["levels"], 1)}
    assert {"Bread", "Jam"} <= l1
    from datamining_app.console.formatter import ConsoleFormatter

    text = "\n".join(ConsoleFormatter().render_step(s) for s in result.steps if s.data.get("k") == 2)
    assert "∧" in text


def test_binary_vector_matches_apriori_standard(seeded_db):
    table = "apriori_standard_9tx"
    ds = load_report_dataset(seeded_db, table)
    params = report_params("apriori", table)
    a = AprioriAlgorithm().run(ds, params)
    b = BinaryVectorAlgorithm().run(ds, params)
    assert a.output["levels"] == b.output["levels"]


def test_mine_binary_vectors_rejects_nonfrequent_and():
    item_vectors = {
        "A": [1, 1, 0, 0],
        "B": [1, 0, 1, 0],
        "C": [0, 0, 0, 1],
    }
    mined = mine_binary_vectors(item_vectors, n=4, minsup=0.5)
    assert mined.levels[1] == [("A",), ("B",)]
    assert 2 not in mined.levels
    assert mined.supports[("A",)] == 0.5

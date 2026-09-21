from datamining_app.algorithms.kmeans import KMeansAlgorithm
from tests.helpers import load_report_dataset, report_params


def _clusters(history_snap):
    return {i + 1: set(members) for i, members in enumerate(history_snap["clusters"])}


def test_kmeans_7pts_euclidean(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_7pts")
    algo = KMeansAlgorithm()
    result = algo.run(ds, report_params("kmeans", "kmeans_2d_7pts"))
    assert len(result.output["history"]) <= 5
    final = _clusters(result.output["history"][-1])
    assert final[1] == {"P1", "P2"}
    assert final[2] == {"P3", "P4", "P5", "P6", "P7"}


def test_kmeans_student_manhattan(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_student_scores")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_student_scores"))
    final = _clusters(result.output["history"][-1])
    groups = {frozenset(v) for v in final.values()}
    assert frozenset({"P1", "P5", "P6", "P7"}) in groups
    assert frozenset({"P2", "P8"}) in groups
    assert frozenset({"P3", "P4", "P9", "P10"}) in groups


def test_kmeans_age_1d_manhattan(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_1d_website_age")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_1d_website_age"))
    cents = sorted(c[0] for c in result.output["centroids"])
    assert abs(cents[0] - 19.5) < 1.0
    assert abs(cents[1] - 47.9) < 1.5


def test_kmeans_9pts_k3_euclidean(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_9pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_9pts"))
    clusters = result.output["history"][-1]["clusters"]
    assert len(clusters) == 3
    assert all(clusters)


def test_kmeans_4pts_simple(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_4pts_simple")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_4pts_simple"))
    assert len(result.output["history"]) <= 3


def test_kmeans_16pts_manual_init(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_16pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_16pts"))
    c1 = result.output["history"][0]["new_centroids"][0]
    assert abs(c1[0] - 4.6) < 0.6
    assert abs(c1[1] - 7.1) < 0.6


def test_kmeans_14pts_k3(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_14pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_14pts"))
    sses = [h["sse"] for h in result.output["history"]]
    assert sses[-1] <= sses[0] + 1e-6


def test_kmeans_euclidean_vs_manhattan(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_9pts")
    base = report_params("kmeans", "kmeans_2d_9pts")
    eu = KMeansAlgorithm().run(ds, {**base, "distance": "euclidean"})
    man = KMeansAlgorithm().run(ds, {**base, "distance": "manhattan"})
    assert eu.output["history"][-1]["clusters"] != man.output["history"][-1]["clusters"] or (
        eu.output["centroids"] != man.output["centroids"]
    )

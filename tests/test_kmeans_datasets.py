from datamining_app.algorithms.kmeans import KMeansAlgorithm, MAX_ITER_CAP, MAX_POINTS
from tests.helpers import load_report_dataset, report_params


def _clusters(history_snap):
    return {i + 1: set(members) for i, members in enumerate(history_snap["clusters"])}


def test_kmeans_7pts_euclidean(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_7pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_7pts"))
    assert result.parameters["init"] == "random"
    assert len(result.output["history"]) <= result.parameters["max_iter"]
    assert len(result.output["history"]) <= MAX_ITER_CAP
    final = _clusters(result.output["history"][-1])
    assert len(final) == 2
    assert sum(len(v) for v in final.values()) == 7
    assert all(final.values())


def test_kmeans_student_manhattan(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_student_scores")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_student_scores"))
    final = _clusters(result.output["history"][-1])
    assert len(final) == 3
    assert sum(len(v) for v in final.values()) == 10


def test_kmeans_age_1d_manhattan(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_1d_website_age")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_1d_website_age"))
    cents = sorted(c[0] for c in result.output["centroids"])
    assert len(cents) == 2
    assert cents[0] < cents[1]
    assert len(result.output["history"]) <= result.parameters["max_iter"]


def test_kmeans_9pts_k3_euclidean(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_9pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_9pts"))
    clusters = result.output["history"][-1]["clusters"]
    assert len(clusters) == 3
    assert all(clusters)


def test_kmeans_4pts_simple(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_4pts_simple")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_4pts_simple"))
    assert len(result.output["history"]) <= result.parameters["max_iter"]
    assert len(result.output["points"]) == 4


def test_kmeans_16pts_terminates(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_16pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_16pts"))
    assert len(result.output["history"]) <= result.parameters["max_iter"] <= MAX_ITER_CAP
    assert len(result.output["centroids"]) == 3
    assert "converged" in result.output


def test_kmeans_14pts_k3(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_14pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_14pts"))
    sses = [h["sse"] for h in result.output["history"]]
    assert len(sses) >= 1
    assert len(sses) <= result.parameters["max_iter"]


def test_kmeans_inertia_label_euclidean_vs_manhattan(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_9pts")
    base = report_params("kmeans", "kmeans_2d_9pts")
    eu = KMeansAlgorithm().run(ds, {**base, "distance": "euclidean"})
    man = KMeansAlgorithm().run(ds, {**base, "distance": "manhattan"})
    assert eu.output["inertia_label"] == "SSE"
    assert man.output["inertia_label"] == "SAE"
    assert all(h["inertia_label"] == "SSE" for h in eu.output["history"])
    assert all(h["inertia_label"] == "SAE" for h in man.output["history"])
    assert any("Bước 2a" in s.title for s in eu.steps)
    assert any("Bước 2b" in s.title for s in eu.steps)
    assert any("SSE" in (s.data or {}).get("conclusion", "") for s in eu.steps if "2b" in s.title)
    assert any("SAE" in (s.data or {}).get("conclusion", "") for s in man.steps if "2b" in s.title)


def test_kmeans_console_matches_pseudocode_wording(seeded_db):
    from datamining_app.console.formatter import ConsoleFormatter

    ds = load_report_dataset(seeded_db, "kmeans_2d_7pts")
    result = KMeansAlgorithm().run(ds, report_params("kmeans", "kmeans_2d_7pts"))
    fmt = ConsoleFormatter()
    text = fmt.render_result(result, "kmeans_2d_7pts", len(ds.rows))

    assert "① Ý tưởng cốt lõi của K-Means" in text
    assert "Tiêu chí đánh giá chất lượng cụm" in text or "Tiêu chí chất lượng" in text
    assert "[Mã giả Bước 2a]" in text
    assert " MÃ GIẢ " in text
    assert text.index(" MÃ GIẢ ") < text.index("Khởi tạo")
    assert "[Mã giả Bước 2b]" in text
    assert "Vòng lặp 1 — Bước 2a: Gán cụm" in text
    assert "Vòng lặp 1 — Bước 2b: Cập nhật trọng tâm" in text
    assert "Bước 2: Vòng lặp" not in text
    assert "Bước 3: Vòng lặp" not in text
    assert any(
        s.title.startswith("Kết thúc — Bước 2c") or s.title.startswith("Kết thúc — Bước 2d")
        for s in result.steps
    )
    # Explain-once: full ① nghĩa only on first assign, not later
    assign_descs = [s.description for s in result.steps if "Bước 2a" in s.title]
    assert any("① Ý nghĩa" in d for d in assign_descs[:1])
    if len(assign_descs) > 1:
        assert all("① Ý nghĩa" not in d for d in assign_descs[1:])


def test_kmeans_euclidean_vs_manhattan(seeded_db):
    ds = load_report_dataset(seeded_db, "kmeans_2d_9pts")
    base = report_params("kmeans", "kmeans_2d_9pts")
    eu = KMeansAlgorithm().run(ds, {**base, "distance": "euclidean"})
    man = KMeansAlgorithm().run(ds, {**base, "distance": "manhattan"})
    assert eu.parameters["distance"] == "euclidean"
    assert man.parameters["distance"] == "manhattan"
    assert len(eu.output["history"]) <= MAX_ITER_CAP
    assert len(man.output["history"]) <= MAX_ITER_CAP


def test_kmeans_clamps_max_iter():
    from datamining_app.core.models import Dataset

    ds = Dataset(
        name="tiny",
        headers=["x", "y"],
        rows=[{"x": i, "y": i} for i in range(6)],
        source="test",
    )
    result = KMeansAlgorithm().run(ds, {"k": 2, "max_iter": 9999, "distance": "euclidean"})
    assert result.parameters["max_iter"] == MAX_ITER_CAP
    assert len(result.output["history"]) <= MAX_ITER_CAP
    assert len(result.output["points"]) <= MAX_POINTS

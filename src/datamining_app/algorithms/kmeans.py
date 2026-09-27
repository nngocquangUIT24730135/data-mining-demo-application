from __future__ import annotations

import random
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.algorithms.dataset_utils import POINT_ID_KEYS, excluded_headers, to_float
from datamining_app.steps.kmeans_steps import _KMeansSteps, build_predict_explanation
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult
from datamining_app.fmt import fmt_inertia

# Hard caps so the desktop demo always terminates in reasonable time.
MAX_K = 20
MAX_ITER_CAP = 50
MAX_POINTS = 500
# Fixed seed: random centroid init must be identical every time the demo runs.
_INTERNAL_SEED = 42


class KMeansAlgorithm(BaseAlgorithm):
    name = "K-Means"
    description = "Phân cụm K-Means với Euclid hoặc Manhattan."
    supports_predict = True
    supports_visualization = True
    param_schema = {
        "k": ParamDef(
            type="int",
            default=3,
            min=2,
            max=MAX_K,
            step=1,
            label_vi="Số cụm (K)",
            label_en="Number of clusters (K)",
        ),
        "max_iter": ParamDef(
            type="int",
            default=20,
            min=1,
            max=MAX_ITER_CAP,
            step=1,
            label_vi="Số vòng lặp tối đa",
            label_en="Maximum iterations",
        ),
        "distance": ParamDef(
            type="radio",
            options=["euclidean", "manhattan"],
            default="euclidean",
            label_vi="Độ đo khoảng cách",
            label_en="Distance metric",
        ),
        "init": ParamDef(
            type="radio",
            options=["random"],
            default="random",
            label_vi="Khởi tạo trọng tâm",
            label_en="Centroid initialization",
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._centroids: list[list[float]] = []
        self._features: list[str] = []
        self._distance: str = "euclidean"
        self._points: list[dict[str, Any]] = []

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        headers = excluded_headers(dataset, params)
        features = [
            h for h in headers if all(to_float(v) is not None for v in dataset.column_values(h))
        ]
        if len(features) < 1:
            raise ValueError("K-Means cần ít nhất một cột số.")

        k = max(2, min(int(params.get("k", 3)), MAX_K))
        max_iter = max(1, min(int(params.get("max_iter", 20)), MAX_ITER_CAP))
        metric = str(params.get("distance", "euclidean")).strip().lower()
        if metric not in {"euclidean", "manhattan"}:
            metric = "euclidean"
        init_mode = "random"

        points = []
        for i, row in enumerate(dataset.rows):
            if len(points) >= MAX_POINTS:
                break
            vec = [to_float(row.get(f)) for f in features]
            if any(v is None for v in vec):
                continue
            label = _point_id(row, i)
            points.append({"id": label, "vector": vec, "row": row})
        if len(points) < k:
            raise ValueError(f"Cần ít nhất K={k} điểm dữ liệu (có {len(points)} điểm hợp lệ).")
        k = min(k, len(points))

        rng = random.Random(_INTERNAL_SEED)
        centroids, init_ids = _initial_centroids(points, k, rng)

        log = self._new_logger()
        formula = (
            "d(x,y) = √(Σ (xᵢ-yᵢ)²)" if metric == "euclidean" else "d(x,y) = Σ |xᵢ-yᵢ|"
        )
        steps = _KMeansSteps(log)
        steps.init_step(k, metric, max_iter, features, centroids, formula, init_ids)

        history = []
        assignments: list[int] = []
        inertia_prev = None
        converged = False
        for iteration in range(1, max_iter + 1):
            assignments = []
            clusters: list[list[int]] = [[] for _ in range(k)]
            dist_rows = []
            for idx, point in enumerate(points):
                distances = [_distance(point["vector"], c, metric) for c in centroids]
                cluster = min(range(k), key=lambda j: (distances[j], j))
                assignments.append(cluster)
                clusters[cluster].append(idx)
                dist_rows.append(
                    {"id": point["id"], "distances": distances, "cluster": cluster + 1}
                )

            inertia, inertia_label = _compute_inertia(points, centroids, assignments, metric)

            new_centroids = []
            for j in range(k):
                members = [points[i]["vector"] for i in clusters[j]]
                if members:
                    dim = len(members[0])
                    new_centroids.append(
                        [sum(m[d] for m in members) / len(members) for d in range(dim)]
                    )
                else:
                    new_centroids.append(list(centroids[j]))

            prev_assign = history[-1]["assignments"] if history else None
            changed = []
            if prev_assign:
                for idx, cluster in enumerate(assignments):
                    old_c = prev_assign[idx]
                    new_c = cluster + 1
                    if old_c != new_c:
                        changed.append(f"{points[idx]['id']} (C{old_c} → C{new_c})")
            snap = {
                "iteration": iteration,
                "centroids": [list(c) for c in centroids],
                "new_centroids": [list(c) for c in new_centroids],
                "assignments": [a + 1 for a in assignments],
                "sse": inertia,
                "inertia": inertia,
                "inertia_label": inertia_label,
                "clusters": [[points[i]["id"] for i in members] for members in clusters],
                "distances": dist_rows,
                "changed": changed,
            }
            history.append(snap)
            steps.assign_step(iteration, snap, points, k)
            steps.update_step(iteration, snap, centroids, new_centroids, inertia, inertia_label)

            same_centroids = _same_centroids(centroids, new_centroids)
            centroids = new_centroids
            if inertia_prev is not None and inertia > inertia_prev + 1e-9:
                steps.inertia_warning_step(iteration, inertia, inertia_prev, inertia_label)
            inertia_prev = inertia
            if same_centroids or not changed and iteration > 1:
                converged = True
                steps.convergence_step(iteration, inertia, inertia_label, k)
                break

        final_inertia = history[-1]["inertia"] if history else 0.0
        final_label = history[-1]["inertia_label"] if history else "SSE"
        if not converged:
            steps.max_iter_stop_step(max_iter, final_inertia, final_label)

        self._centroids = centroids
        self._features = features
        self._distance = metric
        self._points = points
        for i, point in enumerate(points):
            point["cluster"] = assignments[i] + 1

        summary = (
            f"K-Means k={k}, {metric}, {len(history)} vòng"
            f"{' (hội tụ)' if converged else f' (dừng tại max_iter={max_iter})'}. "
            f"{final_label} cuối = {fmt_inertia(final_inertia)}."
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={
                "k": k,
                "max_iter": max_iter,
                "distance": metric,
                "init": init_mode,
            },
            steps=log.steps,
            output={
                "centroids": centroids,
                "features": features,
                "points": [
                    {
                        "id": p["id"],
                        "vector": p["vector"],
                        "cluster": p["cluster"],
                    }
                    for p in points
                ],
                "history": history,
                "sse": final_inertia,
                "inertia": final_inertia,
                "inertia_label": final_label,
                "converged": converged,
            },
            summary=summary,
        )
        return self._finish(result)

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        if not self._trained:
            raise ValueError("Chưa huấn luyện K-Means.")
        vec = []
        for feat in self._features:
            value = to_float(sample.get(feat))
            if value is None:
                raise ValueError(f"Thuộc tính '{feat}' phải là số.")
            vec.append(value)
        distances = [_distance(vec, c, self._distance) for c in self._centroids]
        cluster = min(range(len(distances)), key=lambda j: distances[j]) + 1
        return PredictResult(
            label=f"C{cluster}",
            explanation=build_predict_explanation(self._features, vec, self._distance, distances, cluster),
            details={"distances": distances, "cluster": cluster, "vector": vec},
            sample=sample,
        )


def _compute_inertia(
    points: list[dict[str, Any]],
    centroids: list[list[float]],
    assignments: list[int],
    metric: str,
) -> tuple[float, str]:
    """Return (value, label): SSE for Euclidean, SAE for Manhattan."""
    total = 0.0
    if metric == "euclidean":
        for idx, point in enumerate(points):
            total += _distance(point["vector"], centroids[assignments[idx]], metric) ** 2
        return total, "SSE"
    for idx, point in enumerate(points):
        total += _distance(point["vector"], centroids[assignments[idx]], metric)
    return total, "SAE"


def _initial_centroids(
    points: list[dict[str, Any]],
    k: int,
    rng: random.Random,
) -> tuple[list[list[float]], str]:
    chosen = rng.sample(points, k)
    centroids = [list(p["vector"]) for p in chosen]
    ids = ", ".join(p["id"] for p in chosen)
    return centroids, ids


def _point_id(row: dict[str, Any], index: int) -> str:
    for key, value in row.items():
        if key.lower() in POINT_ID_KEYS and str(value).strip():
            return str(value).strip()
    return f"P{index + 1}"


def _distance(a: list[float], b: list[float], metric: str) -> float:
    if metric == "manhattan":
        return sum(abs(x - y) for x, y in zip(a, b))
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _same_centroids(a: list[list[float]], b: list[list[float]], tol: float = 1e-9) -> bool:
    for ca, cb in zip(a, b):
        if any(abs(x - y) > tol for x, y in zip(ca, cb)):
            return False
    return True

__all__ = ["KMeansAlgorithm", "MAX_ITER_CAP", "MAX_K", "MAX_POINTS"]

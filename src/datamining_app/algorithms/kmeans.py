from __future__ import annotations

import random
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.algorithms.dataset_utils import excluded_headers, to_float
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult


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
            max=20,
            step=1,
            label_vi="Số cụm (K)",
            label_en="Number of clusters (K)",
        ),
        "max_iter": ParamDef(
            type="int",
            default=20,
            min=1,
            max=100,
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
            options=["random", "first_k", "manual"],
            default="first_k",
            label_vi="Khởi tạo trọng tâm",
            label_en="Centroid initialization",
        ),
        "feature_cols": ParamDef(
            type="multiselect",
            options="headers",
            default=[],
            label_vi="Cột đặc trưng số (để trống = tự chọn)",
            label_en="Numeric feature columns (empty = auto)",
        ),
        "random_seed": ParamDef(
            type="int",
            default=42,
            min=0,
            max=9999,
            step=1,
            label_vi="Hạt giống ngẫu nhiên (init)",
            label_en="Random seed (init)",
        ),
        "init_ids": ParamDef(
            type="str",
            default="",
            label_vi="Tâm theo id (phẩy; dùng với init=manual)",
            label_en="Centroid ids (comma; with init=manual)",
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
        requested = [c for c in (params.get("feature_cols") or []) if c in headers]
        features = requested or [
            h for h in headers if all(to_float(v) is not None for v in dataset.column_values(h))
        ]
        if len(features) < 1:
            raise ValueError("K-Means cần ít nhất một cột số.")
        k = int(params.get("k", 3))
        max_iter = int(params.get("max_iter", 20))
        metric = str(params.get("distance", "euclidean"))
        seed = int(params.get("random_seed", 42))

        points = []
        for i, row in enumerate(dataset.rows):
            vec = [to_float(row.get(f)) for f in features]
            if any(v is None for v in vec):
                continue
            label = _point_id(row, i)
            points.append({"id": label, "vector": vec, "row": row})
        if len(points) < k:
            raise ValueError(f"Cần ít nhất K={k} điểm dữ liệu.")

        rng = random.Random(seed)
        centroids, init_note = _initial_centroids(points, k, params, rng)

        log = self._new_logger()
        formula = (
            "d(x,y) = √(Σ (xᵢ-yᵢ)²)" if metric == "euclidean" else "d(x,y) = Σ |xᵢ-yᵢ|"
        )
        init_headers = ["Tâm"] + features
        init_rows = [
            [f"m{i}"] + [f"{v:.3f}" for v in c]
            for i, c in enumerate(centroids, start=1)
        ]
        log.add(
            "Khởi tạo",
            f"K = {k}, metric = {metric}, max_iter = {max_iter}, seed = {seed}\n"
            f"Đặc trưng: {', '.join(features)}\n"
            f"{formula}\n"
            f"{init_note}",
            {
                "k": k,
                "centroids": [list(c) for c in centroids],
                "features": features,
                "metric": metric,
                "headers": init_headers,
                "rows": init_rows,
                "alignments": ["left"] + ["right"] * len(features),
                "phase": "init",
            },
            "STEP_HEADER",
        )

        history = []
        assignments: list[int] = [0] * len(points)
        sse_prev = None
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

            sse = 0.0
            for idx, point in enumerate(points):
                sse += _distance(point["vector"], centroids[assignments[idx]], metric) ** (
                    2 if metric == "euclidean" else 1
                )

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
                        changed.append(f"{points[idx]['id']}: C{old_c} → C{new_c}")
            snap = {
                "iteration": iteration,
                "centroids": [list(c) for c in centroids],
                "new_centroids": [list(c) for c in new_centroids],
                "assignments": [a + 1 for a in assignments],
                "sse": sse,
                "clusters": [[points[i]["id"] for i in members] for members in clusters],
                "distances": dist_rows,
                "changed": changed,
            }
            history.append(snap)
            assign_headers = ["Điểm"] + [f"d(C{j})" for j in range(1, k + 1)] + ["Cụm"]
            assign_rows = [
                [row["id"], *[f"{d:.3f}" for d in row["distances"]], f"C{row['cluster']}"]
                for row in dist_rows
            ]
            cluster_lines = []
            for i, members in enumerate(clusters, start=1):
                ids = ", ".join(points[j]["id"] for j in members) or "∅"
                cluster_lines.append(f"  → Cụm {i}: {{ {ids} }}")
            if changed:
                cluster_lines.append("  ✦ Đổi cụm: " + "; ".join(changed))
            assign_snap = {**snap, "phase": "assign", "headers": assign_headers, "rows": assign_rows,
                           "alignments": ["left"] + ["right"] * k + ["center"],
                           "conclusion": "\n".join(cluster_lines)}
            update_headers = ["Cụm", "Tâm cũ", "Tâm mới"]
            update_rows = []
            for i, (old, new) in enumerate(zip(centroids, new_centroids), start=1):
                update_rows.append(
                    [
                        f"C{i}",
                        "(" + ", ".join(f"{v:.3f}" for v in old) + ")",
                        "(" + ", ".join(f"{v:.3f}" for v in new) + ")",
                    ]
                )
            update_snap = {
                **snap,
                "phase": "update",
                "headers": update_headers,
                "rows": update_rows,
                "alignments": ["left", "left", "left"],
                "conclusion": f"  → SSE = {sse:.4f}",
            }
            log.add(
                f"Iteration {iteration} — Bước Gán (Assignment)",
                f"d(P, C) theo {metric}",
                assign_snap,
                "SUCCESS",
            )
            log.add(
                f"Iteration {iteration} — Bước Cập nhật Trọng tâm (Update)",
                "",
                update_snap,
                "SUCCESS",
            )

            converged = _same_centroids(centroids, new_centroids)
            centroids = new_centroids
            if sse_prev is not None and sse > sse_prev + 1e-9:
                log.add(
                    "Cảnh báo SSE",
                    f"SSE tăng từ {sse_prev:.4f} lên {sse:.4f}.",
                    {"iteration": iteration, "sse": sse, "sse_prev": sse_prev, "phase": "warning"},
                    "WARNING",
                )
            sse_prev = sse
            if converged:
                log.add(
                    "Hội tụ",
                    f"Tâm không đổi sau vòng {iteration}. SSE = {sse:.4f}.",
                    {"iteration": iteration, "sse": sse, "phase": "converged"},
                    "SUCCESS",
                )
                break

        self._centroids = centroids
        self._features = features
        self._distance = metric
        self._points = points
        self._trained = True
        for i, point in enumerate(points):
            point["cluster"] = assignments[i] + 1

        summary = (
            f"K-Means k={k}, {metric}, {len(history)} vòng. "
            f"SSE cuối = {history[-1]['sse']:.4f}."
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={
                "k": k,
                "max_iter": max_iter,
                "distance": metric,
                "init": str(params.get("init") or "first_k"),
                "random_seed": seed,
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
                "sse": history[-1]["sse"],
            },
            summary=summary,
        )
        self._last_result = result
        return result

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
        lines = [
            f"Điểm { {f: v for f, v in zip(self._features, vec)} }",
            f"Độ đo: {self._distance}",
        ]
        for i, dist in enumerate(distances, start=1):
            mark = " ← gần nhất" if i == cluster else ""
            lines.append(f"  d(C{i}) = {dist:.4f}{mark}")
        lines.append(f"⇒ Gán cụm C{cluster}")
        return PredictResult(
            label=f"C{cluster}",
            explanation="\n".join(lines),
            details={"distances": distances, "cluster": cluster, "vector": vec},
            sample=sample,
        )


def _initial_centroids(
    points: list[dict[str, Any]],
    k: int,
    params: dict[str, Any],
    rng: random.Random,
) -> tuple[list[list[float]], str]:
    init_mode = str(params.get("init") or "").strip().lower()
    init_ids = [s.strip() for s in str(params.get("init_ids") or "").split(",") if s.strip()]
    centroids_param = params.get("centroids")
    if isinstance(centroids_param, str) and centroids_param.strip():
        parsed = []
        for chunk in centroids_param.replace(";", "|").split("|"):
            nums = [to_float(p) for p in chunk.replace(",", " ").split() if p.strip()]
            if nums and all(n is not None for n in nums):
                parsed.append([float(n) for n in nums])
        centroids_param = parsed or None
    if centroids_param and not init_mode:
        init_mode = "manual"
    if init_ids and not init_mode:
        init_mode = "manual"
    if not init_mode:
        init_mode = "random"

    if init_mode == "first_k":
        centroids = [list(p["vector"]) for p in points[:k]]
        ids = ", ".join(p["id"] for p in points[:k])
        return centroids, f"Tâm ban đầu (first_k: {ids}):"
    if init_mode == "manual" and centroids_param:
        if len(centroids_param) != k:
            raise ValueError(f"centroids có {len(centroids_param)} tâm nhưng K = {k}.")
        centroids = [list(map(float, c)) for c in centroids_param]
        return centroids, "Tâm ban đầu (manual):"
    if init_ids:
        by_id = {p["id"]: p for p in points}
        missing = [pid for pid in init_ids if pid not in by_id]
        if missing:
            raise ValueError(f"Không tìm thấy id khởi tạo: {', '.join(missing)}")
        if len(init_ids) != k:
            raise ValueError(f"init_ids có {len(init_ids)} tâm nhưng K = {k}.")
        centroids = [list(by_id[pid]["vector"]) for pid in init_ids]
        return centroids, f"Tâm ban đầu (theo id {', '.join(init_ids)}):"
    centroids = [list(p["vector"]) for p in rng.sample(points, k)]
    return centroids, "Tâm ban đầu (ngẫu nhiên):"


def _point_id(row: dict[str, Any], index: int) -> str:
    for key, value in row.items():
        if key.lower() in {
            "id",
            "tid",
            "pid",
            "rid",
            "u",
            "point",
            "point id",
            "student id",
            "object",
            "instance",
        } and str(value).strip():
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

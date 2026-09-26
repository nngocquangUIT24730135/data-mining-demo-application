from __future__ import annotations

import random
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm, StepLogger
from datamining_app.algorithms.dataset_utils import excluded_headers, to_float
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult
from datamining_app.fmt import fmt_centroid, fmt_dist, fmt_inertia

# Hard caps so the desktop demo always terminates in reasonable time.
MAX_K = 20
MAX_ITER_CAP = 50
MAX_POINTS = 500
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
        lines = [
            f"Điểm { {f: v for f, v in zip(self._features, vec)} }",
            f"Độ đo: {self._distance}",
        ]
        for i, dist in enumerate(distances, start=1):
            mark = " ← gần nhất" if i == cluster else ""
            lines.append(f"  d(C{i}) = {fmt_dist(dist)}{mark}")
        lines.append(f"⇒ Gán cụm C{cluster}")
        return PredictResult(
            label=f"C{cluster}",
            explanation="\n".join(lines),
            details={"distances": distances, "cluster": cluster, "vector": vec},
            sample=sample,
        )


class _KMeansSteps:
    def __init__(self, log: StepLogger) -> None:
        self._log = log
        self._metric = "euclidean"
        self._k = 0
        self._first_inertia: float | None = None

    def init_step(
        self,
        k: int,
        metric: str,
        max_iter: int,
        features: list[str],
        centroids: list[list[float]],
        formula: str,
        init_ids: str,
    ) -> None:
        self._metric = metric
        self._k = k
        init_headers = ["Tâm"] + features
        init_rows = [
            [f"m{i}"] + [fmt_dist(v) for v in c]
            for i, c in enumerate(centroids, start=1)
        ]
        metric_name = "Euclidean" if metric == "euclidean" else "Manhattan"
        if metric == "euclidean":
            metric_block = (
                f"② Độ đo khoảng cách — {metric_name} (đã chọn):\n"
                f"  {formula}\n"
                '  → Đo "đường chim bay" giữa 2 điểm trong không gian.\n'
                "  → Nhạy cảm với outlier hơn Manhattan.\n"
                "  [So sánh] Manhattan: d(x,y) = Σ|xᵢ-yᵢ|  (\"đường taxi\")"
            )
            criteria_block = (
                "③ Tiêu chí đánh giá chất lượng cụm — SSE:\n"
                "  SSE (Sum of Squared Errors) = Σᵢ d(xᵢ, μ_cụm(i))²\n"
                "  → Tổng bình phương khoảng cách mỗi điểm đến trọng tâm cụm.\n"
                "  → SSE càng nhỏ → cụm càng chặt chẽ → kết quả càng tốt.\n"
                "  [Lưu ý] Nếu dùng Manhattan → gọi là SAE, không bình phương."
            )
        else:
            metric_block = (
                f"② Độ đo khoảng cách — {metric_name} (đã chọn):\n"
                f"  {formula}\n"
                '  → Đo "đường taxi" (cạnh vuông góc) giữa 2 điểm.\n'
                "  → Ít nhạy cảm với outlier hơn Euclidean.\n"
                "  [So sánh] Euclidean: d(x,y) = √(Σ (xᵢ-yᵢ)²)  (\"đường chim bay\")"
            )
            criteria_block = (
                "③ Tiêu chí đánh giá chất lượng cụm — SAE:\n"
                "  SAE (Sum of Absolute Errors) = Σᵢ d(xᵢ, μ_cụm(i))\n"
                "  → Tổng khoảng cách tuyệt đối mỗi điểm đến trọng tâm cụm.\n"
                "  → SAE càng nhỏ → cụm càng chặt chẽ → kết quả càng tốt.\n"
                "  [Lưu ý] Nếu dùng Euclidean → gọi là SSE, có bình phương."
            )
        description = (
            "① Ý tưởng cốt lõi của K-Means:\n"
            "  Chia N điểm dữ liệu thành K cụm sao cho các điểm trong\n"
            "  cùng cụm càng gần nhau càng tốt (nội cụm gắn kết,\n"
            "  liên cụm tách biệt).\n"
            f"\n{metric_block}\n"
            f"\n{criteria_block}\n"
            f"\n④ Khởi tạo trọng tâm (μ):\n"
            f"  Chọn ngẫu nhiên K={k} điểm từ dữ liệu làm trọng tâm ban đầu.\n"
            f"  → Chọn: {init_ids}\n"
            f"  Đặc trưng số: {', '.join(features)} | max_iter = {max_iter}"
        )
        self._log.add_table(
            "Khởi tạo",
            init_headers,
            init_rows,
            ["left"] + ["right"] * len(features),
            description=description,
            level="STEP_HEADER",
            k=k,
            centroids=[list(c) for c in centroids],
            features=features,
            metric=metric,
            phase="init",
        )

    def assign_step(
        self,
        iteration: int,
        snap: dict[str, Any],
        points: list[dict[str, Any]],
        k: int,
    ) -> None:
        dist_rows = snap["distances"]
        assign_headers = ["Điểm"] + [f"d(C{j})" for j in range(1, k + 1)] + ["Cụm"]
        assign_rows = [
            [row["id"], *[fmt_dist(d) for d in row["distances"]], f"C{row['cluster']}"]
            for row in dist_rows
        ]
        cluster_lines = ["Kết quả phân cụm:"]
        for i in range(1, k + 1):
            members = [
                point["id"]
                for point, assigned in zip(points, snap["assignments"])
                if assigned == i
            ]
            ids = ", ".join(members) or "∅"
            cluster_lines.append(f"  → Cụm {i}: {{ {ids} }}   ({len(members)} điểm)")
        if iteration == 1:
            cluster_lines.append("  [Vòng đầu tiên — chưa có so sánh với vòng trước]")
        elif snap["changed"]:
            cluster_lines.append(
                "  ✦ Điểm thay đổi cụm so với vòng trước: " + "; ".join(snap["changed"])
            )
        if iteration == 1:
            description = (
                "[Mã giả Bước 2a] cluster(xᵢ) ← argminⱼ d(xᵢ, μⱼ)\n"
                "\n"
                "① Ý nghĩa: Mỗi điểm xᵢ được gán vào cụm có trọng tâm\n"
                '  gần nhất. "argmin" = lấy chỉ số j cho d nhỏ nhất.'
            )
        else:
            description = "[Mã giả Bước 2a] cluster(xᵢ) ← argminⱼ d(xᵢ, μⱼ)"
        assign_snap = {
            **snap,
            "phase": "assign",
            "headers": assign_headers,
            "rows": assign_rows,
            "alignments": ["left"] + ["right"] * k + ["center"],
            "conclusion": "\n".join(cluster_lines),
            "omit_step_number": True,
        }
        self._log.add(
            f"Vòng lặp {iteration} — Bước 2a: Gán cụm",
            description,
            assign_snap,
            "SUCCESS",
        )

    def update_step(
        self,
        iteration: int,
        snap: dict[str, Any],
        centroids: list[list[float]],
        new_centroids: list[list[float]],
        inertia: float,
        inertia_label: str,
    ) -> None:
        if self._first_inertia is None:
            self._first_inertia = inertia
        k = len(centroids)
        update_headers = ["Cụm", "Tâm cũ", "Tâm mới"]
        update_rows = []
        diff_lines = ["So sánh tâm cũ — tâm mới:"]
        n_changed = 0
        for i, (old, new) in enumerate(zip(centroids, new_centroids), start=1):
            old_txt = fmt_centroid(old)
            new_txt = fmt_centroid(new)
            update_rows.append([f"C{i}", old_txt, new_txt])
            changed = any(abs(a - b) > 1e-9 for a, b in zip(old, new))
            if changed:
                n_changed += 1
            marker = "← thay đổi" if changed else "← không đổi"
            diff_lines.append(f"  C{i}: {old_txt} → {new_txt}  {marker}")
        if inertia_label == "SSE":
            inertia_hint = f"{inertia_label} = Σ d(xᵢ, μ_cụm)² = tổng bình phương khoảng cách"
        else:
            inertia_hint = f"{inertia_label} = Σ d(xᵢ, μ_cụm) = tổng khoảng cách tuyệt đối"
        if n_changed:
            verdict = (
                f"[Mã giả Bước 2c] Kiểm tra hội tụ:\n"
                f"→ Có {n_changed}/{k} tâm thay đổi ⟹ CHƯA hội tụ, tiếp tục vòng {iteration + 1}"
            )
        else:
            verdict = (
                "[Mã giả Bước 2c] Kiểm tra hội tụ:\n"
                f"→ Tất cả {k} tâm không đổi ⟹ hội tụ"
            )
        conclusion = (
            "\n".join(diff_lines)
            + f"\n\n{inertia_label} = {fmt_inertia(inertia)}\n"
            + f"({inertia_hint})\n\n"
            + verdict
        )
        if iteration == 1:
            description = (
                "[Mã giả Bước 2b] μⱼ ← mean({ xᵢ : cluster(xᵢ) = j })\n"
                "\n"
                "① Ý nghĩa: Trọng tâm mới = trung bình cộng tọa độ\n"
                '  các điểm trong cụm. Đây là điểm "trung tâm nhất"\n'
                "  về mặt hình học."
            )
        else:
            description = "[Mã giả Bước 2b] μⱼ ← mean({ xᵢ : cluster(xᵢ) = j })"
        update_snap = {
            **snap,
            "phase": "update",
            "headers": update_headers,
            "rows": update_rows,
            "alignments": ["left", "left", "left"],
            "conclusion": conclusion,
            "omit_step_number": True,
        }
        self._log.add(
            f"Vòng lặp {iteration} — Bước 2b: Cập nhật trọng tâm",
            description,
            update_snap,
            "SUCCESS",
        )

    def inertia_warning_step(
        self,
        iteration: int,
        inertia: float,
        inertia_prev: float,
        inertia_label: str,
    ) -> None:
        self._log.add(
            f"Cảnh báo {inertia_label}",
            (
                f"{inertia_label} tăng từ {fmt_inertia(inertia_prev)} lên {fmt_inertia(inertia)} "
                f"sau vòng lặp {iteration} (thường không mong muốn)."
            ),
            {
                "iteration": iteration,
                "sse": inertia,
                "inertia": inertia,
                "inertia_prev": inertia_prev,
                "inertia_label": inertia_label,
                "phase": "warning",
                "omit_step_number": True,
            },
            "WARNING",
        )

    def convergence_step(
        self,
        iteration: int,
        inertia: float,
        inertia_label: str,
        k: int,
    ) -> None:
        first = self._first_inertia
        if first is not None and first > inertia + 1e-9:
            compare = (
                f"{inertia_label} cuối = {fmt_inertia(inertia)}"
                f"  (so với vòng 1: {fmt_inertia(first)} → đã giảm tốt)"
            )
        elif first is not None:
            compare = f"{inertia_label} cuối = {fmt_inertia(inertia)}  (vòng 1: {fmt_inertia(first)})"
        else:
            compare = f"{inertia_label} cuối = {fmt_inertia(inertia)}"
        description = (
            "[Mã giả Bước 2c] NẾU trọng tâm / phân hoạch không đổi → DỪNG\n"
            "\n"
            "① Hội tụ là gì?\n"
            "  Khi trọng tâm không còn di chuyển → các điểm không\n"
            "  đổi cụm → thuật toán đã tìm được phân hoạch ổn định.\n"
            "\n"
            f"  Tất cả K={k} trọng tâm KHÔNG thay đổi sau vòng lặp {iteration}.\n"
            "  ✓ Điều kiện hội tụ thỏa mãn ⟹ DỪNG THUẬT TOÁN\n"
            "\n"
            f"{compare}\n"
            f"Kết luận: Thuật toán hội tụ sau {iteration} vòng lặp."
        )
        self._log.add(
            "Kết thúc — Bước 2c: Kiểm tra hội tụ",
            description,
            {
                "iteration": iteration,
                "sse": inertia,
                "inertia": inertia,
                "inertia_label": inertia_label,
                "phase": "converged",
                "omit_step_number": True,
            },
            "SUCCESS",
        )

    def max_iter_stop_step(self, max_iter: int, inertia: float, inertia_label: str) -> None:
        first = self._first_inertia
        compare = f"{inertia_label} cuối = {fmt_inertia(inertia)}"
        if first is not None:
            compare += f"  (vòng 1: {fmt_inertia(first)})"
        description = (
            "[Mã giả Bước 2d] NẾU t = max_iter → DỪNG\n"
            "\n"
            "① Giới hạn vòng lặp là gì?\n"
            "  Dù chưa hội tụ, thuật toán vẫn dừng để tránh chạy vô hạn.\n"
            "\n"
            f"  Đã chạy đủ max_iter={max_iter} vòng lặp, trọng tâm vẫn còn thay đổi.\n"
            "  ✗ Chưa hội tụ ⟹ DỪNG THEO GIỚI HẠN (max_iter)\n"
            "\n"
            f"{compare}\n"
            "Kết quả có thể chưa tối ưu — thử tăng max_iter hoặc kiểm tra dữ liệu."
        )
        self._log.add(
            "Kết thúc — Bước 2d: Kiểm tra giới hạn vòng lặp",
            description,
            {
                "iteration": max_iter,
                "sse": inertia,
                "inertia": inertia,
                "inertia_label": inertia_label,
                "phase": "max_iter",
                "omit_step_number": True,
            },
            "WARNING",
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

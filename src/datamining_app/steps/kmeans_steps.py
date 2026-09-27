"""K-Means console steps. Distance and centroid updates stay in ``algorithms.kmeans``."""
from __future__ import annotations

from typing import Any

from datamining_app.algorithms._logger import StepLogger
from datamining_app.fmt import fmt_centroid, fmt_dist, fmt_inertia

class _KMeansSteps:
    """Ghi bước K-Means vào StepLogger từ tâm cụm, gán cụm và inertia đã tính."""
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
            ids = ", ".join(members) or "{}"
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




def build_predict_explanation(
    features: list[str],
    vec: list[float],
    metric: str,
    distances: list[float],
    cluster: int,
) -> str:
    """Console text for assigning one point to the nearest centroid."""
    lines = [
        f"Điểm { {f: v for f, v in zip(features, vec)} }",
        f"Độ đo: {metric}",
    ]
    for i, dist in enumerate(distances, start=1):
        mark = " ← gần nhất" if i == cluster else ""
        lines.append(f"  d(C{i}) = {fmt_dist(dist)}{mark}")
    lines.append(f"⇒ Gán cụm C{cluster}")
    return "\n".join(lines)


__all__ = ["_KMeansSteps", "build_predict_explanation"]

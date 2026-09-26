from __future__ import annotations

from typing import Any

import numpy as np

from datamining_app.core.models import AlgorithmResult
from datamining_app.fmt import fmt_inertia

PALETTE = ["#4A90D9", "#5CB85C", "#F0AD4E", "#D9534F", "#5BC0DE", "#9B59B6", "#E67E22", "#1ABC9C"]


class ClusterVisualizer:
    def draw(
        self,
        result: AlgorithmResult,
        ax: Any,
        highlight: list[float] | None = None,
        iteration: int | None = None,
        raw: bool = False,
        show_init: bool = False,
        phase: str | None = None,
        voronoi: bool = False,
        **kwargs: Any,
    ) -> None:
        ax.clear()
        output = result.output or {}
        points = [dict(p) for p in (output.get("points") or [])]
        features = output.get("features") or ["x", "y"]
        history = output.get("history") or []
        metric = str((result.parameters or {}).get("distance") or output.get("metric") or "euclidean")
        centroids = output.get("centroids") or []
        sse = output.get("sse")
        title_iter = None
        old_centroids = None
        if history:
            snap = history[-1] if iteration is None else history[min(max(iteration, 0), len(history) - 1)]
            if show_init:
                centroids = snap.get("centroids") or centroids
                raw = True
            elif phase == "assign":
                centroids = snap.get("centroids") or centroids
            else:
                centroids = snap.get("new_centroids") or snap.get("centroids") or centroids
                old_centroids = snap.get("centroids")
            assignments = [] if raw else (snap.get("assignments") or [])
            sse = snap.get("sse", sse)
            title_iter = snap.get("iteration")
            if assignments:
                for point, cluster in zip(points, assignments):
                    point["cluster"] = cluster
        if raw:
            for point in points:
                point["cluster"] = 0
        if not points:
            ax.text(0.5, 0.5, "Chưa có kết quả", ha="center", va="center")
            ax.axis("off")
            return

        marker = "o" if metric != "manhattan" else "D"
        xs, ys, colors = [], [], []
        for point in points:
            vec = point["vector"]
            xs.append(vec[0])
            ys.append(vec[1] if len(vec) > 1 else 0.0)
            cluster = int(point.get("cluster") or 0)
            colors.append("#888888" if cluster <= 0 else PALETTE[(cluster - 1) % len(PALETTE)])

        if voronoi and centroids and not raw:
            _paint_regions(ax, xs, ys, centroids, metric)

        ax.scatter(xs, ys, c=colors, s=70, marker=marker, zorder=3, edgecolors="#333333")
        for point, x, y in zip(points, xs, ys):
            ax.annotate(str(point["id"]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

        if phase == "assign" and centroids:
            for point, x, y in zip(points, xs, ys):
                cluster = int(point.get("cluster") or 1) - 1
                if 0 <= cluster < len(centroids):
                    c = centroids[cluster]
                    cy = c[1] if len(c) > 1 else 0.0
                    ax.plot([x, c[0]], [y, cy], color="#999999", lw=0.6, zorder=2)

        if old_centroids and phase == "update":
            nx_ = [c[0] for c in old_centroids]
            ny_ = [c[1] if len(c) > 1 else 0.0 for c in old_centroids]
            ax.scatter(nx_, ny_, marker="X", s=120, c="#bbbbbb", zorder=4, label="Centroid cũ")
            for old, new in zip(old_centroids, centroids):
                oy = old[1] if len(old) > 1 else 0.0
                ny = new[1] if len(new) > 1 else 0.0
                ax.annotate("", xy=(new[0], ny), xytext=(old[0], oy), arrowprops={"arrowstyle": "->", "color": "#333"})

        cx = [c[0] for c in centroids]
        cy = [c[1] if len(c) > 1 else 0.0 for c in centroids]
        ax.scatter(cx, cy, marker="X", s=200, c="black", linewidths=1.2, zorder=5, label="Centroid")

        if highlight and len(highlight) >= 1:
            hx = highlight[0]
            hy = highlight[1] if len(highlight) > 1 else 0.0
            ax.scatter([hx], [hy], marker="*", s=300, c="gold", edgecolors="black", zorder=6, label="Điểm mới")

        ax.set_xlabel(features[0] if features else "x")
        ax.set_ylabel(features[1] if len(features) > 1 else "y")
        if metric == "manhattan":
            _diamond_grid(ax)
        else:
            ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        metric_label = "Euclide" if metric != "manhattan" else "Manhattan"
        snap_label = ""
        inertia_val = sse
        if history:
            active = history[-1] if iteration is None else history[min(max(iteration, 0), len(history) - 1)]
            snap_label = str(active.get("inertia_label") or "")
            inertia_val = active.get("inertia", active.get("sse", sse))
        inertia_label = (
            output.get("inertia_label")
            or snap_label
            or ("SAE" if metric == "manhattan" else "SSE")
        )
        if raw or show_init:
            title = f"Khởi tạo trọng tâm ({metric_label})"
        elif phase == "assign" and title_iter is not None:
            title = f"Vòng lặp {title_iter} — Bước 2a: Gán cụm ({metric_label})"
        elif phase == "update" and title_iter is not None:
            title = (
                f"Vòng lặp {title_iter} — Bước 2b: Cập nhật tâm"
                f" · {inertia_label} = {fmt_inertia(float(inertia_val or 0))}"
            )
        elif title_iter is not None:
            title = (
                f"Vòng lặp {title_iter} — {inertia_label} ({metric_label})"
                f" = {fmt_inertia(float(inertia_val or 0))}"
            )
        else:
            title = f"Phân cụm K-Means ({metric_label})"
        ax.set_title(title)
        ax.set_facecolor("white")
        ax.figure.patch.set_facecolor("white")


def _paint_regions(ax, xs, ys, centroids, metric: str) -> None:
    if not xs:
        return
    pad = 0.5
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    grid = 120
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid), np.linspace(ymin, ymax, grid))
    zz = np.zeros(xx.shape)
    for i in range(grid):
        for j in range(grid):
            p = [float(xx[i, j]), float(yy[i, j])]
            dists = [_dist(p, c if len(c) > 1 else [c[0], 0.0], metric) for c in centroids]
            zz[i, j] = int(np.argmin(dists))
    colors = [PALETTE[int(k) % len(PALETTE)] for k in range(len(centroids))]
    ax.contourf(xx, yy, zz, levels=np.arange(-0.5, len(centroids) + 0.5, 1), colors=colors, alpha=0.18)


def _diamond_grid(ax) -> None:
    ax.grid(alpha=0.15)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    for k in np.linspace(-span, span, 9):
        ax.plot([xlim[0], xlim[1]], [ylim[0] + k, ylim[0] + k + (xlim[1] - xlim[0])], color="#cccccc", lw=0.5, zorder=0)
        ax.plot([xlim[0], xlim[1]], [ylim[1] - k, ylim[1] - k - (xlim[1] - xlim[0])], color="#cccccc", lw=0.5, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _dist(a: list[float], b: list[float], metric: str) -> float:
    n = min(len(a), len(b))
    if metric == "manhattan":
        return sum(abs(a[i] - b[i]) for i in range(n))
    return sum((a[i] - b[i]) ** 2 for i in range(n)) ** 0.5

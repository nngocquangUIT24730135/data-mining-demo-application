from __future__ import annotations

from typing import Any

import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch

from datamining_app.core.models import AlgorithmResult

INTERNAL = "#4A90D9"
YES = "#5CB85C"
NO = "#D9534F"
OTHER_LEAF = "#7F8C8D"
PENDING = "#B0B0B0"
HIGHLIGHT = "#F0AD4E"


class TreeVisualizer:
    def draw(self, result: AlgorithmResult, ax: Any, tree: dict[str, Any] | None = None, highlight: str | None = None, **kwargs: Any) -> None:
        ax.clear()
        tree = tree if tree is not None else (result.output or {}).get("tree")
        ax.set_facecolor("white")
        ax.figure.patch.set_facecolor("white")
        if not tree:
            ax.text(0.5, 0.5, "Chưa có cây", ha="center", va="center")
            ax.axis("off")
            return
        graph = nx.DiGraph()
        labels: dict[str, str] = {}
        kinds: dict[str, str] = {}
        paths: dict[str, str] = {}
        _add_node(graph, tree, "n0", labels, kinds, paths)
        pos = hierarchy_pos(graph, "n0")
        nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, edge_color="#555555", width=1.3)
        for nid, (x, y) in pos.items():
            kind = kinds[nid]
            color = _color_for(kind)
            active = bool(highlight) and paths.get(nid) == highlight
            edgecolor = HIGHLIGHT if active else "#222222"
            lw = 2.6 if active else 1.1
            if kind == "internal":
                patch = FancyBboxPatch(
                    (x - 0.07, y - 0.035),
                    0.14,
                    0.07,
                    boxstyle="round,pad=0.01,rounding_size=0.01",
                    facecolor=color,
                    edgecolor=edgecolor,
                    linewidth=lw,
                    linestyle="--" if kind == "pending" else "-",
                    zorder=3,
                )
            else:
                patch = Circle(
                    (x, y),
                    0.045,
                    facecolor=color,
                    edgecolor=edgecolor,
                    linewidth=lw,
                    linestyle="--" if kind == "pending" else "-",
                    zorder=3,
                )
            ax.add_patch(patch)
            ax.text(x, y, labels[nid], ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
        edge_labels = nx.get_edge_attributes(graph, "label")
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        ax.set_title("Cây quyết định ID3")
        ax.axis("off")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, 1.15)


def _color_for(kind: str) -> str:
    if kind == "internal":
        return INTERNAL
    if kind == "yes":
        return YES
    if kind == "no":
        return NO
    if kind == "pending":
        return PENDING
    return OTHER_LEAF


def _add_node(
    graph: nx.DiGraph,
    node: dict[str, Any],
    nid: str,
    labels: dict[str, str],
    kinds: dict[str, str],
    paths: dict[str, str],
) -> None:
    paths[nid] = str(node.get("path") or "")
    if node.get("pending") and node.get("is_leaf"):
        labels[nid] = "…"
        kinds[nid] = "pending"
        graph.add_node(nid)
        return
    if node.get("is_leaf"):
        label = str(node.get("label", ""))
        labels[nid] = f"{label}\n(n={node.get('samples', 0)})"
        low = label.lower()
        if low in {"yes", "yes play", "chấp nhận", "play", "t"}:
            kinds[nid] = "yes"
        elif low in {"no", "từ chối", "f"}:
            kinds[nid] = "no"
        else:
            kinds[nid] = "other"
        graph.add_node(nid)
        return
    labels[nid] = str(node.get("attribute", "?"))
    kinds[nid] = "pending" if node.get("pending") else "internal"
    graph.add_node(nid)
    for i, (value, child) in enumerate((node.get("children") or {}).items()):
        cid = f"{nid}_{i}"
        _add_node(graph, child, cid, labels, kinds, paths)
        graph.add_edge(nid, cid, label=str(value))


def hierarchy_pos(
    graph: nx.DiGraph,
    root: str,
    width: float = 1.0,
    vert_gap: float = 0.22,
    vert_loc: float = 1.0,
) -> dict[str, tuple[float, float]]:
    pos = {root: (0.5, vert_loc)}

    def _place(node: str, left: float, right: float, y: float) -> None:
        children = list(graph.successors(node))
        if not children:
            return
        width_each = (right - left) / len(children)
        for i, child in enumerate(children):
            cx = left + width_each * (i + 0.5)
            pos[child] = (cx, y - vert_gap)
            _place(child, left + width_each * i, left + width_each * (i + 1), y - vert_gap)

    _place(root, 0.0, width, vert_loc)
    return pos

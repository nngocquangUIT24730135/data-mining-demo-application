from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm, StepLogger
from datamining_app.algorithms.dataset_utils import excluded_headers
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult


class ID3Algorithm(BaseAlgorithm):
    name = "ID3"
    description = "Xây dựng cây quyết định theo Information Gain."
    supports_predict = True
    supports_visualization = True
    param_schema = {
        "decision_attr": ParamDef(
            type="choice",
            options="headers",
            default=None,
            label_vi="Thuộc tính quyết định (lớp)",
            label_en="Decision attribute (class)",
        ),
        "max_depth": ParamDef(
            type="int",
            default=6,
            min=1,
            max=20,
            step=1,
            label_vi="Độ sâu tối đa",
            label_en="Maximum depth",
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._tree: dict[str, Any] | None = None
        self._features: list[str] = []
        self._decision: str = ""

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        headers = excluded_headers(dataset, params)
        decision = params.get("decision_attr") or headers[-1]
        if decision not in headers:
            raise ValueError(f"Thuộc tính quyết định '{decision}' không tồn tại.")
        features = [h for h in headers if h != decision]
        max_depth = int(params.get("max_depth", 6))
        rows = [{h: str(row.get(h, "")).strip() for h in headers} for row in dataset.rows]

        log = self._new_logger()
        counts = Counter(r[decision] for r in rows)
        info_d = entropy(rows, decision)
        step_builder = _ID3Steps(log, len(rows), dict(counts))
        step_builder.initial_entropy_step(rows, decision, info_d, counts)

        tree = self._build(rows, features, decision, max_depth, 0, step_builder, "Gốc")
        self._tree = tree
        self._features = features
        self._decision = decision

        rules = extract_tree_rules(tree)
        summary = (
            f"Cây ID3 gốc = '{tree.get('attribute', tree.get('label'))}'. "
            f"Info(D) = {info_d:.4f}. Độ sâu tối đa cho phép = {max_depth}."
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"decision_attr": decision, "max_depth": max_depth},
            steps=log.steps,
            output={
                "tree": tree,
                "features": features,
                "decision": decision,
                "entropy": info_d,
                "class_counts": dict(counts),
                "n_rows": len(rows),
                "rules": rules,
                "rows": rows,
            },
            summary=summary,
        )
        return self._finish(result)

    def _build(
        self,
        rows: list[dict[str, str]],
        features: list[str],
        decision: str,
        max_depth: int,
        depth: int,
        step_builder: _ID3Steps,
        path: str,
    ) -> dict[str, Any]:
        counts = Counter(r[decision] for r in rows)
        majority = counts.most_common(1)[0][0]
        node: dict[str, Any] = {
            "samples": len(rows),
            "class_counts": dict(counts),
            "path": path,
        }
        if len(counts) == 1 or not features or depth >= max_depth:
            node.update({"is_leaf": True, "label": majority, "pending": False})
            step_builder.leaf_step(path, majority, counts, node)
            return node

        info_d = entropy(rows, decision)
        gains = []
        for feat in features:
            split_info, gain = information_gain(rows, feat, decision, info_d)
            gains.append((feat, gain, split_info))
        gains.sort(key=lambda x: -x[1])
        best, best_gain, _ = gains[0]
        values = sorted({r[best] for r in rows})
        split_node = {
            "is_leaf": False,
            "attribute": best,
            "gain": best_gain,
            "samples": len(rows),
            "class_counts": dict(counts),
            "path": path,
            "pending": False,
            "children": {
                value: {
                    "is_leaf": True,
                    "label": "…",
                    "pending": True,
                    "samples": 0,
                    "path": f"{path} / {best}={value}",
                }
                for value in values
            },
        }
        step_builder.gain_step(path, rows, decision, info_d, gains, best, split_node)
        remaining = [f for f in features if f != best]
        children = {}
        for value in values:
            subset = [r for r in rows if r[best] == value]
            children[value] = self._build(
                subset, remaining, decision, max_depth, depth + 1, step_builder, f"{path} / {best}={value}"
            )
        node.update({"is_leaf": False, "attribute": best, "gain": best_gain, "children": children})
        return node

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        if not self._trained or self._tree is None:
            raise ValueError("Chưa huấn luyện cây ID3.")
        normalized = {k: str(v).strip() for k, v in sample.items()}
        path: list[str] = []
        node = self._tree
        while not node.get("is_leaf"):
            attr = node["attribute"]
            value = normalized.get(attr, "")
            path.append(f"{attr} = {value}")
            children = node.get("children") or {}
            if value not in children:
                majority = max(node["class_counts"], key=node["class_counts"].get)
                explanation = " → ".join(path) + f"\nGiá trị '{value}' không có nhánh. Lớp đa số = {majority}."
                return PredictResult(
                    label=majority,
                    explanation=explanation,
                    details={"path": path, "fallback": True},
                    sample=normalized,
                )
            node = children[value]
        label = node["label"]
        path.append(f"Lá = {label}")
        return PredictResult(
            label=label,
            explanation="Suy luận trên cây:\n  " + "\n  → ".join(path),
            details={"path": path, "leaf": node},
            sample=normalized,
        )


class _ID3Steps:
    def __init__(self, log: StepLogger, n_rows: int, class_counts: dict[str, int]) -> None:
        self._log = log
        self._growing: dict[str, Any] = {
            "is_leaf": False,
            "attribute": "S",
            "pending": True,
            "children": {},
            "samples": n_rows,
            "class_counts": class_counts,
            "path": "Gốc",
        }

    def initial_entropy_step(
        self,
        rows: list[dict[str, str]],
        decision: str,
        info_d: float,
        counts: Counter,
    ) -> None:
        class_rows = [[label, count] for label, count in counts.items()]
        self._log.add(
            "Tính Entropy ban đầu của tập S",
            self._entropy_formula(rows, decision, "S"),
            {
                "entropy": info_d,
                "class_counts": dict(counts),
                "rows": class_rows,
                "headers": ["Lớp", "Số mẫu"],
                "alignments": ["left", "right"],
                "decision": decision,
                "sample_rows": rows,
                "path": "Gốc",
                "tree": copy.deepcopy(self._growing),
                "conclusion": f"  → Entropy(S) = {info_d:.3f} bit",
            },
            "STEP_HEADER",
        )

    def leaf_step(self, path: str, majority: str, counts: Counter, node: dict[str, Any]) -> None:
        self._place_node(path, node)
        self._log.add(
            f"Lá — {path}",
            f"Gán lớp '{majority}' ({self._fmt_counts(counts)})."
            + (" Tập thuần." if len(counts) == 1 else " Dừng theo ràng buộc."),
            {
                **node,
                "headers": ["Lớp", "Số mẫu"],
                "rows": [[label, count] for label, count in counts.items()],
                "alignments": ["left", "right"],
                "tree": copy.deepcopy(self._growing),
                "conclusion": f"  → Lá = {majority}",
            },
            "SUCCESS",
        )

    def gain_step(
        self,
        path: str,
        rows: list[dict[str, str]],
        decision: str,
        info_d: float,
        gains: list[tuple[str, float, float]],
        best: str,
        split_node: dict[str, Any],
    ) -> None:
        best_gain = next(gain for feat, gain, _ in gains if feat == best)
        values = sorted({r[best] for r in rows})
        self._place_node(path, split_node)
        self._log.add(
            f"Tính Information Gain — {path}",
            self._format_gains(rows, decision, info_d, gains, best),
            {
                "gains": [{"feature": f, "gain": g, "info": s} for f, g, s in gains],
                "chosen": best,
                "entropy": info_d,
                "path": path,
                "split_values": values,
                "headers": ["Thuộc tính", "Info", "Gain"],
                "rows": [
                    [feat, f"{split:.3f}", f"{gain:.3f}" + (" ★" if feat == best else "")]
                    for feat, gain, split in gains
                ],
                "alignments": ["left", "right", "right"],
                "tree": copy.deepcopy(self._growing),
                "conclusion": f'  → Chọn "{best}" làm nút (Gain = {best_gain:.3f})',
            },
            "SUCCESS",
        )

    def _place_node(self, path: str, node: dict[str, Any]) -> None:
        """Insert/replace a node on the growing visualization tree (DFS snapshots)."""
        if path == "Gốc" or not path:
            self._growing.clear()
            self._growing.update(copy.deepcopy(node))
            return
        segments = path.split(" / ")[1:]
        current = self._growing
        for i, segment in enumerate(segments):
            attr, _, value = segment.partition("=")
            children = current.setdefault("children", {})
            if i == len(segments) - 1:
                children[value] = copy.deepcopy(node)
                return
            nxt = children.get(value)
            if nxt is None:
                nxt = {"is_leaf": False, "attribute": attr, "children": {}, "pending": True}
                children[value] = nxt
            current = nxt

    @staticmethod
    def _fmt_counts(counts: Counter) -> str:
        return ", ".join(f"{k}: {v}" for k, v in counts.items())

    @staticmethod
    def _entropy_formula(rows: list[dict[str, str]], decision: str, name: str) -> str:
        n = len(rows)
        counts = Counter(r[decision] for r in rows)
        parts = [f"  |{name}| = {n} mẫu  →  " + ", ".join(f"{v} {k}" for k, v in counts.items())]
        terms = []
        for label, count in counts.items():
            terms.append(f"({count}/{n} × log₂({count}/{n}))")
        parts.append("  Entropy = -" + " - ".join(terms))
        return "\n".join(parts)

    @staticmethod
    def _format_gains(
        rows: list[dict[str, str]],
        decision: str,
        info_d: float,
        gains: list[tuple[str, float, float]],
        best: str,
    ) -> str:
        lines = []
        for feat, gain, split in gains:
            lines.append(f"  [ Thuộc tính: {feat} ]")
            for value in sorted({r[feat] for r in rows}):
                subset = [r for r in rows if r[feat] == value]
                counts = Counter(r[decision] for r in subset)
                dist = ", ".join(f"{v} {k}" for k, v in counts.items())
                lines.append(f"    {value}: {len(subset)} mẫu → {dist}  → Entropy = {entropy(subset, decision):.3f}")
            lines.append(f"    Info({feat}) = {split:.3f}")
            star = "  ★" if feat == best else ""
            lines.append(f"    Gain({feat}) = {info_d:.3f} - {split:.3f} = {gain:.3f}{star}")
            lines.append("")
        return "\n".join(lines)


def entropy(rows: list[dict[str, str]], decision: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    counts = Counter(r[decision] for r in rows)
    total = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            total -= p * math.log2(p)
    return total


def information_gain(
    rows: list[dict[str, str]], feature: str, decision: str, info_d: float
) -> tuple[float, float]:
    n = len(rows)
    split = 0.0
    for value in {r[feature] for r in rows}:
        subset = [r for r in rows if r[feature] == value]
        split += (len(subset) / n) * entropy(subset, decision)
    return split, info_d - split


def extract_tree_rules(tree: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], conds: list[tuple[str, str]]) -> None:
        if node.get("is_leaf"):
            rules.append({"conditions": list(conds), "label": node.get("label"), "samples": node.get("samples", 0)})
            return
        attr = node.get("attribute", "?")
        for value, child in (node.get("children") or {}).items():
            walk(child, conds + [(attr, str(value))])

    walk(tree, [])
    return rules

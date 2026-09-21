from __future__ import annotations

import math
from collections import Counter
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.algorithms.dataset_utils import excluded_headers
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult


class NaiveBayesAlgorithm(BaseAlgorithm):
    name = "Naïve Bayes"
    description = "Phân lớp Bayes với làm mịn Laplace tùy chỉnh."
    supports_predict = True
    param_schema = {
        "decision_attr": ParamDef(
            type="choice",
            options="headers",
            default=None,
            label_vi="Thuộc tính quyết định (lớp)",
            label_en="Decision attribute (class)",
        ),
        "laplace_alpha": ParamDef(
            type="float",
            default=1.0,
            min=0.0,
            max=10.0,
            step=0.1,
            label_vi="Hệ số Laplace (α)",
            label_en="Laplace Alpha (α)",
        ),
    }

    def __init__(self) -> None:
        super().__init__()
        self._priors: dict[str, float] = {}
        self._likelihoods: dict[str, dict[str, dict[str, float]]] = {}
        self._class_counts: dict[str, int] = {}
        self._value_counts: dict[str, dict[str, dict[str, int]]] = {}
        self._vocab: dict[str, list[str]] = {}
        self._features: list[str] = []
        self._decision: str = ""
        self._alpha: float = 1.0
        self._n: int = 0

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        headers = excluded_headers(dataset, params)
        decision = params.get("decision_attr") or headers[-1]
        if decision not in headers:
            raise ValueError(f"Thuộc tính quyết định '{decision}' không tồn tại.")
        features = [h for h in headers if h != decision]
        alpha = float(params.get("laplace_alpha", 1.0))
        rows = [{h: str(row.get(h, "")).strip() for h in headers} for row in dataset.rows]
        n = len(rows)
        class_counts = Counter(r[decision] for r in rows)
        classes = sorted(class_counts)

        vocab: dict[str, list[str]] = {
            feat: sorted({r[feat] for r in rows}) for feat in features
        }
        value_counts: dict[str, dict[str, dict[str, int]]] = {
            cls: {feat: Counter(r[feat] for r in rows if r[decision] == cls) for feat in features}
            for cls in classes
        }

        priors = {cls: class_counts[cls] / n for cls in classes}
        likelihoods: dict[str, dict[str, dict[str, float]]] = {}
        for cls in classes:
            likelihoods[cls] = {}
            n_c = class_counts[cls]
            for feat in features:
                v = len(vocab[feat])
                likelihoods[cls][feat] = {}
                for value in vocab[feat]:
                    count = value_counts[cls][feat][value]
                    likelihoods[cls][feat][value] = (count + alpha) / (n_c + alpha * v) if (n_c + alpha * v) else 0.0

        log = self._new_logger()
        log.add(
            "Xác suất tiên nghiệm P(C)",
            "P(C) = n_C / N",
            {
                "priors": priors,
                "class_counts": dict(class_counts),
                "alpha": alpha,
                "headers": ["Lớp", "Count", "P(C)"],
                "rows": [[cls, class_counts[cls], f"{priors[cls]:.4f}"] for cls in classes],
                "alignments": ["left", "right", "right"],
                "conclusion": f"  → α Laplace = {alpha}",
            },
            "STEP_HEADER",
        )
        for feat in features:
            like_rows = []
            v = len(vocab[feat])
            for cls in classes:
                n_c = class_counts[cls]
                for value in vocab[feat]:
                    count = value_counts[cls][feat][value]
                    p = likelihoods[cls][feat][value]
                    like_rows.append(
                        [
                            cls,
                            value,
                            count,
                            f"{p:.4f}",
                            f"({count}+{alpha})/({n_c}+{alpha}×{v})",
                        ]
                    )
            log.add(
                f"Bảng likelihood — {feat}",
                f"P({feat}=v | C) = (count + α) / (n_C + α·|{feat}|)",
                {
                    "feature": feat,
                    "headers": ["Lớp", "Giá trị", "Count", "P(v|C)", "Công thức"],
                    "rows": like_rows,
                    "alignments": ["left", "left", "right", "right", "left"],
                },
                "DATA",
            )

        self._priors = priors
        self._likelihoods = likelihoods
        self._class_counts = dict(class_counts)
        self._value_counts = {c: {f: dict(value_counts[c][f]) for f in features} for c in classes}
        self._vocab = vocab
        self._features = features
        self._decision = decision
        self._alpha = alpha
        self._n = n
        self._trained = True

        summary = (
            f"Naïve Bayes với α = {alpha}. {len(classes)} lớp, {len(features)} thuộc tính. "
            f"Tiên nghiệm: " + ", ".join(f"{c}={priors[c]:.3f}" for c in classes)
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"decision_attr": decision, "laplace_alpha": alpha},
            steps=log.steps,
            output={
                "priors": priors,
                "likelihoods": likelihoods,
                "vocab": vocab,
                "features": features,
                "decision": decision,
                "class_counts": dict(class_counts),
            },
            summary=summary,
        )
        self._last_result = result
        return result

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        if not self._trained:
            raise ValueError("Chưa huấn luyện Naïve Bayes.")
        normalized = {k: str(v).strip() for k, v in sample.items()}
        scores: dict[str, dict[str, Any]] = {}
        lines = [f"Mẫu: {normalized}", f"α = {self._alpha}"]
        for cls, prior in self._priors.items():
            log_prob = math.log(prior) if prior > 0 else float("-inf")
            terms = [f"P({cls})={prior:.4f}"]
            for feat in self._features:
                value = normalized.get(feat, "")
                table = self._likelihoods[cls][feat]
                if value in table:
                    p = table[value]
                    src = "bảng"
                else:
                    n_c = self._class_counts[cls]
                    v = len(self._vocab[feat])
                    p = self._alpha / (n_c + self._alpha * v) if (n_c + self._alpha * v) else 0.0
                    src = "giá trị mới (Laplace)"
                if p > 0:
                    log_prob += math.log(p)
                else:
                    log_prob = float("-inf")
                terms.append(f"P({feat}={value}|{cls})={p:.4f} [{src}]")
            scores[cls] = {"log": log_prob, "terms": terms}
            lines.append(f"\nLớp {cls}: " + " × ".join(terms))
            lines.append(f"  log-likelihood = {log_prob:.4f}")

        finite = {c: s["log"] for c, s in scores.items() if math.isfinite(s["log"])}
        if finite:
            m = max(finite.values())
            exps = {c: math.exp(s["log"] - m) for c, s in scores.items() if math.isfinite(s["log"])}
            z = sum(exps.values())
            posteriors = {c: (exps.get(c, 0.0) / z if z else 0.0) for c in scores}
        else:
            posteriors = {c: 0.0 for c in scores}
        label = max(posteriors, key=posteriors.get)
        lines.append("\nHậu nghiệm (chuẩn hóa):")
        for cls, p in posteriors.items():
            mark = " ← argmax" if cls == label else ""
            lines.append(f"  P({cls}|x) = {p:.4f}{mark}")
        return PredictResult(
            label=label,
            explanation="\n".join(lines),
            details={"posteriors": posteriors, "scores": {c: s["log"] for c, s in scores.items()}},
            sample=normalized,
        )

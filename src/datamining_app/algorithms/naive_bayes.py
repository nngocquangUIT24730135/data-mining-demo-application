from __future__ import annotations

import math
from abc import ABC
from collections import Counter
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm, StepLogger
from datamining_app.algorithms.dataset_utils import excluded_headers
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult
from datamining_app.fmt import fmt_prob, fmt_score


class _NaiveBayesBase(BaseAlgorithm, ABC):
    """Shared predict() and trained-state storage for classic / Laplace NB."""

    supports_predict = True

    def __init__(self) -> None:
        super().__init__()
        self._priors: dict[str, float] = {}
        self._likelihoods: dict[str, dict[str, dict[str, float]]] = {}
        self._class_counts: dict[str, int] = {}
        self._value_counts: dict[str, dict[str, dict[str, int]]] = {}
        self._vocab: dict[str, list[str]] = {}
        self._features: list[str] = []
        self._decision: str = ""
        self._alpha: float = 0.0
        self._n: int = 0
        self._smoothing: bool = False

    def _store_model(
        self,
        *,
        priors: dict[str, float],
        likelihoods: dict[str, dict[str, dict[str, float]]],
        class_counts: Counter,
        value_counts: dict[str, dict[str, dict[str, int]]],
        vocab: dict[str, list[str]],
        features: list[str],
        decision: str,
        alpha: float,
        n: int,
        smoothing: bool,
    ) -> None:
        classes = sorted(class_counts)
        self._priors = priors
        self._likelihoods = likelihoods
        self._class_counts = dict(class_counts)
        self._value_counts = {c: {f: dict(value_counts[c][f]) for f in features} for c in classes}
        self._vocab = vocab
        self._features = features
        self._decision = decision
        self._alpha = alpha
        self._n = n
        self._smoothing = smoothing

    def predict(self, sample: dict[str, Any]) -> PredictResult:
        if not self._trained:
            raise ValueError("Chưa huấn luyện Naïve Bayes.")
        normalized = {k: str(v).strip() for k, v in sample.items()}
        scores: dict[str, dict[str, Any]] = {}
        lines = [f"Mẫu: {normalized}"]
        if self._smoothing:
            lines.append(f"α = {self._alpha} (Laplace)")
        else:
            lines.append("Không làm mịn (cổ điển) — P=0 → loại lớp")

        for cls, prior in self._priors.items():
            log_prob = math.log(prior) if prior > 0 else float("-inf")
            terms = [f"P({cls})={fmt_prob(prior)}"]
            for feat in self._features:
                value = normalized.get(feat, "")
                table = self._likelihoods[cls][feat]
                if value in table:
                    p = table[value]
                    src = "bảng"
                elif self._smoothing:
                    n_c = self._class_counts[cls]
                    v = len(self._vocab[feat])
                    denom = n_c + self._alpha * v
                    p = self._alpha / denom if denom else 0.0
                    src = "giá trị mới (Laplace)"
                else:
                    p = 0.0
                    src = "giá trị mới / count=0"
                if p > 0:
                    log_prob += math.log(p)
                else:
                    log_prob = float("-inf")
                terms.append(f"P({feat}={value}|{cls})={fmt_prob(p)} [{src}]")
            scores[cls] = {"log": log_prob, "terms": terms}
            lines.append(f"\nLớp {cls}: " + " × ".join(terms))
            lines.append(f"  log-likelihood = {fmt_score(log_prob)}")

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
            lines.append(f"  P({cls}|x) = {fmt_prob(p)}{mark}")
        return PredictResult(
            label=label,
            explanation="\n".join(lines),
            details={"posteriors": posteriors, "scores": {c: s["log"] for c, s in scores.items()}},
            sample=normalized,
        )


def _prepare_nb_tables(
    rows: list[dict[str, str]],
    features: list[str],
    decision: str,
    alpha: float,
) -> tuple[
    list[str],
    Counter,
    dict[str, float],
    dict[str, list[str]],
    dict[str, dict[str, dict[str, int]]],
    dict[str, dict[str, dict[str, float]]],
]:
    n = len(rows)
    class_counts = Counter(r[decision] for r in rows)
    classes = sorted(class_counts)
    vocab: dict[str, list[str]] = {feat: sorted({r[feat] for r in rows}) for feat in features}
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
                if alpha > 0:
                    denom = n_c + alpha * v
                    p = (count + alpha) / denom if denom else 0.0
                else:
                    p = (count / n_c) if n_c else 0.0
                likelihoods[cls][feat][value] = p
    return classes, class_counts, priors, vocab, value_counts, likelihoods


class LaplaceBayesAlgorithm(_NaiveBayesBase):
    name = "Naïve Bayes (Laplace)"
    description = "Phân lớp Naïve Bayes với làm mịn Laplace (α ≥ 0)."
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

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        headers = excluded_headers(dataset, params)
        decision = params.get("decision_attr") or headers[-1]
        if decision not in headers:
            raise ValueError(f"Thuộc tính quyết định '{decision}' không tồn tại.")
        features = [h for h in headers if h != decision]
        alpha = float(params.get("laplace_alpha", 1.0))
        rows = [{h: str(row.get(h, "")).strip() for h in headers} for row in dataset.rows]
        n = len(rows)
        classes, class_counts, priors, vocab, value_counts, likelihoods = _prepare_nb_tables(
            rows, features, decision, alpha
        )

        log = self._new_logger()
        steps = _LaplaceNaiveBayesSteps(log)
        steps.prior_step(priors, class_counts, classes, alpha)
        for feat in features:
            steps.likelihood_step(feat, vocab, classes, class_counts, value_counts, likelihoods, alpha)

        self._store_model(
            priors=priors,
            likelihoods=likelihoods,
            class_counts=class_counts,
            value_counts=value_counts,
            vocab=vocab,
            features=features,
            decision=decision,
            alpha=alpha,
            n=n,
            smoothing=True,
        )
        summary = (
            f"Naïve Bayes (Laplace) α = {alpha}. {len(classes)} lớp, {len(features)} thuộc tính. "
            f"Tiên nghiệm: " + ", ".join(f"{c}={fmt_prob(priors[c])}" for c in classes)
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
                "smoothing": "laplace",
                "alpha": alpha,
            },
            summary=summary,
        )
        return self._finish(result)


class ClassicNaiveBayesAlgorithm(_NaiveBayesBase):
    name = "Naïve Bayes (Cổ điển)"
    description = "Phân lớp Naïve Bayes thuần túy — không làm mịn, P(A=v|C) = count / n_C."
    param_schema = {
        "decision_attr": ParamDef(
            type="choice",
            options="headers",
            default=None,
            label_vi="Thuộc tính quyết định (lớp)",
            label_en="Decision attribute (class)",
        ),
    }

    def run(self, dataset: Dataset, params: dict[str, Any]) -> AlgorithmResult:
        headers = excluded_headers(dataset, params)
        decision = params.get("decision_attr") or headers[-1]
        if decision not in headers:
            raise ValueError(f"Thuộc tính quyết định '{decision}' không tồn tại.")
        features = [h for h in headers if h != decision]
        rows = [{h: str(row.get(h, "")).strip() for h in headers} for row in dataset.rows]
        n = len(rows)
        classes, class_counts, priors, vocab, value_counts, likelihoods = _prepare_nb_tables(
            rows, features, decision, alpha=0.0
        )

        log = self._new_logger()
        steps = _ClassicNaiveBayesSteps(log)
        steps.prior_step(priors, class_counts, classes)
        for feat in features:
            steps.likelihood_step(feat, vocab, classes, class_counts, value_counts, likelihoods)

        self._store_model(
            priors=priors,
            likelihoods=likelihoods,
            class_counts=class_counts,
            value_counts=value_counts,
            vocab=vocab,
            features=features,
            decision=decision,
            alpha=0.0,
            n=n,
            smoothing=False,
        )
        summary = (
            f"Naïve Bayes cổ điển (không làm mịn). {len(classes)} lớp, {len(features)} thuộc tính. "
            f"Tiên nghiệm: " + ", ".join(f"{c}={fmt_prob(priors[c])}" for c in classes)
        )
        result = AlgorithmResult(
            algorithm_name=self.name,
            parameters={"decision_attr": decision},
            steps=log.steps,
            output={
                "priors": priors,
                "likelihoods": likelihoods,
                "vocab": vocab,
                "features": features,
                "decision": decision,
                "class_counts": dict(class_counts),
                "smoothing": "none",
            },
            summary=summary,
        )
        return self._finish(result)


# Backward-compatible alias (older imports / docs).
NaiveBayesAlgorithm = LaplaceBayesAlgorithm


class _LaplaceNaiveBayesSteps:
    def __init__(self, log: StepLogger) -> None:
        self._log = log
        self._likelihood_explained = False

    def prior_step(
        self,
        priors: dict[str, float],
        class_counts: Counter,
        classes: list[str],
        alpha: float,
    ) -> None:
        self._log.add_table(
            "Xác suất tiên nghiệm P(C)",
            ["Lớp", "Count", "P(C)"],
            [[cls, class_counts[cls], fmt_prob(priors[cls])] for cls in classes],
            ["left", "right", "right"],
            description=(
                "① Ý tưởng cốt lõi của Naïve Bayes:\n"
                "  Dựa vào Định lý Bayes để phân lớp:\n"
                "    P(C|x) ∝ P(C) × P(x₁|C) × P(x₂|C) × ... × P(xₙ|C)\n"
                '  "Naïve" (ngây thơ) = giả định các thuộc tính độc lập\n'
                "  nhau khi biết lớp — đơn giản hóa mạnh nhưng thường hiệu quả.\n"
                "\n"
                '② Xác suất tiên nghiệm P(C) — "trước khi thấy dữ liệu":\n'
                "  P(C) = |D_C| / |D|  = tỉ lệ mẫu thuộc lớp C.\n"
                "\n"
                f"③ Laplace Smoothing (α = {alpha}):\n"
                "  Tránh xác suất = 0 khi 1 giá trị không xuất hiện trong D.\n"
                "  → Không có α: P(A=v|C) = 0 → xóa sổ toàn bộ lớp C.\n"
                "  → Có α:       P(A=v|C) = (count + α) / (n_C + α×|A|)"
            ),
            level="STEP_HEADER",
            priors=priors,
            class_counts=dict(class_counts),
            alpha=alpha,
            conclusion=f"  → α Laplace = {alpha}",
        )

    def likelihood_step(
        self,
        feat: str,
        vocab: dict[str, list[str]],
        classes: list[str],
        class_counts: Counter,
        value_counts: dict[str, dict[str, dict[str, int]]],
        likelihoods: dict[str, dict[str, dict[str, float]]],
        alpha: float,
    ) -> None:
        like_rows = []
        v = len(vocab[feat])
        example_line = ""
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
                        fmt_prob(p),
                        f"({count}+{alpha})/({n_c}+{alpha}×{v})",
                    ]
                )
                if not example_line:
                    example_line = (
                        f"  Ví dụ: P({feat}={value} | {cls}) = "
                        f"({count}+{alpha})/({n_c}+{alpha}×{v}) = {fmt_prob(p)}"
                    )
        if not self._likelihood_explained:
            self._likelihood_explained = True
            description = (
                f"[Mã giả Bước 2] P(A=v | C) = (count + α) / (n_C + α·|A|)\n"
                "\n"
                '① Likelihood P(A=v|C) — "khả năng có điều kiện":\n'
                "  Nếu ĐÃ BIẾT lớp là C, thì giá trị A=v có xác suất bao nhiêu?\n"
                "\n"
                "② Cách đọc bảng:\n"
                f"{example_line}"
            )
        else:
            description = f"[Mã giả Bước 2] P({feat}=v | C) = (count + α) / (n_C + α·|{feat}|)"
        self._log.add_table(
            f"Bảng likelihood — P({feat}=v | C)",
            ["Lớp", "Giá trị", "Count", "P(v|C)", "Công thức"],
            like_rows,
            ["left", "left", "right", "right", "left"],
            description=description,
            level="DATA",
            feature=feat,
        )


class _ClassicNaiveBayesSteps:
    def __init__(self, log: StepLogger) -> None:
        self._log = log
        self._likelihood_explained = False

    def prior_step(
        self,
        priors: dict[str, float],
        class_counts: Counter,
        classes: list[str],
    ) -> None:
        self._log.add_table(
            "Xác suất tiên nghiệm P(C)",
            ["Lớp", "Count", "P(C)"],
            [[cls, class_counts[cls], fmt_prob(priors[cls])] for cls in classes],
            ["left", "right", "right"],
            description=(
                "① Ý tưởng cốt lõi Naïve Bayes cổ điển:\n"
                "   P(C|x) ∝ P(C) × ∏ P(xᵢ|C)  (giả định độc lập)\n"
                "\n"
                "② Xác suất tiên nghiệm:\n"
                "   P(C) = |D_C| / |D|\n"
                "\n"
                "③ Không có làm mịn:\n"
                "   Nếu P(A=v|C) = 0 → lớp C bị loại hoàn toàn cho mẫu này."
            ),
            level="STEP_HEADER",
            priors=priors,
            class_counts=dict(class_counts),
            alpha=0.0,
            conclusion="  → Không dùng Laplace (α = 0)",
        )

    def likelihood_step(
        self,
        feat: str,
        vocab: dict[str, list[str]],
        classes: list[str],
        class_counts: Counter,
        value_counts: dict[str, dict[str, dict[str, int]]],
        likelihoods: dict[str, dict[str, dict[str, float]]],
    ) -> None:
        like_rows = []
        example_line = ""
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
                        fmt_prob(p),
                        f"{count}/{n_c}",
                    ]
                )
                if not example_line:
                    example_line = (
                        f"  Ví dụ: P({feat}={value} | {cls}) = {count}/{n_c} = {fmt_prob(p)}"
                    )
        if not self._likelihood_explained:
            self._likelihood_explained = True
            description = (
                "[Mã giả Bước 2] P(A=v|C) = count(A=v, C) / |D_C|\n"
                "\n"
                "① Công thức thuần túy (không cộng α):\n"
                "  P(A=v|C) = count / n_C\n"
                "\n"
                "② Cách đọc bảng:\n"
                f"{example_line}"
            )
        else:
            description = f"[Mã giả Bước 2] P({feat}=v | C) = count / n_C"
        self._log.add_table(
            f"Bảng likelihood — P({feat}=v | C)",
            ["Lớp", "Giá trị", "Count", "P(v|C)", "Công thức"],
            like_rows,
            ["left", "left", "right", "right", "left"],
            description=description,
            level="DATA",
            feature=feat,
        )

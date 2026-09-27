from __future__ import annotations

from abc import ABC
from collections import Counter
from typing import Any

from datamining_app.algorithms.base import BaseAlgorithm
from datamining_app.algorithms.dataset_utils import excluded_headers
from datamining_app.steps.naive_bayes_steps import (
    _ClassicNaiveBayesSteps,
    _LaplaceNaiveBayesSteps,
    _build_nb_summary,
    build_predict_explanation,
)
from datamining_app.core.models import AlgorithmResult, Dataset, ParamDef, PredictResult


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
        return build_predict_explanation(
            sample,
            features=self._features,
            decision=self._decision,
            smoothing=self._smoothing,
            alpha=self._alpha,
            priors=self._priors,
            class_counts=self._class_counts,
            n=self._n,
            likelihoods=self._likelihoods,
            value_counts=self._value_counts,
            vocab=self._vocab,
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
        steps.prior_step(priors, class_counts, classes, alpha, n, decision)
        for feat in features:
            steps.likelihood_step(
                feat, vocab, classes, class_counts, value_counts, likelihoods, alpha, decision
            )

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
        summary = _build_nb_summary(
            smoothing=True,
            dataset_name=dataset.name,
            decision=decision,
            classes=classes,
            class_counts=class_counts,
            priors=priors,
            features=features,
            vocab=vocab,
            value_counts=value_counts,
            likelihoods=likelihoods,
            alpha=alpha,
            n=n,
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
    name = "Naïve Bayes (Không làm trơn)"
    description = "Phân lớp Naïve Bayes thuần túy — không làm trơn, P(A=v|C) = count / n_C."
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
        steps.prior_step(priors, class_counts, classes, n, decision)
        for feat in features:
            steps.likelihood_step(feat, vocab, classes, class_counts, value_counts, likelihoods, decision)

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
        summary = _build_nb_summary(
            smoothing=False,
            dataset_name=dataset.name,
            decision=decision,
            classes=classes,
            class_counts=class_counts,
            priors=priors,
            features=features,
            vocab=vocab,
            value_counts=value_counts,
            likelihoods=likelihoods,
            alpha=0.0,
            n=n,
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



def __getattr__(name: str):
    """Keep the old Laplace alias, and warn on use."""
    if name == "NaiveBayesAlgorithm":
        import warnings

        warnings.warn(
            "NaiveBayesAlgorithm is deprecated; use LaplaceBayesAlgorithm.",
            DeprecationWarning,
            stacklevel=2,
        )
        return LaplaceBayesAlgorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name}")


__all__ = ["ClassicNaiveBayesAlgorithm", "LaplaceBayesAlgorithm"]

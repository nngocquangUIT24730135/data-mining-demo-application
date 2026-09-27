"""Naïve Bayes console steps and prediction text.

Likelihood tables are computed in ``algorithms.naive_bayes``.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from datamining_app.algorithms._logger import StepLogger
from datamining_app.console.formatter import display_width
from datamining_app.core.models import PredictResult
from datamining_app.fmt import fmt_alpha, fmt_number, fmt_pct, fmt_prob

def _prior_cell(decision: str, cls: str, count: int, n: int, prior: float) -> str:
    return f"P({decision}={cls}) = {count}/{n} = {fmt_prob(prior)}  ({fmt_pct(prior)})"


def _prior_conclusion(classes: list[str], class_counts: Counter, priors: dict[str, float], n: int) -> str:
    parts = ", ".join(f"{cls}: {class_counts[cls]} dòng" for cls in classes)
    lines = [f"  → Tổng số mẫu |D| = {n}  ({parts})"]
    if len(classes) >= 2:
        majority = max(classes, key=lambda c: priors[c])
        minority = min(classes, key=lambda c: priors[c])
        if majority != minority and priors[minority] > 0:
            ratio = priors[majority] / priors[minority]
            lines.append(
                f'  → Lớp "{majority}" chiếm ưu thế tiên nghiệm: gấp {ratio:.2f} lần lớp "{minority}"'
            )
    return "\n".join(lines)


def _classic_preface(
    feat: str,
    decision: str,
    classes: list[str],
    class_counts: Counter,
) -> str:
    notes = ",  ".join(f"n_{cls} = {class_counts[cls]}" for cls in classes)
    return (
        f"[Mã giả Bước 2] P({feat}=v | {decision}=C) = count({feat}=v, {decision}=C) / n_C\n"
        f"  {notes}"
    )


def _laplace_preface(
    feat: str,
    values: list[str],
    classes: list[str],
    class_counts: Counter,
    alpha: float,
    decision: str,
) -> str:
    v = len(values)
    a = fmt_alpha(alpha)
    lines = [
        f"[Mã giả Bước 2] P({feat}=v | {decision}=C) = (count + α) / (n_C + α·|V|)",
        f"  |V| = {v} giá trị phân biệt: {{{', '.join(values)}}}",
    ]
    for cls in classes:
        n_c = class_counts[cls]
        denom = n_c + alpha * v
        lines.append(
            f"  n_{cls} + α·|V| = {n_c} + {a}×{v} = {fmt_number(denom)}"
            f"   (mẫu số lớp {decision}={cls})"
        )
    return "\n".join(lines)


def _classic_like_cell(
    feat: str, value: str, decision: str, cls: str, count: int, n_c: int, p: float
) -> str:
    text = f"P({feat}={value} | {decision}={cls}) = {count}/{n_c} = {fmt_prob(p)}"
    if p == 0.0:
        text += "  ⚠"
    return text


def _laplace_like_cell(
    feat: str,
    value: str,
    decision: str,
    cls: str,
    count: int,
    n_c: int,
    alpha: float,
    vocab_size: int,
    p: float,
) -> str:
    a = fmt_alpha(alpha)
    num = count + alpha
    den = n_c + alpha * vocab_size
    return (
        f"P({feat}={value} | {decision}={cls}) = ({count}+{a})/({n_c}+{a}×{vocab_size})"
        f" = {fmt_number(num)}/{fmt_number(den)} = {fmt_prob(p)}"
    )


def _zero_freq_conclusion(
    feat: str,
    decision: str,
    zeros: list[tuple[str, str, int, int]],
) -> str:
    if not zeros:
        return ""
    lines: list[str] = []
    for value, cls, _count, _n_c in zeros:
        lines.append(
            f"  ⚠ P({feat}={value} | {decision}={cls}) = 0 → Zero-Frequency: mọi mẫu có {feat}={value}"
        )
        lines.append(f"    sẽ bị loại hoàn toàn khỏi lớp {decision}={cls} khi nhân tích xác suất")
    return "\n".join(lines)


def _zero_likelihoods(
    features: list[str],
    classes: list[str],
    vocab: dict[str, list[str]],
    value_counts: dict[str, dict[str, dict[str, int]]],
    likelihoods: dict[str, dict[str, dict[str, float]]],
    class_counts: Counter,
) -> list[tuple[str, str, str, int, int, float]]:
    found: list[tuple[str, str, str, int, int, float]] = []
    for feat in features:
        for cls in classes:
            for value in vocab[feat]:
                p = likelihoods[cls][feat][value]
                if p == 0.0:
                    found.append(
                        (feat, value, cls, value_counts[cls][feat][value], class_counts[cls], p)
                    )
    return found


def _min_likelihood(
    features: list[str],
    classes: list[str],
    vocab: dict[str, list[str]],
    value_counts: dict[str, dict[str, dict[str, int]]],
    likelihoods: dict[str, dict[str, dict[str, float]]],
    class_counts: Counter,
    alpha: float,
) -> tuple[str, str, str, int, int, int, float] | None:
    best: tuple[str, str, str, int, int, int, float] | None = None
    for feat in features:
        v = len(vocab[feat])
        for cls in classes:
            for value in vocab[feat]:
                p = likelihoods[cls][feat][value]
                count = value_counts[cls][feat][value]
                n_c = class_counts[cls]
                item = (feat, value, cls, count, n_c, v, p)
                if best is None or p < best[6]:
                    best = item
    return best


def _summary_banner(title: str) -> str:
    prefix = f"── {title} "
    fill = max(78 - display_width(prefix), 8)
    return prefix + "─" * fill


def _prior_advantage_line(decision: str, classes: list[str], priors: dict[str, float]) -> str | None:
    if len(classes) < 2:
        return None
    majority = max(classes, key=lambda c: priors[c])
    minority = min(classes, key=lambda c: priors[c])
    if majority == minority or priors[minority] <= 0:
        return None
    ratio = priors[majority] / priors[minority]
    return (
        f"   → Lớp {decision}={majority} chiếm ưu thế tiên nghiệm"
        f" (gấp {ratio:.2f} lần lớp {decision}={minority})"
    )


def _build_nb_summary(
    *,
    smoothing: bool,
    dataset_name: str,
    decision: str,
    classes: list[str],
    class_counts: Counter,
    priors: dict[str, float],
    features: list[str],
    vocab: dict[str, list[str]],
    value_counts: dict[str, dict[str, dict[str, int]]],
    likelihoods: dict[str, dict[str, dict[str, float]]],
    alpha: float,
    n: int,
) -> str:
    if smoothing:
        title = f"KẾT LUẬN MÔ HÌNH NAÏVE BAYES (LAPLACE α = {fmt_alpha(alpha)})"
    else:
        title = "KẾT LUẬN MÔ HÌNH NAÏVE BAYES (KHÔNG LÀM TRƠN)"

    lines: list[str] = [
        _summary_banner(title),
        "",
        f"Dữ liệu : {dataset_name} ({n} dòng · {len(features)} thuộc tính · cột mục tiêu: {decision})",
        "",
        "① Xác suất tiên nghiệm:",
    ]
    for cls in classes:
        count = class_counts[cls]
        lines.append(f"   P({decision}={cls}) = {count}/{n} = {fmt_prob(priors[cls])}  ({fmt_pct(priors[cls])})")
    advantage = _prior_advantage_line(decision, classes, priors)
    if advantage:
        lines.append(advantage)

    lines.append("")
    if smoothing:
        lines.append(f"② Hiệu quả của Laplace Smoothing (α = {fmt_alpha(alpha)}):")
        lines.append("   Tất cả xác suất điều kiện đều > 0 — không có Zero-Frequency.")
        smallest = _min_likelihood(
            features, classes, vocab, value_counts, likelihoods, class_counts, alpha
        )
        if smallest is not None:
            feat, value, cls, count, n_c, v, p = smallest
            classic = (count / n_c) if n_c else 0.0
            lines.append(
                "   Nhỏ nhất: "
                f"P({feat}={value} | {decision}={cls}) = "
                f"{fmt_number(count + alpha)}/{fmt_number(n_c + alpha * v)} = {fmt_prob(p)}"
            )
            classic_note = f"{count}/{n_c} = {fmt_prob(classic)}"
            if count == 0:
                classic_note += f" → loại lớp {decision}={cls}"
            lines.append(f"              (nếu không làm trơn → {classic_note})")
        lines.append("   → Mô hình ổn định với mọi tổ hợp giá trị thuộc tính.")
    else:
        lines.append("② Cảnh báo — Vấn đề tần suất bằng 0 (Zero-Frequency Problem):")
        zeros = _zero_likelihoods(features, classes, vocab, value_counts, likelihoods, class_counts)
        if zeros:
            lines.append(f"   Phát hiện {len(zeros)} xác suất điều kiện = 0:")
            for feat, value, cls, count, n_c, p in zeros:
                lines.append(
                    f"      P({feat}={value} | {decision}={cls}) = {count}/{n_c} = {fmt_prob(p)}"
                )
            feat0, value0, cls0, _, _, _ = zeros[0]
            lines.append(
                f"   Hệ quả: P({decision}={cls0}) × P({feat0}={value0} | {decision}={cls0}) × ... = 0"
            )
            lines.append(
                f"   → Mô hình KHÔNG BAO GIỜ phân loại mẫu có {feat0}={value0} vào lớp {decision}={cls0},"
            )
            lines.append("     dù các thuộc tính còn lại hoàn toàn phù hợp.")
            lines.append("   Khuyến nghị: Dùng Naïve Bayes Laplace (α > 0) để khắc phục.")
        else:
            lines.append("   Không phát hiện xác suất điều kiện = 0 trên tập huấn luyện.")
            lines.append("   Vẫn nên thận trọng: giá trị mới lúc dự đoán vẫn có thể tạo P = 0.")

    lines.extend(
        [
            "",
            "③ Công thức phân loại mẫu mới X = (x₁, x₂, ..., xₙ):",
            f"   C* = argmax_C  P({decision}=C) × ∏ᵢ P(Aᵢ=xᵢ | {decision}=C)",
        ]
    )
    return "\n".join(lines)


class _LaplaceNaiveBayesSteps:
    """Ghi bước Naïve Bayes Laplace vào StepLogger từ tiên nghiệm và likelihood đã tính."""
    def __init__(self, log: StepLogger) -> None:
        self._log = log

    def prior_step(
        self,
        priors: dict[str, float],
        class_counts: Counter,
        classes: list[str],
        alpha: float,
        n: int,
        decision: str,
    ) -> None:
        self._log.add_table(
            "Xác suất tiên nghiệm P(C)",
            ["Lớp", "Count", "P(C) = |D_C| / |D|"],
            [
                [cls, class_counts[cls], _prior_cell(decision, cls, class_counts[cls], n, priors[cls])]
                for cls in classes
            ],
            ["left", "right", "left"],
            description=(
                "① Ý tưởng cốt lõi của Naïve Bayes:\n"
                f"  P({decision}=C | x) tỉ lệ với P({decision}=C) × ∏ᵢ P(Aᵢ=xᵢ | {decision}=C)\n"
                "  (P(x) giống nhau với mọi lớp → chọn lớp có tích lớn nhất)\n"
                '  "Naïve" = giả định các thuộc tính độc lập khi biết lớp.\n'
                "\n"
                f"② Xác suất tiên nghiệm: P({decision}=C) = |D_C| / |D|.\n"
                "\n"
                f"③ Laplace Smoothing (α = {fmt_alpha(alpha)}):\n"
                f"  P(A=v | {decision}=C) = (count + α) / (n_C + α·|V|)\n"
                "  → Mọi likelihood > 0, không xóa sổ một lớp."
            ),
            level="STEP_HEADER",
            priors=priors,
            class_counts=dict(class_counts),
            alpha=alpha,
            conclusion=_prior_conclusion(classes, class_counts, priors, n),
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
        decision: str,
    ) -> None:
        like_rows = []
        v = len(vocab[feat])
        for cls in classes:
            n_c = class_counts[cls]
            for value in vocab[feat]:
                count = value_counts[cls][feat][value]
                p = likelihoods[cls][feat][value]
                cell = _laplace_like_cell(feat, value, decision, cls, count, n_c, alpha, v, p)
                like_rows.append([cls, value, count, cell])
        self._log.add_table(
            f"Bảng likelihood — P({feat}=v | {decision}=C)",
            ["Lớp", "Giá trị", "Count", f"P({feat}=v | {decision}=C) = (count+α)/(n_C+α·|V|)"],
            like_rows,
            ["left", "left", "right", "left"],
            description=_laplace_preface(feat, vocab[feat], classes, class_counts, alpha, decision),
            level="DATA",
            feature=feat,
        )


class _ClassicNaiveBayesSteps:
    """Ghi bước Naïve Bayes không làm trơn vào StepLogger từ tiên nghiệm và likelihood đã tính."""
    def __init__(self, log: StepLogger) -> None:
        self._log = log

    def prior_step(
        self,
        priors: dict[str, float],
        class_counts: Counter,
        classes: list[str],
        n: int,
        decision: str,
    ) -> None:
        self._log.add_table(
            "Xác suất tiên nghiệm P(C)",
            ["Lớp", "Count", "P(C) = |D_C| / |D|"],
            [
                [cls, class_counts[cls], _prior_cell(decision, cls, class_counts[cls], n, priors[cls])]
                for cls in classes
            ],
            ["left", "right", "left"],
            description=(
                "① Ý tưởng cốt lõi Naïve Bayes (không làm trơn):\n"
                f"   P({decision}=C | x) tỉ lệ với P({decision}=C) × ∏ᵢ P(Aᵢ=xᵢ | {decision}=C)\n"
                "   (P(x) giống nhau với mọi lớp → chọn lớp có tích lớn nhất)\n"
                "\n"
                "② Xác suất tiên nghiệm:\n"
                f"   P({decision}=C) = |D_C| / |D|\n"
                "\n"
                "③ Không làm trơn:\n"
                f"   Nếu P(A=v | {decision}=C) = 0 → lớp {decision}=C bị loại hoàn toàn cho mẫu này."
            ),
            level="STEP_HEADER",
            priors=priors,
            class_counts=dict(class_counts),
            alpha=0.0,
            conclusion=_prior_conclusion(classes, class_counts, priors, n),
        )

    def likelihood_step(
        self,
        feat: str,
        vocab: dict[str, list[str]],
        classes: list[str],
        class_counts: Counter,
        value_counts: dict[str, dict[str, dict[str, int]]],
        likelihoods: dict[str, dict[str, dict[str, float]]],
        decision: str,
    ) -> None:
        like_rows = []
        zeros: list[tuple[str, str, int, int]] = []
        for cls in classes:
            n_c = class_counts[cls]
            for value in vocab[feat]:
                count = value_counts[cls][feat][value]
                p = likelihoods[cls][feat][value]
                cell = _classic_like_cell(feat, value, decision, cls, count, n_c, p)
                like_rows.append([cls, value, count, cell])
                if p == 0.0:
                    zeros.append((value, cls, count, n_c))
        self._log.add_table(
            f"Bảng likelihood — P({feat}=v | {decision}=C)",
            ["Lớp", "Giá trị", "Count", f"P({feat}=v | {decision}=C) = count / n_C"],
            like_rows,
            ["left", "left", "right", "left"],
            description=_classic_preface(feat, decision, classes, class_counts),
            level="DATA",
            feature=feat,
            conclusion=_zero_freq_conclusion(feat, decision, zeros),
        )

def build_predict_explanation(
    sample: dict[str, Any],
    *,
    features: list[str],
    decision: str,
    smoothing: bool,
    alpha: float,
    priors: dict[str, float],
    class_counts: dict[str, int],
    n: int,
    likelihoods: dict[str, dict[str, dict[str, float]]],
    value_counts: dict[str, dict[str, dict[str, int]]],
    vocab: dict[str, list[str]],
) -> PredictResult:
    """Score one sample and build its console explanation."""
    normalized = {k: str(v).strip() for k, v in sample.items()}
    products: dict[str, float] = {}
    killed: dict[str, str | None] = {}

    sample_bits = [f"{feat} = {normalized.get(feat, '')}" for feat in features]
    lines = [
        "① Mẫu cần dự đoán:",
        "   " + ", ".join(sample_bits),
        "",
        "② Ý tưởng: với mỗi lớp C, nhân tiên nghiệm với các likelihood,",
        "   rồi chọn lớp có tích lớn nhất.",
    ]
    if smoothing:
        lines.append(
            f"   Biến thể: Laplace (α = {fmt_alpha(alpha)}) — mọi P(Aᵢ=xᵢ | {decision}=C) > 0."
        )
    else:
        lines.append(
            f"   Biến thể: không làm trơn — nếu có P(Aᵢ=xᵢ | {decision}=C) = 0 thì cả lớp bị loại."
        )
    lines.append("")
    lines.append("③ Tính điểm theo từng lớp:")

    for cls, prior in priors.items():
        n_c = class_counts[cls]
        factors = [prior]
        zero_reason: str | None = None
        lines.append("")
        lines.append(f"── Lớp {cls} ──")
        lines.append(
            f"  P({decision}={cls}) = {n_c}/{n} = {fmt_prob(prior)}  ({fmt_pct(prior)})"
        )

        for feat in features:
            value = normalized.get(feat, "")
            table = likelihoods[cls][feat]
            counts = value_counts[cls][feat]
            if value in table:
                p = table[value]
                count = counts.get(value, 0)
                if smoothing:
                    v = len(vocab[feat])
                    a = fmt_alpha(alpha)
                    extra = alpha * v
                    num = count + alpha
                    den = n_c + extra
                    formula = (
                        f"P({feat}={value} | {decision}={cls}) = ({count} + {a})/({n_c} + {fmt_number(extra)})"
                        f" = {fmt_number(num)}/{fmt_number(den)} = {fmt_prob(p)}"
                    )
                else:
                    formula = (
                        f"P({feat}={value} | {decision}={cls}) = {count}/{n_c} = {fmt_prob(p)}"
                    )
            elif smoothing:
                v = len(vocab[feat])
                denom = n_c + alpha * v
                p = alpha / denom if denom else 0.0
                a = fmt_alpha(alpha)
                formula = (
                    f"P({feat}={value} | {decision}={cls}) = (0 + {a})/({n_c} + {fmt_number(alpha * v)})"
                    f" = {fmt_number(alpha)}/{fmt_number(denom)} = {fmt_prob(p)}"
                    "  ← giá trị chưa thấy lúc huấn luyện (Laplace vẫn > 0)"
                )
            else:
                p = 0.0
                formula = (
                    f"P({feat}={value} | {decision}={cls}) = 0/{n_c} = {fmt_prob(p)}"
                    "  ← không có trong lớp / giá trị mới"
                )
            lines.append(f"  {formula}")
            factors.append(p)
            if p <= 0 and zero_reason is None:
                zero_reason = f"{feat} = {value}"

        killed[cls] = zero_reason
        if zero_reason is not None:
            products[cls] = 0.0
            lines.append(
                f"  → Tích = 0 vì P(...|{decision}={cls}) = 0 tại «{zero_reason}»."
            )
            lines.append(f"  → Lớp {cls} bị loại hoàn toàn cho mẫu này.")
        else:
            shown = [fmt_prob(f) for f in factors]
            product = math.prod(float(piece) for piece in shown)
            products[cls] = product
            lines.append("  → Tích:")
            lines.append("     " + " × ".join(shown))
            lines.append(f"     = {fmt_prob(product)}")

    total = sum(products.values())
    posteriors = {c: (products[c] / total if total else 0.0) for c in products}
    label = max(products, key=products.get)

    lines.append("")
    rivals_killed = [c for c in products if c != label and killed.get(c)]
    if rivals_killed and not killed.get(label):
        lines.append(
            f"④ Kết luận: dự đoán «{decision} = {label}» "
            f"vì lớp {', '.join(rivals_killed)} bị triệt tiêu (có P=0), "
            f"chỉ còn lớp {label}."
        )
    else:
        others = [c for c in products if c != label]
        winner = f"tích lớp {decision}={label} = {fmt_prob(products[label])}"
        if others:
            compared = " và ".join(
                f"tích lớp {decision}={c} = {fmt_prob(products[c])}" for c in others
            )
            reason = f"{winner} lớn hơn {compared}"
        else:
            reason = f"{winner} là lớp duy nhất"
        lines.append(
            f"④ Kết luận: dự đoán «{decision} = {label}» vì {reason}."
        )

    return PredictResult(
        label=label,
        explanation="\n".join(lines),
        details={"posteriors": posteriors, "scores": products},
        sample=normalized,
    )



__all__ = [
    "_ClassicNaiveBayesSteps",
    "_LaplaceNaiveBayesSteps",
    "_build_nb_summary",
    "build_predict_explanation",
]

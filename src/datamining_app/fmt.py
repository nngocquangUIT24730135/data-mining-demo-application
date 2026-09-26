"""Centralized number formatting — single source of truth for all precision."""
from __future__ import annotations

# ── Precision constants (Chuẩn hóa 10 chữ số thập phân) ──────────────────────
PROB_DECIMALS = 10  # Xác suất, likelihood, priors, posteriors → P(C|x) = 0.xxxxxxxxxx
SCORE_DECIMALS = 10  # Entropy, Gini, Information Gain, ΔGini   → H(S)   = 0.xxxxxxxxxx
DIST_DECIMALS = 10  # Khoảng cách Euclid, Manhattan            → d(x,μ) = 0.xxxxxxxxxx
INERTIA_DECIMALS = 10  # SSE / SAE trong K-Means                 → Inertia= 0.xxxxxxxxxx
SUPPORT_DECIMALS = 10  # Support / Confidence tỷ lệ thập phân    → ratio  = 0.xxxxxxxxxx
PCT_DECIMALS = 2  # Tỷ lệ % giao diện tóm tắt tham số       → minsup = 20.00%


# ── Formatter functions ─────────────────────────────────────────────────────
def fmt_prob(v: float) -> str:
    """Format xác suất (10 chữ số thập phân)."""
    return f"{v:.{PROB_DECIMALS}f}"


def fmt_score(v: float) -> str:
    """Format độ đo hỗn loạn, gain, entropy, gini (10 chữ số thập phân)."""
    return f"{v:.{SCORE_DECIMALS}f}"


def fmt_score_diff(minuend: float, subtrahend: float) -> str:
    """Format phép trừ minuend − subtrahend đồng bộ từng chữ số hiển thị."""
    a = float(fmt_score(minuend))
    b = float(fmt_score(subtrahend))
    return fmt_score(a - b)


def fmt_dist(v: float) -> str:
    """Format khoảng cách (10 chữ số thập phân)."""
    return f"{v:.{DIST_DECIMALS}f}"


def fmt_inertia(v: float) -> str:
    """Format SSE/SAE (10 chữ số thập phân)."""
    return f"{v:.{INERTIA_DECIMALS}f}"


def fmt_support(v: float) -> str:
    """Format support/confidence tỷ lệ (10 chữ số thập phân)."""
    return f"{v:.{SUPPORT_DECIMALS}f}"


def fmt_pct(v: float, decimals: int = PCT_DECIMALS) -> str:
    """Format hiển thị phần trăm con người đọc."""
    return f"{v * 100:.{decimals}f}%"


def fmt_centroid(coords: list[float] | tuple[float, ...]) -> str:
    """Format tọa độ trọng tâm (mỗi chiều 10 chữ số thập phân)."""
    return "(" + ", ".join(f"{v:.{DIST_DECIMALS}f}" for v in coords) + ")"

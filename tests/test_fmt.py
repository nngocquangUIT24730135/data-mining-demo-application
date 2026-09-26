from datamining_app.fmt import (
    DIST_DECIMALS,
    INERTIA_DECIMALS,
    PROB_DECIMALS,
    SCORE_DECIMALS,
    SUPPORT_DECIMALS,
    fmt_prob,
    fmt_score,
    fmt_score_diff,
    fmt_support,
)


def test_precision_constants_are_ten():
    assert PROB_DECIMALS == 10
    assert SCORE_DECIMALS == 10
    assert DIST_DECIMALS == 10
    assert INERTIA_DECIMALS == 10
    assert SUPPORT_DECIMALS == 10


def test_fmt_outputs_ten_fractional_digits():
    assert fmt_prob(9 / 14) == "0.6428571429"
    assert fmt_score(0.9400296572) == "0.9400296572"
    assert fmt_support(3 / 7) == "0.4285714286"
    assert fmt_score_diff(0.9400296572, 0.9110633931) == "0.0289662641"

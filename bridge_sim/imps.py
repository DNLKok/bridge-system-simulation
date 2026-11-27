from __future__ import annotations

from bisect import bisect_right


_IMP_THRESHOLDS = (
    20,
    50,
    90,
    130,
    170,
    220,
    270,
    320,
    370,
    430,
    500,
    600,
    750,
    900,
    1100,
    1300,
    1500,
    1750,
    2000,
    2250,
    2500,
    3000,
    3500,
    4000,
)


def calculate_imps(score_difference_ns: int) -> int:
    """Convert an NS score difference to IMPs using the standard IMP scale."""
    if (
        not isinstance(score_difference_ns, int)
        or isinstance(score_difference_ns, bool)
    ):
        raise ValueError("score_difference_ns must be an integer.")

    magnitude = bisect_right(_IMP_THRESHOLDS, abs(score_difference_ns))
    return magnitude if score_difference_ns >= 0 else -magnitude

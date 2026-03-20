"""
Horizon confirmation utilities — 2d follow-through, 10d persistence, 120d secular.
"""


def compute_follow_through_quality(
    rs_2d: float, rs_5d: float, direction: int,
) -> tuple[str, int]:
    """
    2d follow-through confirmation.

    Returns
    -------
    (quality, confidence_modifier)

    direction=+1 (long):
      rs_2d > 0.001 AND same sign as rs_5d  -> ("confirmed", +5)
      abs(rs_2d) <= 0.001                    -> ("uncertain", 0)
      rs_2d < -0.001 AND contradicts rs_5d   -> ("failed", -10)

    direction=-1 (short):
      Reversed: rs_2d < -0.001 AND same sign as rs_5d -> ("confirmed", +5)
      etc.

    direction=0: always ("neutral", 0)
    """
    if direction == 0:
        return ("neutral", 0)

    if direction == 1:
        if rs_2d > 0.001 and rs_5d > 0:
            return ("confirmed", 5)
        if abs(rs_2d) <= 0.001:
            return ("uncertain", 0)
        if rs_2d < -0.001 and rs_5d > 0:
            return ("failed", -10)
        # rs_2d beyond threshold but rs_5d doesn't confirm/contradict clearly
        if rs_2d > 0.001:
            return ("uncertain", 0)
        return ("failed", -10)

    # direction == -1 (short)
    if rs_2d < -0.001 and rs_5d < 0:
        return ("confirmed", 5)
    if abs(rs_2d) <= 0.001:
        return ("uncertain", 0)
    if rs_2d > 0.001 and rs_5d < 0:
        return ("failed", -10)
    # rs_2d beyond threshold but rs_5d doesn't confirm/contradict clearly
    if rs_2d < -0.001:
        return ("uncertain", 0)
    return ("failed", -10)


def compute_persistence_quality(
    rs_10d: float, rs_20d: float,
) -> tuple[str, int]:
    """
    10d persistence check.

    Returns
    -------
    (quality, confidence_modifier)

    Both same sign (both > 0.001 or both < -0.001) -> ("persisting", +5)
    rs_10d near zero (abs <= 0.003)                 -> ("stalling", -5)
    Opposite sign                                   -> ("reversing", -15)
    """
    if abs(rs_10d) <= 0.003:
        return ("stalling", -5)

    both_positive = rs_10d > 0.001 and rs_20d > 0.001
    both_negative = rs_10d < -0.001 and rs_20d < -0.001

    if both_positive or both_negative:
        return ("persisting", 5)

    return ("reversing", -15)


def compute_secular_alignment(
    rs_120d: float, direction: int,
) -> tuple[str, int]:
    """
    120d secular context.

    Returns
    -------
    (alignment, confidence_modifier)

    direction=+1: rs_120d > 0.01  -> ("with-secular", +5)
                  abs(rs_120d) <= 0.01 -> ("neutral-secular", 0)
                  rs_120d < -0.01 -> ("counter-secular", -15)

    direction=-1: reversed
    direction=0: always ("neutral-secular", 0)
    """
    if direction == 0:
        return ("neutral-secular", 0)

    if direction == 1:
        if rs_120d > 0.01:
            return ("with-secular", 5)
        if abs(rs_120d) <= 0.01:
            return ("neutral-secular", 0)
        return ("counter-secular", -15)

    # direction == -1
    if rs_120d < -0.01:
        return ("with-secular", 5)
    if abs(rs_120d) <= 0.01:
        return ("neutral-secular", 0)
    return ("counter-secular", -15)

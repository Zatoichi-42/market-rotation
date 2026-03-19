"""
Turnover Filter — Prevents rotation for marginal improvement.

Rules:
1. Candidate Pump delta must exceed current by >= min_delta_advantage
   for >= min_persistence_sessions consecutive sessions
2. OR current holding is in an exempt state (Exhaustion, Ambiguous)
"""
from engine.schemas import TurnoverCheck, AnalysisState


def check_turnover(
    candidate_ticker: str,
    current_ticker: str,
    pump_deltas: dict[str, list[float]],
    current_states: dict[str, AnalysisState],
    settings: dict,
) -> TurnoverCheck:
    """Should we rotate from current_ticker to candidate_ticker?"""
    min_advantage = settings.get("min_delta_advantage", 0.08)
    min_persistence = settings.get("min_persistence_sessions", 3)
    exempt_state_names = settings.get("exempt_states", ["Exhaustion", "Ambiguous"])

    cand_deltas = pump_deltas.get(candidate_ticker, [])
    curr_deltas = pump_deltas.get(current_ticker, [])

    # Current delta advantage (latest)
    if cand_deltas and curr_deltas:
        delta_adv = cand_deltas[-1] - curr_deltas[-1]
    elif cand_deltas:
        delta_adv = cand_deltas[-1]
    else:
        delta_adv = 0.0

    # Count consecutive sessions where advantage >= min_advantage
    min_len = min(len(cand_deltas), len(curr_deltas))
    persistence = 0
    if min_len > 0:
        for i in range(min_len - 1, -1, -1):
            adv = cand_deltas[i] - curr_deltas[i]
            if adv >= min_advantage:
                persistence += 1
            else:
                break

    # Check exempt state
    current_state = current_states.get(current_ticker)
    is_exempt = False
    if current_state is not None:
        state_val = current_state.value if isinstance(current_state, AnalysisState) else str(current_state)
        is_exempt = state_val in exempt_state_names

    # Decision
    if is_exempt:
        passes = True
        reason = (
            f"PASS (exempt): {current_ticker} is in {state_val} — "
            f"exempt from turnover threshold. Candidate {candidate_ticker} "
            f"has delta advantage {delta_adv:+.3f}."
        )
    elif delta_adv >= min_advantage and persistence >= min_persistence:
        passes = True
        reason = (
            f"PASS: {candidate_ticker} Pump delta exceeds {current_ticker} by "
            f"{delta_adv:+.3f} for {persistence} consecutive sessions "
            f"(min required: {min_advantage} for {min_persistence} sessions)."
        )
    else:
        passes = False
        if delta_adv < min_advantage:
            reason = (
                f"FAIL: {candidate_ticker} delta advantage {delta_adv:+.3f} "
                f"below threshold {min_advantage}. Do not rotate."
            )
        else:
            reason = (
                f"FAIL: {candidate_ticker} delta advantage {delta_adv:+.3f} meets threshold "
                f"but only persisted {persistence} sessions (min required: {min_persistence})."
            )

    return TurnoverCheck(
        candidate_ticker=candidate_ticker,
        current_ticker=current_ticker,
        delta_advantage=delta_adv,
        persistence_sessions=persistence,
        current_state_exempt=is_exempt,
        passes_filter=passes,
        reason=reason,
    )


def find_rotation_candidates(
    current_holdings: list[str],
    all_groups: list[str],
    pump_deltas: dict[str, list[float]],
    current_states: dict[str, AnalysisState],
    settings: dict,
) -> list[TurnoverCheck]:
    """Find groups that pass the turnover filter for any current holding."""
    passing = []
    for current in current_holdings:
        for candidate in all_groups:
            if candidate == current:
                continue
            result = check_turnover(candidate, current, pump_deltas, current_states, settings)
            if result.passes_filter:
                passing.append(result)

    # Sort by delta advantage descending
    passing.sort(key=lambda tc: tc.delta_advantage, reverse=True)
    return passing

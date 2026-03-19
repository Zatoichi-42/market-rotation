"""
Pump Score — Composite scoring for sector momentum assessment.

Computes a weighted blend of three pillars:
- RS pillar (0-100): relative strength percentile
- Participation pillar (0-100): breadth/volume confirmation
- Flow pillar (0-100): money flow / CLV proxy

Pump Score = weighted average of pillars, scaled to [0, 1].
Pump Delta = session-over-session change in score.
Pump Delta 5d Avg = 5-session rolling average of delta.
"""
from engine.schemas import PumpScoreReading


def compute_pump_score(
    rs_pillar: float,
    participation_pillar: float,
    flow_pillar: float,
    weights: dict,
) -> float:
    """
    Compute pump score from three pillar values (each 0-100).

    Returns score in [0, 1].
    """
    weighted = (
        weights["rs_weight"] * rs_pillar
        + weights["participation_weight"] * participation_pillar
        + weights["flow_weight"] * flow_pillar
    )
    # Pillars are 0-100, divide by 100 to get 0-1
    score = weighted / 100.0
    return max(0.0, min(1.0, score))


def build_readings_from_score_history(
    ticker: str,
    name: str,
    scores: list[float],
) -> list[PumpScoreReading]:
    """
    Build a sequence of PumpScoreReadings from a list of pre-computed scores.
    Computes delta and 5d rolling average of delta.

    Used for testing and for building historical sequences.
    """
    readings = []
    deltas = []

    for i, score in enumerate(scores):
        if i == 0:
            delta = 0.0
        else:
            delta = score - scores[i - 1]

        deltas.append(delta)

        # 5d rolling average of delta
        window = deltas[-5:] if len(deltas) >= 5 else deltas
        delta_5d_avg = sum(window) / len(window)

        readings.append(PumpScoreReading(
            ticker=ticker,
            name=name,
            rs_pillar=0.0,  # Not tracked in score-history mode
            participation_pillar=0.0,
            flow_pillar=0.0,
            pump_score=score,
            pump_delta=delta,
            pump_delta_5d_avg=delta_5d_avg,
        ))

    return readings


def compute_pump_scores_all(
    pillar_data: dict[str, list[tuple[float, float, float]]],
    sector_names: dict[str, str],
    weights: dict,
) -> list[PumpScoreReading]:
    """
    Compute pump scores for all sectors across multiple sessions.

    pillar_data: {ticker: [(rs, participation, flow), ...]} per session
    Returns the LATEST PumpScoreReading per sector (with delta computed from history).
    """
    results = []

    for ticker in sector_names:
        sessions = pillar_data.get(ticker, [])
        if not sessions:
            results.append(PumpScoreReading(
                ticker=ticker,
                name=sector_names[ticker],
                rs_pillar=0.0,
                participation_pillar=0.0,
                flow_pillar=0.0,
                pump_score=0.0,
                pump_delta=0.0,
                pump_delta_5d_avg=0.0,
            ))
            continue

        # Compute score for each session
        scores = []
        for rs, part, flow in sessions:
            score = compute_pump_score(rs, part, flow, weights)
            scores.append(score)

        # Build delta history
        deltas = []
        for i in range(len(scores)):
            if i == 0:
                deltas.append(0.0)
            else:
                deltas.append(scores[i] - scores[i - 1])

        # Latest values
        latest_rs, latest_part, latest_flow = sessions[-1]
        latest_score = scores[-1]
        latest_delta = deltas[-1]

        window = deltas[-5:] if len(deltas) >= 5 else deltas
        delta_5d_avg = sum(window) / len(window)

        results.append(PumpScoreReading(
            ticker=ticker,
            name=sector_names[ticker],
            rs_pillar=latest_rs,
            participation_pillar=latest_part,
            flow_pillar=latest_flow,
            pump_score=latest_score,
            pump_delta=latest_delta,
            pump_delta_5d_avg=delta_5d_avg,
        ))

    return results

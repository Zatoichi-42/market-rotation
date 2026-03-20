"""
Baton Pass Alert — detects when one group is overtaking another.

Checks delta histories: if candidate delta exceeds current leader's delta
by a sustained margin, flags a rotation signal.

Also checks if the signal is confirmed at both sector AND industry level (DUAL LEVEL).
"""
import streamlit as st


def render_baton_pass_alerts(result: dict):
    """Detect and display baton-pass alerts from pump delta histories."""
    pumps = result.get("pumps", {})
    states = result.get("states", {})
    industry_rs = result.get("industry_rs", [])

    if not pumps:
        return

    # Build delta and score data
    sectors = sorted(pumps.keys())
    sector_data = []
    for t in sectors:
        p = pumps[t]
        s = states.get(t)
        # Get RS rank from rs_readings if available
        rs_rank = 11
        for rr in result.get("rs_readings", []):
            if rr.ticker == t:
                rs_rank = rr.rs_rank
                break
        sector_data.append({
            "ticker": t, "name": p.name,
            "pump_score": p.pump_score, "pump_delta": p.pump_delta,
            "pump_delta_5d": p.pump_delta_5d_avg,
            "state": s.state.value if s else "—",
            "rs_rank": rs_rank,
        })

    # Sort by pump delta descending
    sector_data.sort(key=lambda x: x["pump_delta"], reverse=True)

    # Find baton passes: rising group overtaking a declining group
    alerts = []
    rising = [d for d in sector_data if d["pump_delta"] > 0.005]
    declining = [d for d in sector_data if d["pump_delta"] < -0.005]

    for r in rising:
        for d in declining:
            delta_diff = r["pump_delta"] - d["pump_delta"]
            if (delta_diff >= 0.08 and r.get("rs_rank", 11) <= 5 and d.get("rs_rank", 0) >= 4):  # Meaningful gap
                # Check if confirmed at industry level
                industry_confirmed = _check_industry_confirmation(
                    r["ticker"], d["ticker"], industry_rs)

                level = "DUAL LEVEL" if industry_confirmed else "SECTOR LEVEL"
                confirms = "Sector ✓ Industry ✓" if industry_confirmed else "Sector ✓ Industry —"

                alerts.append({
                    "rising": r["ticker"],
                    "rising_name": r["name"],
                    "declining": d["ticker"],
                    "declining_name": d["name"],
                    "delta_diff": delta_diff,
                    "rising_delta": r["pump_delta"],
                    "declining_delta": d["pump_delta"],
                    "rising_state": r["state"],
                    "declining_state": d["state"],
                    "level": level,
                    "confirms": confirms,
                })

    if not alerts:
        return

    st.subheader("Baton Pass Alerts")
    for a in alerts[:3]:  # Show top 3
        color = "#22c55e" if "DUAL" in a["level"] else "#eab308"
        st.markdown(
            f"<div style='border-left: 4px solid {color}; padding: 8px 12px; "
            f"margin-bottom: 8px; background: rgba(0,0,0,0.2); border-radius: 4px;'>"
            f"<strong style='font-size: 1.1em;'>BATON PASS ALERT: "
            f"{a['rising']} ({a['rising_name']}) overtaking "
            f"{a['declining']} ({a['declining_name']})</strong><br>"
            f"Delta diff: <strong>{a['delta_diff']:+.3f}</strong> "
            f"({a['rising']} Δ={a['rising_delta']:+.3f} vs "
            f"{a['declining']} Δ={a['declining_delta']:+.3f})<br>"
            f"States: {a['rising']}={a['rising_state']}, "
            f"{a['declining']}={a['declining_state']}<br>"
            f"Confirmed: {a['confirms']} → <strong>{a['level']}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _check_industry_confirmation(rising_sector, declining_sector, industry_rs):
    """
    Check if any industry within the rising sector is also rising
    and any within the declining sector is also declining.
    """
    if not industry_rs:
        return False

    rising_children = [r for r in industry_rs if r.parent_sector == rising_sector]
    declining_children = [r for r in industry_rs if r.parent_sector == declining_sector]

    has_rising_child = any(r.rs_slope > 0 for r in rising_children) if rising_children else False
    has_declining_child = any(r.rs_slope < 0 for r in declining_children) if declining_children else False

    return has_rising_child and has_declining_child

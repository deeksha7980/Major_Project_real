"""
risk_service.py — Explainable Multimodal Risk Engine.
Calculates 0-100 risk scores with transparent contributing factor breakdowns.
"""

def calculate_cow_risk(profile, cow_events, cow_anomalies, cow_interactions):
    """
    Calculates transparent risk score (0-100) for an individual cow.
    Formula:
    Risk = Movement_Contrib + Behavior_Contrib + Audio_Contrib + Multimodal_Contrib + Proximity_Contrib
    """
    factors = []

    # 1. Movement Contribution (Max 25 pts)
    mov_score = profile.get("movement_score", 0.0)
    mov_contrib = min(25.0, round(mov_score * 15.0, 1))
    if mov_contrib > 5.0:
        factors.append(f"+{mov_contrib:.0f} unusual movement")

    # 2. Behavior Contribution (Max 20 pts)
    agg_pct = profile.get("aggressive_pct", 0.0)
    trans_cnt = profile.get("transitions_count", 0)
    beh_contrib = min(20.0, round((agg_pct * 0.2) + (trans_cnt * 2.0), 1))
    if beh_contrib > 5.0:
        factors.append(f"+{beh_contrib:.0f} behavior deviation")

    # 3. Audio Contribution (Max 25 pts)
    audio_cnt = profile.get("audio_event_count", 0)
    audio_contrib = min(25.0, round(audio_cnt * 12.5, 1))
    if audio_contrib > 5.0:
        factors.append(f"+{audio_contrib:.0f} high-frequency call events")

    # 4. Multimodal Correlation Contribution (Max 20 pts)
    multi_cnt = len(cow_events)
    multi_contrib = min(20.0, round(multi_cnt * 10.0, 1))
    if multi_contrib > 5.0:
        factors.append(f"+{multi_contrib:.0f} multimodal temporal correlation")

    # 5. Spatial Proximity Contribution (Max 10 pts)
    inter_cnt = len(cow_interactions)
    prox_contrib = min(10.0, round(inter_cnt * 2.5, 1))
    if prox_contrib > 2.0:
        factors.append(f"+{prox_contrib:.0f} high proximity interactions")

    total_risk = round(min(100.0, mov_contrib + beh_contrib + audio_contrib + multi_contrib + prox_contrib), 1)

    if total_risk <= 29.0:
        status = "NORMAL"
    elif total_risk <= 59.0:
        status = "LOW ATTENTION"
    elif total_risk <= 79.0:
        status = "ELEVATED"
    else:
        status = "HIGH"

    contributor_str = ", ".join(factors) if factors else "Normal activity profile"

    return {
        "risk_score": total_risk,
        "status": status,
        "contributing_factors": factors,
        "contributor_str": contributor_str,
        "breakdown": {
            "movement": mov_contrib,
            "behavior": beh_contrib,
            "audio": audio_contrib,
            "multimodal": multi_contrib,
            "proximity": prox_contrib
        }
    }

def calculate_herd_risk(cow_profiles_dict, session_events, session_anomalies):
    """Calculates overall herd risk score (0-100)."""
    if not cow_profiles_dict:
        return 0.0, "NORMAL"

    risk_scores = [p.get("risk_score", 0.0) for p in cow_profiles_dict.values()]
    avg_risk = sum(risk_scores) / len(risk_scores)
    max_risk = max(risk_scores)

    herd_risk = round(min(100.0, (avg_risk * 0.6) + (max_risk * 0.4)), 1)

    if herd_risk <= 29.0:
        status = "NORMAL"
    elif herd_risk <= 59.0:
        status = "LOW ATTENTION"
    elif herd_risk <= 79.0:
        status = "ELEVATED"
    else:
        status = "HIGH"

    return herd_risk, status

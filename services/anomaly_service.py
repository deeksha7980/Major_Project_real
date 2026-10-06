"""
anomaly_service.py — Interpretable Statistical Anomaly Detection Service.
Uses statistical z-scores and percentage deviations to flag behavioral/movement anomalies.
"""
import numpy as np

def detect_anomalies(cow_profiles_dict, session_windows):
    """
    Computes statistical anomalies across all active cows in the session.
    Features: Movement velocity, behavior transition rate, HFC call frequency.
    """
    anomalies = []
    if len(cow_profiles_dict) < 1:
        return anomalies

    movements = [p['movement_score'] for p in cow_profiles_dict.values()]
    transitions = [p['transitions_count'] for p in cow_profiles_dict.values()]

    mean_mov = np.mean(movements) if movements else 0.0
    std_mov  = np.std(movements) if len(movements) > 1 else 0.5
    if std_mov < 1e-3: std_mov = 0.5

    mean_trans = np.mean(transitions) if transitions else 0.0
    std_trans  = np.std(transitions) if len(transitions) > 1 else 1.0
    if std_trans < 1e-3: std_trans = 1.0

    for cow_id, p in cow_profiles_dict.items():
        # 1. Movement Anomaly
        z_mov = (p['movement_score'] - mean_mov) / std_mov
        if z_mov >= 1.8 and p['movement_score'] > 0.6:
            pct_dev = round(((p['movement_score'] - mean_mov) / max(0.1, mean_mov)) * 100, 1)
            anomalies.append({
                "timestamp": p['first_seen'],
                "cow_id": cow_id,
                "feature": "Movement Velocity",
                "baseline_val": round(mean_mov, 2),
                "current_val": round(p['movement_score'], 2),
                "deviation_pct": pct_dev,
                "severity": "high" if z_mov >= 2.5 else "medium",
                "explanation": f"Cow {cow_id} movement velocity ({p['movement_score']:.2f}) deviated by +{pct_dev}% from session mean baseline ({mean_mov:.2f})."
            })

        # 2. Rapid State Transition Anomaly
        z_trans = (p['transitions_count'] - mean_trans) / std_trans
        if z_trans >= 2.0 and p['transitions_count'] >= 4:
            pct_dev = round(((p['transitions_count'] - mean_trans) / max(1.0, mean_trans)) * 100, 1)
            anomalies.append({
                "timestamp": p['first_seen'] + 2.0,
                "cow_id": cow_id,
                "feature": "Behavior State Transitions",
                "baseline_val": round(mean_trans, 1),
                "current_val": p['transitions_count'],
                "deviation_pct": pct_dev,
                "severity": "medium",
                "explanation": f"Cow {cow_id} exhibited {p['transitions_count']} rapid behavior transitions (mean: {mean_trans:.1f})."
            })

    return anomalies

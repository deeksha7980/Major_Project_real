"""
multimodal_service.py — Multimodal Event Engine & Temporal Correlation Service.
Correlates visual behavior evidence and left/right acoustic evidence into unified Multimodal Events.
"""

def detect_multimodal_events(session_windows, cow_observations, alerts):
    """
    Correlates temporal coincidence of visual motion/posture changes and HFC acoustic vocalizations.
    """
    multimodal_events = []

    for win in session_windows:
        w_ts = win['start_time']
        w_end = win['end_time']
        left_hfc = win.get('left_audio_confidence', 0.0) if win.get('left_audio_class') == 'High-frequency call (HFC)' else 0.0
        right_hfc = win.get('right_audio_confidence', 0.0) if win.get('right_audio_class') == 'High-frequency call (HFC)' else 0.0

        max_hfc = max(left_hfc, right_hfc)
        if max_hfc < 0.70:
            continue

        # Find visual observations active in this window
        for cow_id, obs_list in cow_observations.items():
            win_obs = [o for o in obs_list if w_ts <= o['ts'] <= w_end]
            if not win_obs:
                continue

            max_vel = max(o.get('vel', 0.0) for o in win_obs)
            has_agg = any(o.get('beh') == 'aggressive' for o in win_obs)

            if max_vel >= 1.0 or has_agg or max_hfc >= 0.85:
                vis_desc = "High rapid movement" if max_vel >= 1.0 else ("Aggressive posture" if has_agg else "Active posture")
                aud_desc = f"HFC vocalization (Left: {left_hfc*100:.1f}%, Right: {right_hfc*100:.1f}%)"
                
                event_conf = round((max_hfc * 0.5) + (min(1.0, max_vel / 2.0) * 0.5), 2)
                severity = "high" if (max_hfc >= 0.85 and (max_vel >= 1.2 or has_agg)) else "medium"

                multimodal_events.append({
                    "timestamp": round(w_ts + 1.0, 1),
                    "cow_id": cow_id,
                    "event_type": "MULTIMODAL_VOCALIZATION_MOTION",
                    "visual_evidence": f"Cow {cow_id} displaying {vis_desc.lower()} (velocity: {max_vel:.2f})",
                    "audio_evidence": aud_desc,
                    "left_confidence": round(left_hfc, 2),
                    "right_confidence": round(right_hfc, 2),
                    "audio_agreement": win.get('audio_agreement', 1.0),
                    "event_confidence": event_conf,
                    "severity": severity,
                    "description": f"Temporally correlated multimodal event: Cow {cow_id} {vis_desc.lower()} synchronized with {aud_desc}."
                })

    return multimodal_events

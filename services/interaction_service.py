"""
interaction_service.py — Spatial Cow Interaction Graph & Herd Analytics Service.
Tracks bounding-box spatial proximity between cows and aggregates herd-level metrics.
"""
from collections import Counter

def overlap_ratio(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / amin if amin > 0 else 0.0

def build_cow_interaction_graph(all_active_frames):
    """
    Computes spatial proximity interactions between pairs of cows across video frames.
    Input: list of frame records [{'ts': 0.04, 'boxes': [{'id': '1', 'box': (...)}, ...]}, ...]
    """
    interactions = []
    pair_obs = {}

    for f_rec in all_active_frames:
        ts = f_rec['ts']
        boxes = f_rec.get('boxes', [])

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                c1, c2 = boxes[i], boxes[j]
                id1, id2 = sorted([str(c1['id']), str(c2['id'])])

                ov = overlap_ratio(c1['box'], c2['box'])
                if ov > 0.05:
                    key = (id1, id2)
                    pair_obs.setdefault(key, []).append({'ts': ts, 'ov': ov})

    for (cow_a, cow_b), obs_list in pair_obs.items():
        if len(obs_list) >= 2:
            start_t = obs_list[0]['ts']
            end_t   = obs_list[-1]['ts']
            dur     = max(0.5, round(end_t - start_t, 1))
            avg_ov  = sum(o['ov'] for o in obs_list) / len(obs_list)

            itype = "High-Movement Interaction" if avg_ov > 0.30 else "Proximity Event"
            interactions.append({
                "cow_a": cow_a,
                "cow_b": cow_b,
                "start_time": round(start_t, 1),
                "end_time": round(end_t, 1),
                "duration": dur,
                "proximity_score": round(avg_ov, 2),
                "interaction_type": itype
            })

    return interactions

def calculate_herd_analytics(cow_profiles_dict, session_windows, multimodal_events, anomalies):
    """Aggregates overall herd-level statistics."""
    total_cows = len(cow_profiles_dict)
    if total_cows == 0:
        return {
            "total_cows": 0, "standing_cows": 0, "lying_cows": 0, "eating_cows": 0,
            "walking_cows": 0, "acoustic_events_count": 0, "anomalies_count": 0,
            "multimodal_events_count": 0, "behavior_distribution": {}
        }

    beh_counter = Counter(p['dominant_behavior'] for p in cow_profiles_dict.values())
    standing_cows = beh_counter.get('standing', 0)
    lying_cows    = beh_counter.get('lying', 0)
    eating_cows   = beh_counter.get('eating', 0) + beh_counter.get('drinking', 0)
    
    # Audio events count in windows
    hfc_events = sum(1 for w in session_windows if w.get('left_audio_class') == 'High-frequency call (HFC)' or w.get('right_audio_class') == 'High-frequency call (HFC)')

    return {
        "total_cows": total_cows,
        "standing_cows": standing_cows,
        "lying_cows": lying_cows,
        "eating_cows": eating_cows,
        "acoustic_events_count": hfc_events,
        "anomalies_count": len(anomalies),
        "multimodal_events_count": len(multimodal_events),
        "behavior_distribution": dict(beh_counter)
    }

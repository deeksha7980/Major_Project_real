"""
cow_service.py — Individual Cow Intelligence & Behavior Timeline Service.
Aggregates per-cow tracking profiles, behavior distributions, timelines, and state transition matrices.
"""
from collections import Counter, defaultdict

BEHAVIOURS = ["standing", "lying", "eating", "drinking", "aggressive", "rumination"]

def aggregate_cow_profiles(cow_frame_observations, total_session_fps=25.0):
    """
    Input: dict mapping cow_id -> list of frame dicts:
    [{'ts': 0.04, 'beh': 'standing', 'conf': 0.85, 'vel': 0.12, 'side': 'left'}, ...]
    Outputs individual cow profiles, behavior timelines, and transition matrices.
    """
    cow_profiles = {}
    behavior_timelines = {}
    transition_matrices = {}

    for cow_id, obs_list in cow_frame_observations.items():
        if not obs_list:
            continue

        sorted_obs = sorted(obs_list, key=lambda x: x['ts'])
        first_seen = sorted_obs[0]['ts']
        last_seen = sorted_obs[-1]['ts']
        visible_duration = max(0.5, round(last_seen - first_seen, 1))

        counts = Counter(o['beh'] for o in sorted_obs)
        total_obs = len(sorted_obs)
        
        standing_pct = round((counts['standing'] / total_obs) * 100, 1)
        lying_pct    = round((counts['lying'] / total_obs) * 100, 1)
        eating_pct   = round((counts['eating'] / total_obs) * 100, 1)
        drinking_pct = round((counts['drinking'] / total_obs) * 100, 1)
        aggressive_pct = round((counts['aggressive'] / total_obs) * 100, 1)
        rumination_pct = round((counts['rumination'] / total_obs) * 100, 1)

        dominant_beh = counts.most_common(1)[0][0] if counts else "standing"
        avg_movement = round(sum(o.get('vel', 0.0) for o in sorted_obs) / total_obs, 2)

        # Build behavior timeline segments
        timeline = []
        cur_beh = None
        cur_start = first_seen
        cur_confs = []
        cur_vels = []

        transitions = 0
        trans_counts = defaultdict(int)

        for i, o in enumerate(sorted_obs):
            beh = o['beh']
            if cur_beh is None:
                cur_beh = beh
                cur_start = o['ts']

            if beh != cur_beh or i == len(sorted_obs) - 1:
                end_t = o['ts']
                if end_t > cur_start:
                    avg_c = sum(cur_confs) / max(1, len(cur_confs))
                    avg_v = sum(cur_vels) / max(1, len(cur_vels))
                    timeline.append({
                        "cow_id": cow_id,
                        "start_time": round(cur_start, 1),
                        "end_time": round(end_t, 1),
                        "behavior": cur_beh,
                        "confidence": round(avg_c, 2),
                        "movement": round(avg_v, 2)
                    })
                if beh != cur_beh:
                    transitions += 1
                    trans_counts[(cur_beh, beh)] += 1
                    cur_beh = beh
                    cur_start = o['ts']
                    cur_confs = []
                    cur_vels = []

            cur_confs.append(o.get('conf', 0.5))
            cur_vels.append(o.get('vel', 0.0))

        cow_profiles[cow_id] = {
            "cow_id": cow_id,
            "first_seen": round(first_seen, 1),
            "last_seen": round(last_seen, 1),
            "visible_duration": visible_duration,
            "dominant_behavior": dominant_beh,
            "standing_pct": standing_pct,
            "lying_pct": lying_pct,
            "eating_pct": eating_pct,
            "drinking_pct": drinking_pct,
            "aggressive_pct": aggressive_pct,
            "rumination_pct": rumination_pct,
            "movement_score": avg_movement,
            "transitions_count": transitions,
        }
        behavior_timelines[cow_id] = timeline
        transition_matrices[cow_id] = dict(trans_counts)

    return {
        "cow_profiles": cow_profiles,
        "behavior_timelines": behavior_timelines,
        "transition_matrices": transition_matrices
    }

"""
audio_service.py — Left/Right Acoustic Zone & Confidence Intelligence Service.
Calculates spatial acoustic zones, left/right agreement, and confidence handling.
"""

def analyze_acoustic_zone(left_hfc_prob, right_hfc_prob, left_conf, right_conf, hfc_threshold=0.75, diff_threshold=0.25):
    """
    Determines acoustic spatial zone based on Left and Right microphone predictions.
    """
    avg_hfc = (left_hfc_prob + right_hfc_prob) / 2.0
    hfc_diff = abs(left_hfc_prob - right_hfc_prob)
    audio_agreement = round(max(0.0, 1.0 - hfc_diff), 4)

    left_is_high = left_hfc_prob >= hfc_threshold
    right_is_high = right_hfc_prob >= hfc_threshold

    if left_conf < 0.60 and right_conf < 0.60:
        acoustic_zone = "UNCERTAIN"
    elif left_is_high and right_is_high:
        if hfc_diff < 0.15:
            acoustic_zone = "BROAD/CENTER"
        elif left_hfc_prob > right_hfc_prob:
            acoustic_zone = "LEFT-DOMINANT"
        else:
            acoustic_zone = "RIGHT-DOMINANT"
    elif left_is_high and not right_is_high:
        acoustic_zone = "LEFT-DOMINANT"
    elif right_is_high and not left_is_high:
        acoustic_zone = "RIGHT-DOMINANT"
    else:
        acoustic_zone = "QUIET/BACKGROUND"

    return {
        "avg_hfc": round(avg_hfc, 4),
        "hfc_diff": round(hfc_diff, 4),
        "audio_agreement": audio_agreement,
        "acoustic_zone": acoustic_zone,
        "left_hfc_pct": round(left_hfc_prob * 100, 1),
        "right_hfc_pct": round(right_hfc_prob * 100, 1)
    }

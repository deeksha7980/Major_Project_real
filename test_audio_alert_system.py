"""
test_audio_alert_system.py
---------------------------
Unit test suite for CowCareAI Audio & Multimodal Alert Logic.
"""
import os, sys

def evaluate_audio_alert(
    left_hfc_prob: float,
    right_hfc_prob: float,
    left_class: str = "High-frequency call (HFC)",
    right_class: str = "High-frequency call (HFC)",
    video_anomaly: bool = False,
    hfc_threshold: float = 0.75,
    strict_dual_threshold: float = 0.85
) -> dict:
    """
    Evaluates audio alert conditions based on PyTorch Audio CNN model predictions.
    Does NOT require video anomaly to trigger an audio alert.
    """
    both_high = (left_hfc_prob >= hfc_threshold) and (right_hfc_prob >= hfc_threshold)
    left_high = (left_hfc_prob >= hfc_threshold)
    right_high = (right_hfc_prob >= hfc_threshold)
    
    avg_hfc_prob = (left_hfc_prob + right_hfc_prob) / 2.0
    
    if both_high and (avg_hfc_prob >= strict_dual_threshold):
        audio_state = "HIGH_CONFIDENCE_HFC"
        audio_alert = True
        severity = "high"
        reason = f"Both microphones detected high-confidence HFC (Left: {left_hfc_prob*100:.1f}%, Right: {right_hfc_prob*100:.1f}%)"
    elif left_high:
        audio_state = "LEFT_HFC"
        audio_alert = True
        severity = "medium"
        reason = f"Left microphone detected HFC (Confidence: {left_hfc_prob*100:.1f}%)"
    elif right_high:
        audio_state = "RIGHT_HFC"
        audio_alert = True
        severity = "medium"
        reason = f"Right microphone detected HFC (Confidence: {right_hfc_prob*100:.1f}%)"
    else:
        audio_state = "NO_HFC_EVENT"
        audio_alert = False
        severity = "info"
        reason = f"No significant HFC vocalization detected (Left HFC: {left_hfc_prob*100:.1f}%, Right HFC: {right_hfc_prob*100:.1f}%)"
        
    multimodal_alert = audio_alert and video_anomaly
    
    return {
        "audio_alert": audio_alert,
        "multimodal_alert": multimodal_alert,
        "audio_state": audio_state,
        "severity": severity,
        "reason": reason,
        "left_hfc_prob": left_hfc_prob,
        "right_hfc_prob": right_hfc_prob,
        "avg_hfc_prob": avg_hfc_prob,
        "video_anomaly": video_anomaly
    }

def run_tests():
    print("=== RUNNING AUDIO ALERT UNIT TESTS ===")
    
    # TEST 1: Dual Channel High-Confidence HFC
    t1 = evaluate_audio_alert(0.95, 0.93, video_anomaly=False)
    print(f"\n[TEST 1] Left HFC: 0.95, Right HFC: 0.93, Video Normal")
    print(f"Result: Alert={t1['audio_alert']}, State={t1['audio_state']}, Reason: {t1['reason']}")
    assert t1['audio_alert'] == True and t1['audio_state'] == "HIGH_CONFIDENCE_HFC"
    
    # TEST 2: Single Channel Left HFC
    t2 = evaluate_audio_alert(0.91, 0.10, video_anomaly=False)
    print(f"\n[TEST 2] Left HFC: 0.91, Right HFC: 0.10, Video Normal")
    print(f"Result: Alert={t2['audio_alert']}, State={t2['audio_state']}, Reason: {t2['reason']}")
    assert t2['audio_alert'] == True and t2['audio_state'] == "LEFT_HFC"
    
    # TEST 3: Low Confidence Audio (Below Threshold)
    t3 = evaluate_audio_alert(0.40, 0.35, video_anomaly=False)
    print(f"\n[TEST 3] Left HFC: 0.40, Right HFC: 0.35, Video Normal")
    print(f"Result: Alert={t3['audio_alert']}, State={t3['audio_state']}, Reason: {t3['reason']}")
    assert t3['audio_alert'] == False and t3['audio_state'] == "NO_HFC_EVENT"
    
    # TEST 4: Dual High HFC with Normal Video (Critical Multimodal Test)
    t4 = evaluate_audio_alert(0.99, 0.98, video_anomaly=False)
    print(f"\n[TEST 4 - CRITICAL] Left HFC: 0.99, Right HFC: 0.98, Video Normal")
    print(f"Result: AUDIO ALERT={t4['audio_alert']}, State={t4['audio_state']}, Reason: {t4['reason']}")
    assert t4['audio_alert'] == True, "CRITICAL ERROR: High HFC audio failed to trigger audio alert when video was normal!"
    
    print("\n[SUCCESS] ALL 4 ALERT UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    run_tests()


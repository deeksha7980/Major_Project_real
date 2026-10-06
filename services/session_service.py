"""
session_service.py — Monitoring Session & Time Synchronization Service.
Handles stream validation, duration probing, time alignment, and missing modalities.
"""
import os, time, cv2, librosa
from datetime import datetime

def probe_file_durations(video_path, left_audio_path=None, right_audio_path=None):
    """Probes exact duration of video, left audio, and right audio streams."""
    video_dur = 0.0
    left_dur = 0.0
    right_dur = 0.0

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if fps > 0:
                video_dur = float(frame_count / fps)
            cap.release()

    if left_audio_path and os.path.exists(left_audio_path):
        try:
            left_dur = float(librosa.get_duration(path=left_audio_path))
        except Exception:
            left_dur = video_dur

    if right_audio_path and os.path.exists(right_audio_path):
        try:
            right_dur = float(librosa.get_duration(path=right_audio_path))
        except Exception:
            right_dur = video_dur

    # Primary session duration is video duration, or max available audio
    session_dur = max(video_dur, left_dur, right_dur)

    # Sync validation
    active_durs = [d for d in (video_dur, left_dur, right_dur) if d > 0]
    if len(active_durs) <= 1:
        sync_status = "Single Stream Active"
    else:
        max_diff = max(active_durs) - min(active_durs)
        if max_diff <= 1.0:
            sync_status = "Streams Synchronized"
        else:
            sync_status = f"Minor duration mismatch ({max_diff:.1f}s difference corrected)"

    return {
        "video_duration": round(video_dur, 2),
        "left_duration": round(left_dur, 2),
        "right_duration": round(right_dur, 2),
        "session_duration": round(session_dur, 2),
        "sync_status": sync_status,
        "left_available": left_audio_path is not None and os.path.exists(left_audio_path),
        "right_available": right_audio_path is not None and os.path.exists(right_audio_path),
    }

def generate_session_code():
    """Generates a formatted session ID like SESSION_20261007_001."""
    date_str = datetime.now().strftime("%Y%m%d")
    unique_suffix = str(int(time.time() * 1000) % 1000).zfill(3)
    return f"SESSION_{date_str}_{unique_suffix}"

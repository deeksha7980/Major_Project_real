"""
integrate_audio_model.py
-------------------------
Integrates the trained PyTorch Audio CNN model (cattle_audio_cnn.pth)
with the existing CowCareAI pipeline.
"""
import os, torch, numpy as np, librosa
from audio_inference import CattleAudioClassifier

def run_multimodal_analysis(video_path, audio_path=None, model_path='cattle_audio_cnn.pth'):
    print("=== CowCareAI Multimodal Analysis Pipeline ===")
    print(f"Video file: {video_path}")
    
    # 1. Video Analysis Branch (Simulated / YOLO Detection)
    video_result = {
        'cow_id': 'Cow 1',
        'visual_posture': 'Standing',
        'visual_confidence': 0.94
    }
    
    # 2. Audio Analysis Branch (PyTorch Audio CNN Model)
    audio_classifier = CattleAudioClassifier(model_path=model_path)
    
    # Determine audio source
    target_audio = audio_path if (audio_path and os.path.exists(audio_path)) else video_path
    audio_result = audio_classifier.predict(target_audio)
    
    print("\n--- Pipeline Outputs ---")
    print(f"[VIDEO BRANCH] Detected {video_result['cow_id']} | Posture: {video_result['visual_posture']} (Conf: {video_result['visual_confidence']*100:.1f}%)")
    print(f"[AUDIO BRANCH] Vocalization Call: {audio_result['class_name']} (Conf: {audio_result['confidence']*100:.1f}%)")
    
    # 3. Multimodal Synthesis
    multimodal_summary = (
        f"{video_result['cow_id']} detected {video_result['visual_posture'].lower()} "
        f"and producing a {audio_result['class_name'].lower()} "
        f"(Audio Confidence: {audio_result['confidence']*100:.1f}%)."
    )
    print(f"\n[FINAL MULTIMODAL REPORT]: {multimodal_summary}")
    return {
        'video': video_result,
        'audio': audio_result,
        'summary': multimodal_summary
    }

if __name__ == '__main__':
    test_video = '664.mp4'
    if os.path.exists(test_video):
        run_multimodal_analysis(test_video)
    else:
        print(f"Test video {test_video} not found.")

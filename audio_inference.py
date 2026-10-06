import os, sys, torch, torch.nn as nn, numpy as np, librosa, scipy.io.wavfile as wav, scipy.signal as signal

# ----------------------------------------------------
# Audio CNN PyTorch Model Definition
# ----------------------------------------------------
class AudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

CLASS_NAMES = ['High-frequency call (HFC)', 'Low-frequency call (LFC)']
SAMPLE_RATE = 22050
TARGET_DURATION = 2.0
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
FIXED_TIME_STEPS = 87

def extract_log_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) > int(0.2 * SAMPLE_RATE):
        y = y_trimmed
    if len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)), mode='constant')
    else:
        y = y[:TARGET_LENGTH]
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    if log_mel.shape[1] > FIXED_TIME_STEPS:
        log_mel = log_mel[:, :FIXED_TIME_STEPS]
    elif log_mel.shape[1] < FIXED_TIME_STEPS:
        log_mel = np.pad(log_mel, ((0, 0), (0, FIXED_TIME_STEPS - log_mel.shape[1])), mode='constant')
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel.astype(np.float32)

def extract_log_mel_signal(y, sr=SAMPLE_RATE):
    if sr != SAMPLE_RATE and len(y) > 0:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    # Trim silence
    if len(y) > 0:
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) > int(0.2 * SAMPLE_RATE):
            y = y_trimmed
    if len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)), mode='constant')
    else:
        y = y[:TARGET_LENGTH]
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    if log_mel.shape[1] > FIXED_TIME_STEPS:
        log_mel = log_mel[:, :FIXED_TIME_STEPS]
    elif log_mel.shape[1] < FIXED_TIME_STEPS:
        log_mel = np.pad(log_mel, ((0, 0), (0, FIXED_TIME_STEPS - log_mel.shape[1])), mode='constant')
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel.astype(np.float32)

class CattleAudioClassifier:
    def __init__(self, model_path='cattle_audio_cnn.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioCNN(num_classes=2).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[CattleAudioClassifier] Loaded model weights from {model_path}")
        else:
            print(f"[CattleAudioClassifier] Model file {model_path} not found!")

    def predict(self, audio_path):
        mel = extract_log_mel(audio_path)
        tensor_mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor_mel)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
        return dict(
            class_idx=pred_idx,
            class_name=CLASS_NAMES[pred_idx],
            confidence=conf,
            hfc_prob=float(probs[0]),
            lfc_prob=float(probs[1])
        )

    def predict_signal(self, y, sr=SAMPLE_RATE):
        mel = extract_log_mel_signal(y, sr)
        tensor_mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor_mel)
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])
        return dict(
            class_idx=pred_idx,
            class_name=CLASS_NAMES[pred_idx],
            confidence=conf,
            hfc_prob=float(probs[0]),
            lfc_prob=float(probs[1])
        )


if __name__ == '__main__':
    classifier = CattleAudioClassifier()
    test_files = ['high.mp3', 'low.mp3']
    for tf in test_files:
        if os.path.exists(tf):
            res = classifier.predict(tf)
            print(f"File: {tf} -> Prediction: {res['class_name']} (Confidence: {res['confidence']*100:.2f}%)")

import os, glob, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wav
import scipy.signal as signal
import librosa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------
# 1. Dataset Collection & Group Extraction (Individual Cow IDs)
# ----------------------------------------------------
db_path = r'Cattle sounds database (raw data)'
records = []

for root, dirs, files in os.walk(db_path):
    for f in files:
        if f.lower().endswith('.wav'):
            fp = os.path.join(root, f)
            if 'hfc' in root.lower() or 'hfc' in f.lower():
                label = 0 # High Frequency Call
            elif 'lfc' in root.lower() or 'lfc' in f.lower():
                label = 1 # Low Frequency Call
            else:
                continue

            parts = root.split(os.sep)
            cow_id = 'unknown'
            for p in parts:
                if 'cow id' in p.lower() or 'cow_id' in p.lower():
                    cow_id = p.strip()
                    break
            records.append((fp, label, cow_id))

print(f"Total collected audio samples: {len(records)}")
hfc_cnt = sum(1 for r in records if r[1] == 0)
lfc_cnt = sum(1 for r in records if r[1] == 1)
print(f"HFC (Label 0): {hfc_cnt} | LFC (Label 1): {lfc_cnt}")

# ----------------------------------------------------
# 2. Group-Aware Train / Val / Test Split
# ----------------------------------------------------
unique_cows = sorted(list(set(r[2] for r in records)))
print(f"Unique Individual Cow IDs ({len(unique_cows)}): {unique_cows}")

np.random.seed(42)
np.random.shuffle(unique_cows)

n_cows = len(unique_cows)
train_cows = set(unique_cows[:int(n_cows * 0.70)])
val_cows   = set(unique_cows[int(n_cows * 0.70):int(n_cows * 0.85)])
test_cows  = set(unique_cows[int(n_cows * 0.85):])

train_records = [r for r in records if r[2] in train_cows]
val_records   = [r for r in records if r[2] in val_cows]
test_records  = [r for r in records if r[2] in test_cows]

print(f"Train split: {len(train_records)} samples ({len(train_cows)} cows)")
print(f"Val split:   {len(val_records)} samples ({len(val_cows)} cows)")
print(f"Test split:  {len(test_records)} samples ({len(test_cows)} cows)")

# ----------------------------------------------------
# 3. Fixed-Shape Audio Feature Extraction (Fixed 22050 Hz -> 64x87 Log-Mel)
# ----------------------------------------------------
SAMPLE_RATE = 22050
TARGET_DURATION = 2.0 # 2 seconds
TARGET_LENGTH = int(SAMPLE_RATE * TARGET_DURATION) # 44100 samples
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
FIXED_TIME_STEPS = 87

def extract_log_mel_fixed(audio_path):
    try:
        sr_orig, data = wav.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # Simple energy threshold trimming
        abs_data = np.abs(data)
        thr = 0.02 * np.max(abs_data)
        nz = np.where(abs_data > thr)[0]
        if len(nz) > int(0.2 * sr_orig):
            data = data[nz[0]:nz[-1]]

        # Resample to 22050 Hz using SciPy resample
        if sr_orig != SAMPLE_RATE:
            num_samples = int(len(data) * SAMPLE_RATE / sr_orig)
            data = signal.resample(data, num_samples)

        # Pad or truncate to TARGET_LENGTH (44100 samples)
        if len(data) < TARGET_LENGTH:
            data = np.pad(data, (0, TARGET_LENGTH - len(data)), mode='constant')
        else:
            data = data[:TARGET_LENGTH]

        # Peak normalization
        mx = np.max(np.abs(data))
        if mx > 1e-6:
            data = data / mx

        # Compute Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=data, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure fixed shape (64, 87)
        if log_mel.shape[1] > FIXED_TIME_STEPS:
            log_mel = log_mel[:, :FIXED_TIME_STEPS]
        elif log_mel.shape[1] < FIXED_TIME_STEPS:
            log_mel = np.pad(log_mel, ((0, 0), (0, FIXED_TIME_STEPS - log_mel.shape[1])), mode='constant')

        # Per-sample standardization
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        return log_mel.astype(np.float32)
    except Exception as e:
        return np.zeros((N_MELS, FIXED_TIME_STEPS), dtype=np.float32)

print("Pre-computing Log-Mel Spectrograms...")
def preprocess_list(record_list):
    feats, labels = [], []
    for fp, lbl, _ in record_list:
        feats.append(extract_log_mel_fixed(fp))
        labels.append(lbl)
    return np.array(feats), np.array(labels)

X_train, y_train = preprocess_list(train_records)
X_val, y_val     = preprocess_list(val_records)
X_test, y_test   = preprocess_list(test_records)

print(f"Feature shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

class CachedDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1) # (N, 1, 64, 87)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        mel = self.X[idx].clone()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                f_bar = torch.randint(1, N_MELS // 8, (1,)).item()
                f0 = torch.randint(0, N_MELS - f_bar, (1,)).item()
                mel[0, f0:f0+f_bar, :] = 0.0
            if torch.rand(1).item() > 0.5:
                t_len = mel.shape[2]
                t_bar = torch.randint(1, max(2, t_len // 8), (1,)).item()
                t0 = torch.randint(0, max(1, t_len - t_bar), (1,)).item()
                mel[0, :, t0:t0+t_bar] = 0.0
        return mel, self.y[idx]

train_loader = DataLoader(CachedDataset(X_train, y_train, augment=True), batch_size=32, shuffle=True)
val_loader   = DataLoader(CachedDataset(X_val, y_val, augment=False), batch_size=32, shuffle=False)
test_loader  = DataLoader(CachedDataset(X_test, y_test, augment=False), batch_size=32, shuffle=False)

# ----------------------------------------------------
# 4. Audio CNN PyTorch Model Architecture
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")
model = AudioCNN(num_classes=2).to(device)

# Handle Class Imbalance via Class Weights
c0 = float(np.sum(y_train == 0))
c1 = float(np.sum(y_train == 1))
weights = torch.tensor([1.0, c0 / max(1.0, c1)], dtype=torch.float32).to(device)
print(f"Class weights: {weights}")

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# ----------------------------------------------------
# 5. Fast Training Loop
# ----------------------------------------------------
EPOCHS = 20
best_val_f1 = -1.0
model_save_path = 'cattle_audio_cnn.pth'

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for mel, label in train_loader:
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(mel)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == label).sum().item()
        total += mel.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds_list, val_targets_list = [], []
    with torch.no_grad():
        for mel, label in val_loader:
            mel, label = mel.to(device), label.to(device)
            out = model(mel)
            loss = criterion(out, label)
            val_loss += loss.item() * mel.size(0)
            _, preds = torch.max(out, 1)
            val_preds_list.extend(preds.cpu().numpy())
            val_targets_list.extend(label.cpu().numpy())

    val_loss = val_loss / len(y_val)
    val_acc = accuracy_score(val_targets_list, val_preds_list)
    val_f1 = f1_score(val_targets_list, val_preds_list, average='macro', zero_division=0)

    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro F1: {val_f1:.4f}")

    if val_f1 >= best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), model_save_path)
        print(f"  --> Saved best model checkpoint to {model_save_path}")

# ----------------------------------------------------
# 6. Test Set Evaluation
# ----------------------------------------------------
print("\n=== Evaluating Best Model on Test Set ===")
model.load_state_dict(torch.load(model_save_path))
model.eval()

test_preds, test_probs, test_targets = [], [], []
with torch.no_grad():
    for mel, label in test_loader:
        mel = mel.to(device)
        out = model(mel)
        probs = torch.softmax(out, dim=1)[:, 1]
        _, preds = torch.max(out, 1)
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_targets.extend(label.numpy())

test_acc = accuracy_score(test_targets, test_preds)
test_prec = precision_score(test_targets, test_preds, average='macro', zero_division=0)
test_rec = recall_score(test_targets, test_preds, average='macro', zero_division=0)
test_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
cm = confusion_matrix(test_targets, test_preds)

print(f"Test Accuracy:  {test_acc * 100:.2f}%")
print(f"Test Precision: {test_prec * 100:.2f}%")
print(f"Test Recall:    {test_rec * 100:.2f}%")
print(f"Test Macro F1:  {test_f1 * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(test_targets, test_preds, target_names=['High Frequency Call (HFC)', 'Low Frequency Call (LFC)'], zero_division=0))

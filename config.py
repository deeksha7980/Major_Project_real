import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ---------- Database ----------
DB_PATH     = os.path.join(BASE_DIR, "instance", "livestock.db")
PIPELINE_DB = os.path.join(BASE_DIR, "instance", "cow_monitoring.db")

# ---------- Uploads (NOT inside /static) ----------
MEDIA_DIR        = os.path.join(BASE_DIR, "media")
UPLOAD_VIDEO_DIR = os.path.join(MEDIA_DIR, "videos")
UPLOAD_AUDIO_DIR = os.path.join(MEDIA_DIR, "audio")
OUTPUT_DIR       = os.path.join(MEDIA_DIR, "outputs")
FRAME_DIR        = os.path.join(MEDIA_DIR, "frames")
REPORT_DIR       = os.path.join(BASE_DIR, "reports")
LOG_DIR          = os.path.join(BASE_DIR, "logs")

# Create every folder we need at import time
for d in (UPLOAD_VIDEO_DIR, UPLOAD_AUDIO_DIR, OUTPUT_DIR, FRAME_DIR,
          REPORT_DIR, LOG_DIR,
          os.path.join(BASE_DIR, "instance")):
    os.makedirs(d, exist_ok=True)
    print(d)

ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_AUDIO = {"wav", "mp3", "m4a", "flac", "ogg", "aac"}

# ---------- Telegram ----------
TELEGRAM_BOT_TOKEN = "8614269440:AAHsITznIg3KIahap32wHft2qMEH5MU3Lfc"
TELEGRAM_CHAT_ID   = "1039027361"

# ---------- Pipeline & Temporal Config ----------
YOLO_MODEL = os.path.join(BASE_DIR, "cattle.pt")
WINDOW_SIZE_SECONDS = 5.0
SYNC_TOLERANCE_SECONDS = 1.0

# ---------- Audio CNN & Acoustic Zone Config ----------
DEBUG_MULTIMODAL_ALERTS    = True
AUDIO_HFC_THRESHOLD        = 0.75
STRICT_DUAL_HFC_THRESHOLD  = 0.85
ACOUSTIC_DIFF_THRESHOLD    = 0.25
AUDIO_MODEL_PATH           = os.path.join(BASE_DIR, "cattle_audio_cnn.pth")

# ---------- Multimodal Risk Engine Thresholds ----------
RISK_NORMAL_MAX   = 29.0
RISK_ATTENTION_MAX = 59.0
RISK_ELEVATED_MAX  = 79.0
# RISK > 79 is HIGH RISK

# Risk Contributors Weightings
RISK_WEIGHT_MOVEMENT    = 25.0
RISK_WEIGHT_BEHAVIOR    = 20.0
RISK_WEIGHT_AUDIO_EVENT = 25.0
RISK_WEIGHT_MULTIMODAL  = 20.0
RISK_WEIGHT_PROXIMITY   = 10.0

# ---------- Anomaly Engine Config ----------
ANOMALY_ZSCORE_THRESHOLD = 2.0
ANOMALY_PCT_DEVIATION    = 0.50

# ---------- Spatial Interaction Config ----------
INTERACTION_PROXIMITY_THRESHOLD = 0.20 # IoU or overlap ratio

# ---------- Auth ----------
SECRET_KEY = "change-this-secret-key-in-production"

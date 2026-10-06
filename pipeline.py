"""
pipeline.py — Multimodal cow-behaviour pipeline.
Called by pipeline_runner.py.
"""
import os, csv, time, traceback
from collections import deque, Counter, defaultdict

import cv2
import numpy as np
import librosa
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import (YOLO_MODEL, FRAME_DIR, DEBUG_MULTIMODAL_ALERTS,
                    AUDIO_HFC_THRESHOLD, STRICT_DUAL_HFC_THRESHOLD, ACOUSTIC_DIFF_THRESHOLD,
                    AUDIO_MODEL_PATH, WINDOW_SIZE_SECONDS)
from audio_inference import CattleAudioClassifier
from test_audio_alert_system import evaluate_audio_alert

from services.session_service import probe_file_durations, generate_session_code
from services.audio_service import analyze_acoustic_zone
from services.cow_service import aggregate_cow_profiles
from services.multimodal_service import detect_multimodal_events
from services.risk_service import calculate_cow_risk, calculate_herd_risk
from services.anomaly_service import detect_anomalies
from services.baseline_service import compare_cow_baseline, update_cow_baseline
from services.interaction_service import build_cow_interaction_graph, calculate_herd_analytics
from services.report_service import generate_csv_report, generate_html_report

# ================================================================
# CONFIG
# ================================================================
CONF, IOU = 0.30, 0.45
FRAME_W, FRAME_H = 640, 480
PANEL_W = 440
W_VIDEO, W_AUDIO = 0.65, 0.35

THRESHOLD_DB = -15.6
AUDIO_WIN    = 1.0
ABN_MIN_WIN  = 2
EPS          = 1e-10

VIDEO_SMOOTH = 15
FUSED_EMA    = 0.30
AGG_SPEED_LO, AGG_SPEED_HI = 0.5, 2.0
AGG_ALERT_SCORE  = 0.40
AGG_ALERT_FRAMES = 5
ALERT_COOLDOWN_S = 5.0
LYING_LIMIT_S    = 600
SAVE_EVERY       = 5

BEHAVIOURS = ["standing", "lying", "eating", "drinking", "aggressive", "rumination"]
ALIAS = {"standing": 0, "lying_down": 1, "lying": 1, "feeding": 2, "eating": 2,
         "drinking": 3, "aggressive": 4, "rumination": 5, "ruminating": 5}
COLORS = {"standing": (0, 200, 0), "lying": (200, 150, 0), "eating": (0, 180, 255),
          "drinking": (255, 120, 0), "aggressive": (0, 0, 255), "rumination": (200, 0, 200)}

PA_QUIET    = np.array([.20, .20, .20, .10, .02, .28])
PA_LOUD     = np.array([.35, .10, .10, .05, .30, .10])
PA_ABNORMAL = np.array([.25, .03, .03, .02, .60, .07])


def log(msg):
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================
# Audio helpers
# ================================================================
def window_levels(y, sr):
    n = int(AUDIO_WIN * sr)
    out = []
    for s in range(0, len(y), n):
        seg = y[s:s + n]
        if len(seg) < 2048:
            out.append(-100.0); continue
        rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
        out.append(float(np.percentile(20 * np.log10(np.maximum(rms, EPS)), 95)))
    return np.array(out)

import subprocess, shutil

def reencode_for_browser(raw_mp4_path):
    """
    Convert the raw OpenCV mp4v video into browser-compatible H.264.
    Replaces the file in-place. Falls back silently if ffmpeg is missing.
    """
    if not os.path.exists(raw_mp4_path) or os.path.getsize(raw_mp4_path) < 1024:
        log(f"skip re-encode: file missing or too small ({raw_mp4_path})")
        return raw_mp4_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log("ffmpeg not found on PATH — output may not play in browsers")
        return raw_mp4_path

    fixed = raw_mp4_path.replace(".mp4", "_h264.mp4")
    cmd = [
        ffmpeg, "-y",
        "-i", raw_mp4_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",                       # drop audio track (dashboard has none)
        fixed,
    ]
    log(f"Re-encoding for browser: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=600)
        if r.returncode != 0:
            log(f"ffmpeg failed rc={r.returncode}:\n{r.stdout[-800:]}")
            return raw_mp4_path
        if os.path.getsize(fixed) < 1024:
            log("re-encoded file too small — keeping original")
            return raw_mp4_path
        # replace original with the re-encoded version
        try:
            os.remove(raw_mp4_path)
        except Exception:
            pass
        os.rename(fixed, raw_mp4_path)
        log(f"re-encoded OK → {raw_mp4_path} ({os.path.getsize(raw_mp4_path)} bytes)")
        return raw_mp4_path
    except Exception as e:
        log(f"re-encode exception: {e}")
        return raw_mp4_path
    
def analyse_audio(video_path, left_path, right_path):
    try:
        classifier = CattleAudioClassifier(model_path=AUDIO_MODEL_PATH)
        if left_path and right_path:
            log(f"Loading LEFT audio: {left_path}")
            yl, srl = librosa.load(left_path, sr=22050, mono=True)
            log(f"Loading RIGHT audio: {right_path}")
            yr, srr = librosa.load(right_path, sr=22050, mono=True)
        else:
            log(f"Loading audio from video: {video_path}")
            y, srl = librosa.load(video_path, sr=22050, mono=False)
            srr = srl
            yl, yr = (y, y) if y.ndim == 1 else (y[0], y[1])
        
        ldb, rdb = window_levels(yl, srl), window_levels(yr, srr)
        win_size = int(AUDIO_WIN * 22050)
        n_windows = max(len(ldb), len(rdb), int(np.ceil(max(len(yl), len(yr)) / win_size)))
        if n_windows == 0:
            n_windows = 1
            
        left_evals, right_evals, eval_results = [], [], []
        for w in range(n_windows):
            s_idx = w * win_size
            e_idx = s_idx + win_size
            seg_l = yl[s_idx:e_idx] if s_idx < len(yl) else np.array([], dtype=np.float32)
            seg_r = yr[s_idx:e_idx] if s_idx < len(yr) else np.array([], dtype=np.float32)
            
            res_l = classifier.predict_signal(seg_l, 22050)
            res_r = classifier.predict_signal(seg_r, 22050)
            
            left_evals.append(res_l)
            right_evals.append(res_r)
            
            eval_res = evaluate_audio_alert(
                left_hfc_prob=res_l["hfc_prob"],
                right_hfc_prob=res_r["hfc_prob"],
                left_class=res_l["class_name"],
                right_class=res_r["class_name"],
                video_anomaly=False,
                hfc_threshold=AUDIO_HFC_THRESHOLD,
                strict_dual_threshold=STRICT_DUAL_HFC_THRESHOLD
            )
            eval_results.append(eval_res)

        ldb = np.pad(ldb, (0, max(0, n_windows - len(ldb))), constant_values=-100)[:n_windows]
        rdb = np.pad(rdb, (0, max(0, n_windows - len(rdb))), constant_values=-100)[:n_windows]
        lf, rf = (ldb >= THRESHOLD_DB).astype(int), (rdb >= THRESHOLD_DB).astype(int)

        def sustained(flags):
            c, out = 0, []
            for f in flags:
                c = c + 1 if f else 0
                out.append(c)
            return np.array(out)

        log(f"Audio CNN OK: {n_windows} windows analyzed with PyTorch Audio Classifier")
        return dict(ldb=ldb, rdb=rdb, lf=lf, rf=rf,
                    labn=(sustained(lf) >= ABN_MIN_WIN).astype(int),
                    rabn=(sustained(rf) >= ABN_MIN_WIN).astype(int),
                    left_evals=left_evals, right_evals=right_evals, eval_results=eval_results)
    except Exception as e:
        log(f"!!! AUDIO FAILED: {e}")
        traceback.print_exc()
        return None


def audio_state_at(audio, ts, video_anomaly=False):
    if audio is None or len(audio.get("eval_results", [])) == 0:
        return dict(
            ok=False, ldb=-100.0, rdb=-100.0, lf=0, rf=0, labn=0, rabn=0,
            label="No audio", audio_alert=False, audio_state="NO_AUDIO",
            reason="No audio available", severity="info",
            left_hfc_prob=0.0, right_hfc_prob=0.0,
            left_lfc_prob=0.0, right_lfc_prob=0.0,
            left_class="N/A", right_class="N/A",
            left_conf=0.0, right_conf=0.0, window_idx=0,
            audio_eval={"avg_hfc_prob": 0.0, "video_anomaly": False, "multimodal_alert": False}
        )
    
    w = min(int(ts / AUDIO_WIN), len(audio["eval_results"]) - 1)
    res_l = audio["left_evals"][w]
    res_r = audio["right_evals"][w]
    eval_res = evaluate_audio_alert(
        left_hfc_prob=res_l["hfc_prob"],
        right_hfc_prob=res_r["hfc_prob"],
        left_class=res_l["class_name"],
        right_class=res_r["class_name"],
        video_anomaly=video_anomaly,
        hfc_threshold=AUDIO_HFC_THRESHOLD,
        strict_dual_threshold=STRICT_DUAL_HFC_THRESHOLD
    )
    
    return dict(
        ok=True,
        ldb=float(audio["ldb"][w]),
        rdb=float(audio["rdb"][w]),
        lf=int(audio["lf"][w]),
        rf=int(audio["rf"][w]),
        labn=int(audio["labn"][w]),
        rabn=int(audio["rabn"][w]),
        left_hfc_prob=res_l["hfc_prob"],
        right_hfc_prob=res_r["hfc_prob"],
        left_lfc_prob=res_l["lfc_prob"],
        right_lfc_prob=res_r["lfc_prob"],
        left_class=res_l["class_name"],
        right_class=res_r["class_name"],
        left_conf=res_l["confidence"],
        right_conf=res_r["confidence"],
        audio_eval=eval_res,
        audio_alert=eval_res["audio_alert"],
        audio_state=eval_res["audio_state"],
        reason=eval_res["reason"],
        severity=eval_res["severity"],
        label=eval_res["audio_state"],
        window_idx=w
    )


# ================================================================
# Misc helpers
# ================================================================
def overlap_ratio(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    amin = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / amin if amin > 0 else 0.0


def draw_panel(h, rows, herd, total_ids, alerts, astate, ts):
    p = np.full((h, PANEL_W, 3), 30, np.uint8)
    put = lambda t, x, y, s=0.45, c=(255, 255, 255), th=1: cv2.putText(
        p, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, th)
    put("MULTIMODAL COW DASHBOARD", 10, 22, 0.55, (0, 255, 255), 2)
    put(f"t={ts:6.1f}s  Total cows: {total_ids}  Active: {len(rows)}", 10, 44)
    ac = (0, 0, 255) if astate["label"] not in ("Normal", "No audio", "NO_HFC_EVENT") else (0, 220, 0)
    put(f"Audio: {astate['label']}  L {astate['ldb']:.1f}dB  R {astate['rdb']:.1f}dB", 10, 64, 0.45, ac)
    put("Herd: " + "  ".join(f"{b[:5]}:{herd.get(b, 0)}" for b in BEHAVIOURS), 10, 84, 0.38, (200, 200, 200))
    put("ID     Behaviour     Conf   Health", 10, 108, 0.45, (150, 255, 150))
    y = 128
    for r in rows[:9]:
        put(f"{r['id']:<5}", 10, y)
        put(r["beh"], 60, y, 0.45, COLORS[r["beh"]], 1)
        put(f"{r['conf']*100:4.0f}%", 185, y)
        put(r["health"][:22], 240, y, 0.4,
            (0, 0, 255) if r["health"] != "Healthy" else (0, 220, 0))
        y += 22
    put("ALERT HISTORY", 10, 345, 0.5, (0, 165, 255), 2)
    y = 365
    for a in list(alerts)[-5:][::-1]:
        put(a[:60], 10, y, 0.36, (80, 80, 255))
        y += 20
    return p


# ================================================================
# MAIN
# ================================================================
def run(video_path, left_audio, right_audio, out_video, out_csv, out_html,
        prediction_id, db_conn, telegram_callback=None, show_window=True):

    log(f"START pid={prediction_id}")
    log(f"video     = {video_path}  exists={os.path.exists(video_path)}")
    log(f"left      = {left_audio}")
    log(f"right     = {right_audio}")
    log(f"out_video = {out_video}")

    # ---------- Sanity: input video ----------
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video missing: {video_path}")

    # ---------- Probe Stream Durations & Time Sync ----------
    session_meta = probe_file_durations(video_path, left_audio, right_audio)
    log(f"Session Sync: {session_meta['sync_status']} (Video: {session_meta['video_duration']}s, L: {session_meta['left_duration']}s, R: {session_meta['right_duration']}s)")

    # Fetch user_id if available
    user_id = 1
    try:
        p_row = db_conn.execute("SELECT user_id, session_code FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        if p_row:
            user_id = p_row["user_id"]
            session_code = p_row["session_code"] or generate_session_code()
        else:
            session_code = generate_session_code()
    except Exception:
        session_code = generate_session_code()

    # ---------- Make sure output dir exists ----------
    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv),   exist_ok=True)
    os.makedirs(os.path.dirname(out_html),  exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    # ---------- Load YOLO ----------
    log(f"Loading YOLO model: {YOLO_MODEL}  exists={os.path.exists(YOLO_MODEL)}")
    try:
        model = YOLO(YOLO_MODEL)
        log("YOLO loaded OK")
    except Exception as e:
        log(f"!!! YOLO load failed: {e}")
        raise

    # ---------- DeepSORT ----------
    tracker = DeepSort(max_age=30, n_init=2, max_iou_distance=0.7)
    log("DeepSORT ready")

    # ---------- Audio ----------
    audio = analyse_audio(video_path, left_audio, right_audio)
    w_v, w_a = (W_VIDEO, W_AUDIO) if audio is not None else (1.0, 0.0)
    log(f"Fusion weights: video={w_v} audio={w_a}")

    # ---------- Open video ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    log(f"Video opened: fps={fps:.2f}  frames={total_frames}")

    # ---------- VideoWriter ----------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video, fourcc, fps, (FRAME_W + PANEL_W, FRAME_H))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for {out_video}")
    log(f"VideoWriter ready → {out_video}")

    S, seen_ids = {}, set()
    alerts_log, last_alert = deque(maxlen=50), {}
    frame_no, last_audio_row = 0, -1
    telegram_sent_set = set()

    cow_frame_obs = defaultdict(list)
    all_active_frames = []

    WINDOW_NAME = "Multimodal Cow Monitoring (Video 65% + Audio 35%)"
    if show_window:
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, FRAME_W + PANEL_W, FRAME_H)
            log("OpenCV window opened")
        except Exception as e:
            log(f"cv2 window failed: {e}")
            show_window = False

    def raise_alert(kind, key, ts, severity, msg, cow_id, frame=None, risk_score=0.0, factors=""):
        if ts - last_alert.get((kind, key), -1e9) < ALERT_COOLDOWN_S:
            return
        last_alert[(kind, key)] = ts
        alerts_log.append(f"[{ts:6.1f}s] {kind}: {msg}")
        log(f"ALERT {kind}: {msg}")

        img_path = None
        if frame is not None:
            try:
                img_path = os.path.join(FRAME_DIR, f"p{prediction_id}_{kind}_{int(ts*1000)}.jpg")
                ok = cv2.imwrite(img_path, frame)
                if not ok:
                    img_path = None
            except Exception as e:
                log(f"frame save failed: {e}")
                img_path = None

        try:
            db_conn.execute("""
                INSERT INTO prediction_alerts(prediction_id, ts, alert_type, cow_id,
                                              severity, message, image_frame, risk_score, contributing_factors)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (prediction_id, ts, kind, str(cow_id), severity, msg, img_path, risk_score, factors))
            db_conn.commit()
        except Exception as e:
            log(f"DB alert insert failed: {e}")

        if telegram_callback and key not in telegram_sent_set:
            telegram_sent_set.add(key)
            try:
                telegram_callback(msg, img_path)
            except Exception as e:
                log(f"Telegram failed: {e}")

    user_quit = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            frame_no += 1
            ts = frame_no / fps
            astate = audio_state_at(audio, ts)

            # ---- YOLO ----
            dets = []
            try:
                for r in model.predict(frame, conf=CONF, iou=IOU, verbose=False):
                    if r.boxes is None:
                        continue
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        dets.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                     float(b.conf[0]), model.names[int(b.cls[0])]))
            except Exception as e:
                log(f"YOLO predict failed on frame {frame_no}: {e}")

            # ---- DeepSORT ----
            try:
                tracks = [t for t in tracker.update_tracks(dets, frame=frame) if t.is_confirmed()]
            except Exception as e:
                log(f"DeepSORT failed: {e}")
                tracks = []

            active = []
            for t in tracks:
                tid = str(t.track_id)
                l, tp, r_, bt = [float(v) for v in t.to_ltrb()]
                box = (max(0, l), max(0, tp), min(FRAME_W - 1, r_), min(FRAME_H - 1, bt))
                st = S.setdefault(tid, dict(hist=deque(maxlen=VIDEO_SMOOTH), prev=None, vel=0.0,
                                            ema=None, cur=None, lying_since=None, agg_n=0))
                seen_ids.add(tid)
                if t.time_since_update == 0:
                    idx = ALIAS.get(str(t.get_det_class()).lower())
                    if idx is not None:
                        st["hist"].append((idx, t.get_det_conf() or 0.5))
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                diag = max(1.0, np.hypot(box[2] - box[0], box[3] - box[1]))
                if st["prev"] is not None:
                    v = np.hypot(cx - st["prev"][0], cy - st["prev"][1]) / diag * fps
                    st["vel"] = 0.7 * st["vel"] + 0.3 * v
                st["prev"] = (cx, cy)
                if st["hist"]:
                    active.append(dict(id=tid, box=box, cx=cx, st=st))

            # ---- per-cow fusion ----
            rows = []
            frame_boxes = []
            for a in active:
                st, tid = a["st"], a["id"]
                base = np.zeros(6)
                for idx, c in st["hist"]:
                    base[idx] += c
                s = base.sum()
                if s == 0:
                    base = np.ones(6) / 6
                else:
                    base /= s
                motion = float(np.clip((st["vel"] - AGG_SPEED_LO) / (AGG_SPEED_HI - AGG_SPEED_LO), 0, 1))
                contact = max([overlap_ratio(a["box"], o["box"]) > 0.05 for o in active if o is not a] or [False])
                agg_v = motion * (0.6 + 0.4 * float(contact))
                pv = base * (1 - agg_v)
                pv[4] += agg_v

                side = "left" if a["cx"] < FRAME_W / 2 else "right"
                side_high = astate["lf"] if side == "left" else astate["rf"]
                side_abn  = astate["labn"] if side == "left" else astate["rabn"]
                pa = PA_ABNORMAL if side_abn else (PA_LOUD if side_high else PA_QUIET)

                fused = w_v * pv + w_a * pa
                st["ema"] = fused if st["ema"] is None else (1 - FUSED_EMA) * st["ema"] + FUSED_EMA * fused
                fs = st["ema"] / st["ema"].sum()
                k = int(np.argmax(fs))
                beh, conf = BEHAVIOURS[k], float(fs[k])

                st["lying_since"] = (st["lying_since"] or ts) if beh == "lying" else None
                st["agg_n"] = st["agg_n"] + 1 if fs[4] >= AGG_ALERT_SCORE else 0
                health = "Healthy"
                if st["agg_n"] >= AGG_ALERT_FRAMES:
                    health = "ALERT: Aggressive"
                    raise_alert("AGGRESSIVE", f"agg_{tid}_{int(ts)}", ts, "high",
                                f"Cow {tid} aggressive (score {fs[4]:.2f})", tid, frame.copy())
                elif side_abn:
                    health = "Abnormal vocalization"
                elif st["lying_since"] and ts - st["lying_since"] > LYING_LIMIT_S:
                    health = "Check: prolonged lying"

                rows.append(dict(id=tid, beh=beh, conf=conf, health=health, side=side, box=a["box"]))
                frame_boxes.append(dict(id=tid, box=a["box"]))

                cow_frame_obs[tid].append(dict(
                    ts=ts, beh=beh, conf=conf, vel=float(st["vel"]), side=side, box=a["box"]
                ))

                if frame_no % SAVE_EVERY == 0:
                    try:
                        db_conn.execute("""
                            INSERT INTO prediction_results(prediction_id, cow_id, behaviour,
                                                           confidence, health, history)
                            VALUES(?,?,?,?,?,?)
                        """, (prediction_id, tid, beh, conf, health, f"{ts:.1f}s:{beh}"))
                        db_conn.commit()
                    except Exception as e:
                        log(f"result insert failed: {e}")

                x1, y1, x2, y2 = map(int, a["box"])
                col = COLORS[beh]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                lab = f"Cow {tid} | {beh} {conf*100:.0f}%"
                cv2.putText(frame, lab, (x1 + 3, max(12, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            all_active_frames.append(dict(ts=ts, boxes=frame_boxes))

            # ---- PyTorch Audio CNN Alert + Debug Logging ----
            if DEBUG_MULTIMODAL_ALERTS and astate["ok"] and astate.get("window_idx") != last_audio_row:
                last_audio_row = astate["window_idx"]
                w_start = astate["window_idx"] * AUDIO_WIN
                w_end = w_start + AUDIO_WIN
                log("-" * 50)
                log("AUDIO ALERT DEBUG")
                log(f"Window:\n{int(w_start//60):02d}:{w_start%60:05.2f} - {int(w_end//60):02d}:{w_end%60:05.2f}")
                log(f"LEFT:\nclass = {astate['left_class']}\nHFC = {astate['left_hfc_prob']:.4f}\nLFC = {astate['left_lfc_prob']:.4f}")
                log(f"RIGHT:\nclass = {astate['right_class']}\nHFC = {astate['right_hfc_prob']:.4f}\nLFC = {astate['right_lfc_prob']:.4f}")
                log(f"Audio agreement:\n{astate['audio_eval']['avg_hfc_prob']:.4f}")
                log(f"Combined audio state:\n{astate['audio_state']}")
                log(f"Video anomaly:\n{str(astate['audio_eval']['video_anomaly']).upper()}")
                log(f"Audio alert condition:\n{str(astate['audio_alert']).upper()}")
                log(f"Multimodal alert condition:\n{str(astate['audio_eval']['multimodal_alert']).upper()}")
                log(f"Final alert:\n{str(astate['audio_alert']).upper()}")
                log(f"Reason:\n{astate['reason']}")
                log("-" * 50)

            if astate["ok"] and astate["audio_alert"]:
                cow_ids_str = ",".join(seen_ids) if seen_ids else "Herd"
                raise_alert("HIGH_FREQUENCY_CALL_ALERT", f"hfc_{astate['window_idx']}", ts,
                            astate["severity"], astate["reason"], cow_ids_str, frame.copy())

            # ---- sound group + legacy vocalization alert ----
            group = {"left": [r["id"] for r in rows if r["side"] == "left"],
                     "right": [r["id"] for r in rows if r["side"] == "right"]}
            if astate["ok"] and (astate["labn"] or astate["rabn"]):
                for sd, flag in (("left", astate["labn"]), ("right", astate["rabn"])):
                    if flag:
                        raise_alert("ABNORMAL_VOCALIZATION", f"vocal_{sd}_{astate['window_idx']}", ts, "medium",
                                    f"Abnormal vocalization on {sd.upper()} side",
                                    ",".join(group[sd]), frame.copy())

            # ---- panel + write + show ----
            herd_now = Counter(r["beh"] for r in rows)
            panel = draw_panel(FRAME_H, rows, herd_now, len(seen_ids), alerts_log, astate, ts)
            canvas = np.hstack([frame, panel])
            out.write(canvas)

            if frame_no % 30 == 0:
                log(f"frame {frame_no}/{total_frames}  active={len(rows)}  cows_total={len(seen_ids)}")

            if show_window:
                try:
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log("User pressed Q — stopping early.")
                        user_quit = True
                        break
                except Exception as e:
                    log(f"imshow failed: {e}")
                    show_window = False
    finally:
        try: cap.release()
        except Exception as e: log(f"cap.release error: {e}")
        try: out.release()
        except Exception as e: log(f"out.release error: {e}")
        if show_window:
            try: cv2.destroyWindow(WINDOW_NAME)
            except Exception: pass
        try: reencode_for_browser(out_video)
        except Exception as e: log(f"re-encode wrapper failed: {e}")

    time.sleep(0.5)

    # ================================================================
    # POST-PROCESSING INTELLIGENCE ENGINE (PHASES 3 - 25)
    # ================================================================
    log("=== RUNNING POST-PROCESSING INTELLIGENCE ENGINE ===")

    session_duration = round(frame_no / fps if fps > 0 else session_meta["video_duration"], 1)

    # 1. Temporal Windows & Acoustic Zones
    session_windows = []
    if audio and "eval_results" in audio:
        for w_idx in range(len(audio["eval_results"])):
            w_start = w_idx * AUDIO_WIN
            w_end = w_start + AUDIO_WIN
            res_l = audio["left_evals"][w_idx]
            res_r = audio["right_evals"][w_idx]

            ac_info = analyze_acoustic_zone(
                res_l["hfc_prob"], res_r["hfc_prob"],
                res_l["confidence"], res_r["confidence"],
                hfc_threshold=AUDIO_HFC_THRESHOLD, diff_threshold=ACOUSTIC_DIFF_THRESHOLD
            )

            # Count active cows in window
            w_cows = set()
            for tid, obs_list in cow_frame_obs.items():
                if any(w_start <= o['ts'] <= w_end for o in obs_list):
                    w_cows.add(tid)

            w_risk = round(min(100.0, ac_info["avg_hfc"] * 80.0), 1)

            session_windows.append({
                "window_index": w_idx,
                "start_time": round(w_start, 1),
                "end_time": round(w_end, 1),
                "cow_count": len(w_cows),
                "left_audio_class": res_l["class_name"],
                "left_audio_confidence": round(res_l["hfc_prob"], 4),
                "right_audio_class": res_r["class_name"],
                "right_audio_confidence": round(res_r["hfc_prob"], 4),
                "audio_agreement": ac_info["audio_agreement"],
                "acoustic_zone": ac_info["acoustic_zone"],
                "risk_score": w_risk
            })

            try:
                db_conn.execute("""
                    INSERT INTO session_windows(prediction_id, window_index, start_time, end_time,
                                                cow_count, left_audio_class, left_audio_confidence,
                                                right_audio_class, right_audio_confidence,
                                                audio_agreement, acoustic_zone, risk_score)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """, (prediction_id, w_idx, w_start, w_end, len(w_cows),
                      res_l["class_name"], res_l["hfc_prob"],
                      res_r["class_name"], res_r["hfc_prob"],
                      ac_info["audio_agreement"], ac_info["acoustic_zone"], w_risk))
            except Exception as e:
                log(f"window DB insert warning: {e}")

    # 2. Individual Cow Aggregation & Timelines
    cow_agg = aggregate_cow_profiles(cow_frame_obs, total_session_fps=fps)
    cow_profiles_dict = cow_agg["cow_profiles"]
    behavior_timelines = cow_agg["behavior_timelines"]

    # 3. Spatial Interaction Graph
    interactions = build_cow_interaction_graph(all_active_frames)

    # 4. Multimodal Events Engine
    multimodal_events = detect_multimodal_events(session_windows, cow_frame_obs, alerts_log)

    # 5. Statistical Anomaly Engine
    anomalies = detect_anomalies(cow_profiles_dict, session_windows)

    # 6. Baseline & Risk Evaluation per Cow
    for cow_id, profile in cow_profiles_dict.items():
        # Baseline check
        base_info = compare_cow_baseline(db_conn, user_id, cow_id, profile)
        update_cow_baseline(db_conn, user_id, cow_id, profile)

        # Cow events & interactions
        c_events = [e for e in multimodal_events if str(e['cow_id']) == str(cow_id)]
        c_anoms  = [a for a in anomalies if str(a['cow_id']) == str(cow_id)]
        c_inters = [i for i in interactions if str(i['cow_a']) == str(cow_id) or str(i['cow_b']) == str(cow_id)]

        profile["audio_event_count"] = sum(1 for w in session_windows if (w['left_audio_class'] == 'High-frequency call (HFC)' or w['right_audio_class'] == 'High-frequency call (HFC)'))
        profile["anomaly_count"] = len(c_anoms)
        profile["interaction_count"] = len(c_inters)

        risk_info = calculate_cow_risk(profile, c_events, c_anoms, c_inters)
        profile["risk_score"] = risk_info["risk_score"]
        profile["risk_status"] = risk_info["status"]
        profile["contributor_str"] = risk_info["contributor_str"]

        try:
            db_conn.execute("""
                INSERT INTO cow_profiles(prediction_id, cow_id, first_seen, last_seen, visible_duration,
                                         dominant_behavior, standing_pct, lying_pct, eating_pct,
                                         drinking_pct, aggressive_pct, rumination_pct, movement_score,
                                         transitions_count, interaction_count, anomaly_count,
                                         audio_event_count, risk_score, risk_status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (prediction_id, cow_id, profile['first_seen'], profile['last_seen'], profile['visible_duration'],
                  profile['dominant_behavior'], profile['standing_pct'], profile['lying_pct'], profile['eating_pct'],
                  profile['drinking_pct'], profile['aggressive_pct'], profile['rumination_pct'], profile['movement_score'],
                  profile['transitions_count'], profile['interaction_count'], profile['anomaly_count'],
                  profile['audio_event_count'], profile['risk_score'], profile['risk_status']))

            for seg in behavior_timelines.get(cow_id, []):
                db_conn.execute("""
                    INSERT INTO behavior_timelines(prediction_id, cow_id, start_time, end_time, behavior, confidence, movement)
                    VALUES(?,?,?,?,?,?,?)
                """, (prediction_id, cow_id, seg['start_time'], seg['end_time'], seg['behavior'], seg['confidence'], seg['movement']))
        except Exception as e:
            log(f"cow profile DB insert warning: {e}")

    # 7. DB Persistence for Multimodal Events, Anomalies, Interactions
    for e in multimodal_events:
        try:
            db_conn.execute("""
                INSERT INTO multimodal_events(prediction_id, timestamp, cow_id, event_type, visual_evidence,
                                             audio_evidence, left_confidence, right_confidence, audio_agreement,
                                             event_confidence, severity, description)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """, (prediction_id, e['timestamp'], str(e['cow_id']), e['event_type'], e['visual_evidence'],
                  e['audio_evidence'], e['left_confidence'], e['right_confidence'], e['audio_agreement'],
                  e['event_confidence'], e['severity'], e['description']))
        except Exception as ex:
            log(f"event DB insert warning: {ex}")

    for a in anomalies:
        try:
            db_conn.execute("""
                INSERT INTO anomalies(prediction_id, timestamp, cow_id, feature, baseline_val, current_val, deviation_pct, severity, explanation)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (prediction_id, a['timestamp'], str(a['cow_id']), a['feature'], a['baseline_val'], a['current_val'], a['deviation_pct'], a['severity'], a['explanation']))
        except Exception as ex:
            log(f"anomaly DB insert warning: {ex}")

    for i in interactions:
        try:
            db_conn.execute("""
                INSERT INTO cow_interactions(prediction_id, cow_a, cow_b, start_time, end_time, duration, proximity_score, interaction_type)
                VALUES(?,?,?,?,?,?,?,?)
            """, (prediction_id, i['cow_a'], i['cow_b'], i['start_time'], i['end_time'], i['duration'], i['proximity_score'], i['interaction_type']))
        except Exception as ex:
            log(f"interaction DB insert warning: {ex}")

    # 8. Herd Level Risk Score & Session Update
    herd_risk, herd_status = calculate_herd_risk(cow_profiles_dict, multimodal_events, anomalies)
    log(f"HERD INTELLIGENCE SUMMARY: Total Cows={len(seen_ids)}, Events={len(multimodal_events)}, Anomalies={len(anomalies)}, Risk Score={herd_risk} ({herd_status})")

    try:
        db_conn.execute("""
            UPDATE predictions SET
                session_code    = ?,
                duration        = ?,
                left_duration   = ?,
                right_duration  = ?,
                sync_status     = ?,
                total_cows      = ?,
                total_events    = ?,
                total_anomalies = ?,
                overall_risk    = ?
            WHERE id = ?
        """, (session_code, session_duration, session_meta["left_duration"], session_meta["right_duration"],
              session_meta["sync_status"], len(seen_ids), len(multimodal_events), len(anomalies), herd_risk, prediction_id))
        db_conn.commit()
    except Exception as e:
        log(f"session DB update warning: {e}")

    # 9. Upgraded Reports
    session_info = {
        "session_code": session_code,
        "duration": session_duration,
        "sync_status": session_meta["sync_status"],
        "overall_risk": herd_risk
    }
    herd_analytics = calculate_herd_analytics(cow_profiles_dict, session_windows, multimodal_events, anomalies)

    generate_csv_report(out_csv, session_info, session_windows, cow_profiles_dict, multimodal_events, anomalies)
    generate_html_report(out_html, session_info, session_windows, cow_profiles_dict, multimodal_events, anomalies, herd_analytics)

    log(f"DONE pid={prediction_id} cows={len(seen_ids)} events={len(multimodal_events)} alerts={len(alerts_log)}")
    return dict(
        total_cows=len(seen_ids),
        total_events=len(multimodal_events),
        total_anomalies=len(anomalies),
        total_alerts=len(alerts_log),
        overall_risk=herd_risk,
        user_quit=user_quit
    )
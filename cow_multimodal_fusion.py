"""
pipeline.py
------------
Reusable multimodal cow-behaviour pipeline.
Call pipeline.run(...) from the Flask app.
"""
import os, csv, time, sqlite3
from collections import deque, Counter, defaultdict
import cv2, numpy as np, librosa
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import YOLO_MODEL

# ------------- Config copied from your script -------------
CONF, IOU = 0.30, 0.45
FRAME_W, FRAME_H = 640, 480
PANEL_W = 440
W_VIDEO, W_AUDIO = 0.65, 0.35
THRESHOLD_DB = -15.6
AUDIO_WIN = 1.0
ABN_MIN_WIN = 2
EPS = 1e-10
VIDEO_SMOOTH = 15
FUSED_EMA = 0.30
AGG_SPEED_LO, AGG_SPEED_HI = 0.5, 2.0
AGG_ALERT_SCORE = 0.40
AGG_ALERT_FRAMES = 5
ALERT_COOLDOWN_S = 5.0
LYING_LIMIT_S = 600
SAVE_EVERY = 5

BEHAVIOURS = ["standing", "lying", "eating", "drinking", "aggressive", "rumination"]
ALIAS = {"standing": 0, "lying_down": 1, "lying": 1, "feeding": 2, "eating": 2,
         "drinking": 3, "aggressive": 4, "rumination": 5, "ruminating": 5}
COLORS = {"standing": (0, 200, 0), "lying": (200, 150, 0), "eating": (0, 180, 255),
          "drinking": (255, 120, 0), "aggressive": (0, 0, 255), "rumination": (200, 0, 200)}

PA_QUIET    = np.array([.20, .20, .20, .10, .02, .28])
PA_LOUD     = np.array([.35, .10, .10, .05, .30, .10])
PA_ABNORMAL = np.array([.25, .03, .03, .02, .60, .07])


# ============================================================
# Helpers (audio, panel, etc.)
# ============================================================
def window_levels(y, sr):
    n = int(AUDIO_WIN * sr); out = []
    for s in range(0, len(y), n):
        seg = y[s:s+n]
        if len(seg) < 2048:
            out.append(-100.0); continue
        rms = librosa.feature.rms(y=seg, frame_length=2048, hop_length=512)[0]
        out.append(float(np.percentile(20*np.log10(np.maximum(rms, EPS)), 95)))
    return np.array(out)


def analyse_audio(video_path, left_path, right_path):
    try:
        if left_path and right_path:
            yl, srl = librosa.load(left_path, sr=None, mono=True)
            yr, srr = librosa.load(right_path, sr=None, mono=True)
        else:
            y, srl = librosa.load(video_path, sr=None, mono=False)
            srr = srl
            yl, yr = (y, y) if y.ndim == 1 else (y[0], y[1])
        ldb, rdb = window_levels(yl, srl), window_levels(yr, srr)
        n = max(len(ldb), len(rdb))
        ldb = np.pad(ldb, (0, n-len(ldb)), constant_values=-100)
        rdb = np.pad(rdb, (0, n-len(rdb)), constant_values=-100)
        lf, rf = (ldb >= THRESHOLD_DB).astype(int), (rdb >= THRESHOLD_DB).astype(int)
        def sustained(f):
            c, o = 0, []
            for x in f:
                c = c + 1 if x else 0
                o.append(c)
            return np.array(o)
        return dict(ldb=ldb, rdb=rdb, lf=lf, rf=rf,
                    labn=(sustained(lf) >= ABN_MIN_WIN).astype(int),
                    rabn=(sustained(rf) >= ABN_MIN_WIN).astype(int))
    except Exception as e:
        print("Audio failed:", e); return None


def audio_state_at(audio, ts):
    if audio is None:
        return dict(ok=False, ldb=-100, rdb=-100, lf=0, rf=0, labn=0, rabn=0, label="No audio")
    w = min(int(ts/AUDIO_WIN), len(audio["lf"])-1)
    lf, rf = int(audio["lf"][w]), int(audio["rf"][w])
    label = {(1,0):"HIGH LEFT",(0,1):"HIGH RIGHT",(1,1):"HIGH BOTH",(0,0):"Normal"}[(lf,rf)]
    return dict(ok=True, ldb=float(audio["ldb"][w]), rdb=float(audio["rdb"][w]),
                lf=lf, rf=rf, labn=int(audio["labn"][w]), rabn=int(audio["rabn"][w]), label=label)


def overlap_ratio(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    amin = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return inter/amin if amin > 0 else 0.0


def draw_panel(h, rows, herd, total_ids, alerts, astate, ts):
    p = np.full((h, PANEL_W, 3), 30, np.uint8)
    put = lambda t,x,y,s=0.45,c=(255,255,255),th=1: cv2.putText(p,t,(x,y),cv2.FONT_HERSHEY_SIMPLEX,s,c,th)
    put("MULTIMODAL COW DASHBOARD", 10, 22, 0.55, (0,255,255), 2)
    put(f"t={ts:6.1f}s  Total: {total_ids}  Active: {len(rows)}", 10, 44)
    ac = (0,0,255) if astate["label"] not in ("Normal","No audio") else (0,220,0)
    put(f"Audio: {astate['label']}  L {astate['ldb']:.1f}  R {astate['rdb']:.1f}", 10, 64, 0.45, ac)
    put("Herd: " + "  ".join(f"{b[:5]}:{herd.get(b,0)}" for b in BEHAVIOURS), 10, 84, 0.38, (200,200,200))
    put("ID     Behaviour     Conf   Health", 10, 108, 0.45, (150,255,150))
    y = 128
    for r in rows[:9]:
        put(f"{r['id']:<5}", 10, y)
        put(r["beh"], 60, y, 0.45, COLORS[r["beh"]], 1)
        put(f"{r['conf']*100:4.0f}%", 185, y)
        put(r["health"][:22], 240, y, 0.4, (0,0,255) if r["health"]!="Healthy" else (0,220,0))
        y += 22
    put("ALERT HISTORY", 10, 345, 0.5, (0,165,255), 2)
    y = 365
    for a in list(alerts)[-5:][::-1]:
        put(a[:60], 10, y, 0.36, (80,80,255)); y += 20
    return p


# ============================================================
# Main entry-point used by Flask
# ============================================================
def run(video_path, left_audio, right_audio, out_video, out_csv, out_html,
        prediction_id, db_conn, telegram_callback=None):
    """
    Executes the full pipeline and returns a summary dict.
    Also inserts per-cow results into 'prediction_results' and alerts into 'prediction_alerts'.
    """
    model = YOLO(YOLO_MODEL)
    tracker = DeepSort(max_age=30, n_init=2, max_iou_distance=0.7)

    audio = analyse_audio(video_path, left_audio, right_audio)
    w_v, w_a = (W_VIDEO, W_AUDIO) if audio is not None else (1.0, 0.0)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                          (FRAME_W + PANEL_W, FRAME_H))

    S, seen_ids = {}, set()
    alerts_log, last_alert = deque(maxlen=50), {}
    frame_no, last_audio_row = 0, -1
    telegram_sent_set = set()

    def raise_alert(kind, key, ts, severity, msg, cow_id, frame=None):
        if ts - last_alert.get((kind, key), -1e9) < ALERT_COOLDOWN_S:
            return
        last_alert[(kind, key)] = ts
        alerts_log.append(f"[{ts:6.1f}s] {kind}: {msg}")

        # save frame image (for Telegram)
        img_path = None
        if frame is not None:
            img_dir = os.path.join(os.path.dirname(out_video), "frames")
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"p{prediction_id}_{kind}_{int(ts*1000)}.jpg")
            cv2.imwrite(img_path, frame)

        # store in DB
        db_conn.execute("""
            INSERT INTO prediction_alerts(prediction_id, ts, alert_type, cow_id, severity, message, image_frame)
            VALUES(?,?,?,?,?,?,?)
        """, (prediction_id, ts, kind, str(cow_id), severity, msg, img_path))
        db_conn.commit()

        # Telegram
        if telegram_callback and key not in telegram_sent_set:
            telegram_sent_set.add(key)
            telegram_callback(msg, img_path)

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_no += 1
        ts = frame_no / fps
        astate = audio_state_at(audio, ts)

        dets = []
        for r in model.predict(frame, conf=CONF, iou=IOU, verbose=False):
            if r.boxes is None: continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                dets.append(([float(x1), float(y1), float(x2-x1), float(y2-y1)],
                             float(b.conf[0]), model.names[int(b.cls[0])]))

        tracks = [t for t in tracker.update_tracks(dets, frame=frame) if t.is_confirmed()]
        active = []
        for t in tracks:
            tid = str(t.track_id)
            l, tp, r_, bt = [float(v) for v in t.to_ltrb()]
            box = (max(0,l), max(0,tp), min(FRAME_W-1,r_), min(FRAME_H-1,bt))
            st = S.setdefault(tid, dict(hist=deque(maxlen=VIDEO_SMOOTH), prev=None, vel=0.0,
                                        ema=None, cur=None, lying_since=None, agg_n=0))
            seen_ids.add(tid)
            if t.time_since_update == 0:
                idx = ALIAS.get(str(t.get_det_class()).lower())
                if idx is not None:
                    st["hist"].append((idx, t.get_det_conf() or 0.5))
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            diag = max(1.0, np.hypot(box[2]-box[0], box[3]-box[1]))
            if st["prev"] is not None:
                v = np.hypot(cx-st["prev"][0], cy-st["prev"][1]) / diag * fps
                st["vel"] = 0.7*st["vel"] + 0.3*v
            st["prev"] = (cx, cy)
            if st["hist"]:
                active.append(dict(id=tid, box=box, cx=cx, st=st))

        rows = []
        for a in active:
            st, tid = a["st"], a["id"]
            base = np.zeros(6)
            for idx, c in st["hist"]:
                base[idx] += c
            base /= base.sum()
            motion = float(np.clip((st["vel"]-AGG_SPEED_LO)/(AGG_SPEED_HI-AGG_SPEED_LO), 0, 1))
            contact = max([overlap_ratio(a["box"], o["box"]) > 0.05 for o in active if o is not a] or [False])
            agg_v = motion * (0.6 + 0.4*float(contact))
            pv = base * (1-agg_v); pv[4] += agg_v

            side = "left" if a["cx"] < FRAME_W/2 else "right"
            side_high = astate["lf"] if side == "left" else astate["rf"]
            side_abn  = astate["labn"] if side == "left" else astate["rabn"]
            pa = PA_ABNORMAL if side_abn else (PA_LOUD if side_high else PA_QUIET)

            fused = w_v*pv + w_a*pa
            st["ema"] = fused if st["ema"] is None else (1-FUSED_EMA)*st["ema"] + FUSED_EMA*fused
            fs = st["ema"] / st["ema"].sum()
            k = int(np.argmax(fs))
            beh, conf = BEHAVIOURS[k], float(fs[k])

            st["lying_since"] = (st["lying_since"] or ts) if beh == "lying" else None
            st["agg_n"] = st["agg_n"] + 1 if fs[4] >= AGG_ALERT_SCORE else 0
            health = "Healthy"
            if st["agg_n"] >= AGG_ALERT_FRAMES:
                health = "ALERT: Aggressive"
                raise_alert("AGGRESSIVE", tid, ts, "high",
                            f"Cow {tid} aggressive (score {fs[4]:.2f})", tid, frame.copy())
            elif side_abn:
                health = "Abnormal vocalization"
            elif st["lying_since"] and ts - st["lying_since"] > LYING_LIMIT_S:
                health = "Check: prolonged lying"

            vk = int(np.argmax(pv))
            rows.append(dict(id=tid, beh=beh, conf=conf, health=health, side=side, box=a["box"]))

            if frame_no % SAVE_EVERY == 0:
                db_conn.execute("""
                    INSERT INTO prediction_results(prediction_id, cow_id, behaviour, confidence, health, history)
                    VALUES(?,?,?,?,?,?)
                """, (prediction_id, tid, beh, conf, health, f"{ts:.1f}s:{beh}"))
                db_conn.commit()

            x1, y1, x2, y2 = map(int, a["box"])
            col = COLORS[beh]
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            lab = f"Cow {tid} | {beh} {conf*100:.0f}%"
            cv2.putText(frame, lab, (x1+3, max(12, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)

        # sound-group + vocalization alert
        group = {"left":[r["id"] for r in rows if r["side"]=="left"],
                 "right":[r["id"] for r in rows if r["side"]=="right"]}
        if astate["ok"] and (astate["labn"] or astate["rabn"]):
            for sd, flag in (("left", astate["labn"]), ("right", astate["rabn"])):
                if flag:
                    raise_alert("ABNORMAL_VOCALIZATION", sd, ts, "medium",
                                f"Abnormal vocalization on {sd.upper()} side",
                                ",".join(group[sd]), frame.copy())

        herd_now = Counter(r["beh"] for r in rows)
        panel = draw_panel(FRAME_H, rows, herd_now, len(seen_ids), alerts_log, astate, ts)
        canvas = np.hstack([frame, panel])
        out.write(canvas)

    cap.release(); out.release()

    # CSV report
    hist = defaultdict(list)
    for row in db_conn.execute("SELECT cow_id, behaviour, confidence, health FROM prediction_results WHERE prediction_id=?", (prediction_id,)):
        hist[row["cow_id"]].append(row["behaviour"])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["Cow", "Behaviour_History", "Latest_Behaviour"])
        for cid, seq in hist.items():
            wr.writerow([f"Cow {cid}", " -> ".join(seq), seq[-1] if seq else ""])

    # HTML report
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<html><body style='font-family:Arial;background:#111;color:#eee'>")
        f.write(f"<h1>Dashboard — Prediction #{prediction_id}</h1>")
        f.write(f"<p>Total cows: {len(seen_ids)} | Alerts: {len(alerts_log)}</p>")
        f.write("<h2>Alerts</h2><ul>")
        for a in alerts_log:
            f.write(f"<li>{a}</li>")
        f.write("</ul></body></html>")

    return dict(total_cows=len(seen_ids), total_alerts=len(alerts_log))
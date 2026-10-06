"""
app.py — Flask Web Application
"""

import os, sys, uuid, threading, subprocess
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, send_file, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import (SECRET_KEY, UPLOAD_VIDEO_DIR, UPLOAD_AUDIO_DIR,
                    OUTPUT_DIR, FRAME_DIR, REPORT_DIR, LOG_DIR,
                    ALLOWED_VIDEO, ALLOWED_AUDIO)
from database import get_db, init_db
import time
app = Flask(__name__)
app.secret_key = SECRET_KEY


# ===============================================================
# Helpers
# ===============================================================
def allowed_file(name, allowed):
    return "." in name and name.rsplit(".", 1)[1].lower() in allowed

import shutil, subprocess

def normalize_audio_for_playback(src_path: str) -> str:
    """
    Re-encode an uploaded audio file into a browser-friendly, loudness-
    normalized MP3. Returns the path to the new file (or src_path on failure).

    - keeps the original sample rate / channel count
    - applies EBU R128 loudness normalization (-16 LUFS)
    - bitrate 192 kbps CBR — plays in every browser
    """
    if not src_path or not os.path.exists(src_path):
        return src_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[audio] ffmpeg not found — skipping normalization")
        return src_path

    root, _ext = os.path.splitext(src_path)
    out_path = root + "_norm.mp3"

    cmd = [
        ffmpeg, "-y",
        "-i", src_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "192k",
        out_path,
    ]
    print(f"[audio] normalizing: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
            print(f"[audio] ffmpeg failed rc={r.returncode}")
            print(r.stdout[-800:])
            return src_path

        # Replace the original with the normalized file
        try:
            os.remove(src_path)
        except Exception:
            pass
        os.rename(out_path, src_path)
        print(f"[audio] normalized OK → {src_path} ({os.path.getsize(src_path)} bytes)")
        return src_path
    except Exception as e:
        print(f"[audio] normalization error: {e}")
        return src_path

    
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def basename(p):
    """Get filename from an absolute path (Windows or POSIX)."""
    if not p:
        return None
    return os.path.basename(p.replace("\\", "/"))


# ===============================================================
# Media / report serving
# ===============================================================
@app.route("/media/video/<path:fname>")
@login_required
def serve_video(fname):
    for base in (UPLOAD_VIDEO_DIR, OUTPUT_DIR):
        full = os.path.abspath(os.path.join(base, fname))

        print(full)

        if os.path.isfile(full):
            return send_file(
                full,
                mimetype="video/mp4",
                conditional=True
            )

    abort(404)



@app.route("/media/audio/<path:fname>")
@login_required
def serve_audio(fname):
    full = os.path.join(UPLOAD_AUDIO_DIR, fname)
    if not os.path.exists(full):
        abort(404)
    return send_file(full, conditional=True)


@app.route("/media/frame/<path:fname>")
@login_required
def serve_frame(fname):
    full = os.path.join(FRAME_DIR, fname)
    if not os.path.exists(full):
        abort(404)
    return send_file(full, conditional=True)


@app.route("/report/<path:fname>")
@login_required
def serve_report(fname):
    full = os.path.join(REPORT_DIR, fname)
    if not os.path.exists(full):
        abort(404)
    as_attach = request.args.get("download") == "1"
    return send_file(full, as_attachment=as_attach, conditional=True)


# ===============================================================
# Auth
# ===============================================================
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email    = request.form["email"].strip()
        password = request.form["password"]
        confirm  = request.form["confirm"]
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))
        conn = get_db()
        try:
            conn.execute("INSERT INTO users(username,email,password) VALUES(?,?,?)",
                         (username, email, generate_password_hash(password)))
            conn.commit()
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Error: {e}", "danger")
        finally:
            conn.close()
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ===============================================================
# Dashboard
# ===============================================================
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    uid = session["user_id"]
    rows = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC", (uid,)).fetchall()
    
    # Calculate Aggregate Stats
    total_sessions = len(rows)
    latest = rows[0] if rows else None
    
    # Total unique cows across profiles or predictions total_cows sum
    total_cows_row = conn.execute("""
        SELECT COUNT(DISTINCT cow_id) as cow_cnt 
        FROM cow_profiles cp 
        JOIN predictions p ON cp.prediction_id = p.id 
        WHERE p.user_id = ?
    """, (uid,)).fetchone()
    total_cows = total_cows_row["cow_cnt"] if total_cows_row and total_cows_row["cow_cnt"] > 0 else sum(r["total_cows"] or 0 for r in rows)
    
    total_anomalies_row = conn.execute("""
        SELECT COUNT(*) as anom_cnt 
        FROM anomalies a 
        JOIN predictions p ON a.prediction_id = p.id 
        WHERE p.user_id = ?
    """, (uid,)).fetchone()
    total_anomalies = total_anomalies_row["anom_cnt"] if total_anomalies_row else sum(r["total_anomalies"] or 0 for r in rows)

    total_alerts_row = conn.execute("""
        SELECT COUNT(*) as alert_cnt 
        FROM prediction_alerts pa 
        JOIN predictions p ON pa.prediction_id = p.id 
        WHERE p.user_id = ?
    """, (uid,)).fetchone()
    total_alerts = total_alerts_row["alert_cnt"] if total_alerts_row else sum(r["total_alerts"] or 0 for r in rows)

    avg_risk = 0.0
    if rows:
        valid_risks = [r["overall_risk"] for r in rows if r["overall_risk"] is not None]
        if valid_risks:
            avg_risk = round(sum(valid_risks) / len(valid_risks), 1)

    # Herd Status Distribution (Normal, Attention, Elevated, High)
    status_counts = {"NORMAL": 0, "ATTENTION": 0, "ELEVATED": 0, "HIGH": 0}
    cow_profiles_rows = conn.execute("""
        SELECT cp.*, p.session_code 
        FROM cow_profiles cp 
        JOIN predictions p ON cp.prediction_id = p.id 
        WHERE p.user_id = ?
    """, (uid,)).fetchall()

    for cp in cow_profiles_rows:
        st = (cp["risk_status"] or "NORMAL").upper()
        if st in status_counts:
            status_counts[st] += 1
        else:
            status_counts["NORMAL"] += 1

    # Cows Requiring Attention (Risk >= 30 or Non-Normal)
    attention_cows = [dict(cp) for cp in cow_profiles_rows if (cp["risk_score"] or 0) >= 30 or (cp["risk_status"] or "").upper() != "NORMAL"]
    attention_cows.sort(key=lambda x: x["risk_score"] or 0, reverse=True)

    # Recent Multimodal Events
    recent_events = conn.execute("""
        SELECT me.*, p.session_code 
        FROM multimodal_events me 
        JOIN predictions p ON me.prediction_id = p.id 
        WHERE p.user_id = ? 
        ORDER BY me.id DESC LIMIT 10
    """, (uid,)).fetchall()

    conn.close()

    stats = {
        "total_cows": total_cows,
        "total_sessions": total_sessions,
        "total_anomalies": total_anomalies,
        "total_alerts": total_alerts,
        "avg_risk": avg_risk,
        "herd_status": status_counts
    }

    return render_template("dashboard.html", 
                           predictions=rows, 
                           latest=latest,
                           stats=stats, 
                           attention_cows=attention_cows[:6],
                           recent_events=recent_events,
                           basename=basename)


@app.route("/prediction/<int:pid>")
@login_required
def prediction_detail(pid):
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                        (pid, session["user_id"])).fetchone()
    if not pred:
        conn.close()
        abort(404)
    results = conn.execute("SELECT * FROM prediction_results WHERE prediction_id=?", (pid,)).fetchall()
    alerts  = conn.execute("SELECT * FROM prediction_alerts WHERE prediction_id=? ORDER BY ts", (pid,)).fetchall()
    conn.close()
    return render_template("prediction_detail.html", pred=pred, results=results, alerts=alerts)


@app.route("/prediction/<int:pid>/delete", methods=["POST"])
@login_required
def delete_prediction(pid):
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                        (pid, session["user_id"])).fetchone()
    if not pred:
        conn.close()
        abort(404)

    for key in ("video_input", "left_audio", "right_audio",
                "video_output", "csv_report", "html_report"):
        p = pred[key]
        if p and os.path.exists(p):
            try: os.remove(p)
            except Exception: pass

    for row in conn.execute("SELECT image_frame FROM prediction_alerts WHERE prediction_id=?", (pid,)):
        img = row["image_frame"]
        if img and os.path.exists(img):
            try: os.remove(img)
            except Exception: pass

    conn.execute("DELETE FROM predictions WHERE id=?", (pid,))
    conn.commit()
    conn.close()

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify(ok=True)
    flash("Prediction deleted.", "success")
    return redirect(url_for("dashboard"))


# ===============================================================
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        video = request.files.get("video")
        left  = request.files.get("left_audio")
        right = request.files.get("right_audio")

        farm_name = request.form.get("farm_name", "Main Farm").strip() or "Main Farm"
        location  = request.form.get("location", "Barn A").strip() or "Barn A"
        camera_id = request.form.get("camera_id", "Cam 1").strip() or "Cam 1"

        if not video or video.filename == "":
            flash("Video file is required.", "danger")
            return redirect(url_for("predict"))
        if not allowed_file(video.filename, ALLOWED_VIDEO):
            flash("Unsupported video format.", "danger")
            return redirect(url_for("predict"))

        tag   = f"{session['user_id']}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}"
        vname = f"{tag}_{secure_filename(video.filename)}"
        vpath = os.path.join(UPLOAD_VIDEO_DIR, vname)
        video.save(vpath)
        print(f"[app] Saved video: {vpath} ({os.path.getsize(vpath)} bytes)")

        lpath = rpath = None
        if left and left.filename and allowed_file(left.filename, ALLOWED_AUDIO):
            lname = f"{tag}_L_{secure_filename(left.filename)}"
            lpath = os.path.join(UPLOAD_AUDIO_DIR, lname)
            left.save(lpath)
            print(f"[app] Saved left audio: {lpath} ({os.path.getsize(lpath)} bytes)")
            # Normalize for browser playback + loudness
            lpath = normalize_audio_for_playback(lpath)
            print(f"[app] Final left audio: {lpath}")

        if right and right.filename and allowed_file(right.filename, ALLOWED_AUDIO):
            rname = f"{tag}_R_{secure_filename(right.filename)}"
            rpath = os.path.join(UPLOAD_AUDIO_DIR, rname)
            right.save(rpath)
            print(f"[app] Saved right audio: {rpath} ({os.path.getsize(rpath)} bytes)")
            rpath = normalize_audio_for_playback(rpath)
            print(f"[app] Final right audio: {rpath}")

        session_code = f"SESSION_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:4].upper()}"

        conn = get_db()
        cur = conn.execute("""
            INSERT INTO predictions(session_code, user_id, farm_name, location, camera_id,
                                    video_input, left_audio, right_audio, status)
            VALUES(?,?,?,?,?,?,?,?, 'pending')
        """, (session_code, session["user_id"], farm_name, location, camera_id, vpath, lpath, rpath))
        pid = cur.lastrowid
        conn.commit()
        conn.close()
        print(f"[app] Created monitoring session #{pid} ({session_code})")

        launch_pipeline(pid, vpath, lpath, rpath)

        flash(f"Monitoring session {session_code} initiated successfully.", "info")
        return redirect(url_for("result", pid=pid))

    return render_template("predict.html")



def launch_pipeline(pid, vpath, lpath, rpath):
    runner = os.path.join(app.root_path, "pipeline_runner.py")
    out_video = os.path.join(OUTPUT_DIR, f"out_{pid}.mp4")
    out_csv   = os.path.join(REPORT_DIR,  f"report_{pid}.csv")
    out_html  = os.path.join(REPORT_DIR,  f"dashboard_{pid}.html")
    log_path  = os.path.join(LOG_DIR, f"prediction_{pid}.log")

    cmd = [sys.executable, "-u", runner,     # -u → unbuffered
           "--prediction_id", str(pid),
           "--video",     vpath,
           "--out_video", out_video,
           "--out_csv",   out_csv,
           "--out_html",  out_html,
           "--show",      "1"]
    if lpath: cmd += ["--left",  lpath]
    if rpath: cmd += ["--right", rpath]

    print(f"[app] Launching pipeline #{pid}")
    print(f"[app] cmd: {' '.join(cmd)}")
    print(f"[app] log: {log_path}")

    # Use a file handle we keep alive for the child. Open with line buffering.
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # allow window to appear

    # NOTE: we do NOT use CREATE_NEW_CONSOLE here — the cv2 window
    # will still appear on top of the parent console. If you want a
    # separate console, switch to subprocess.CREATE_NEW_CONSOLE.
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        cwd=app.root_path,
    )

    threading.Thread(
        target=_watch_pipeline,
        args=(pid, proc, out_video, out_csv, out_html, log_file),
        daemon=True,
    ).start()


def _watch_pipeline(pid, proc, out_video, out_csv, out_html, log_file):
    proc.wait()
    try:
        log_file.flush()
        log_file.close()
    except Exception:
        pass

    # Give the subprocess time to flush files
    time.sleep(1.5)

    # ---- Read sidecar status.json ----
    status_json = os.path.join(LOG_DIR, f"prediction_{pid}.status.json")
    pipeline_ok = (proc.returncode == 0)
    info = {}
    if os.path.exists(status_json):
        try:
            import json
            with open(status_json, "r", encoding="utf-8") as f:
                info = json.load(f)
            pipeline_ok = (info.get("status") == "done")
            print(f"[watch] #{pid} status.json → {info}")
        except Exception as e:
            print(f"[watch] #{pid} could not read status.json: {e}")

    exists = os.path.exists(out_video) and os.path.getsize(out_video) > 0
    print(f"[watch] #{pid} rc={proc.returncode} pipeline_ok={pipeline_ok} "
          f"out_video_exists={exists} "
          f"size={os.path.getsize(out_video) if os.path.exists(out_video) else -1}")

    # ---- Choose final status ----
    final_status = "done" if (pipeline_ok and exists) else "failed"

    # ---- Count cows + alerts ----
    total_cows = 0
    total_alerts = 0
    if final_status == "done":
        conn = get_db()
        try:
            row = conn.execute("""
                SELECT COUNT(DISTINCT cow_id) AS cows,
                       (SELECT COUNT(*) FROM prediction_alerts WHERE prediction_id=?) AS alerts
                FROM prediction_results WHERE prediction_id=?
            """, (pid, pid)).fetchone()
            total_cows   = row["cows"]   if row else 0
            total_alerts = row["alerts"] if row else 0
        finally:
            conn.close()

    # ---- Write to DB ----
    conn = get_db()
    try:
        if final_status == "done":
            conn.execute("""
                UPDATE predictions SET
                    status        = 'done',
                    finished      = ?,
                    video_output  = ?,
                    csv_report    = ?,
                    html_report   = ?,
                    total_cows    = ?,
                    total_alerts  = ?,
                    alert_flag    = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(timespec="seconds"),
                out_video, out_csv, out_html,
                total_cows, total_alerts,
                1 if total_alerts > 0 else 0,
                pid,
            ))
        else:
            # Still save whatever files exist so the result page can show them
            conn.execute("""
                UPDATE predictions SET
                    status       = 'failed',
                    finished     = ?,
                    video_output = ?,
                    csv_report   = ?,
                    html_report  = ?
                WHERE id = ?
            """, (
                datetime.now().isoformat(timespec="seconds"),
                out_video if os.path.exists(out_video) else None,
                out_csv   if os.path.exists(out_csv)   else None,
                out_html  if os.path.exists(out_html)  else None,
                pid,
            ))
        conn.commit()

        # ---- Verify the write took effect ----
        check = conn.execute("""
            SELECT status, video_output, csv_report, html_report,
                   total_cows, total_alerts
            FROM predictions WHERE id=?
        """, (pid,)).fetchone()
        print(f"[watch] #{pid} DB after update: "
              f"status={check['status']} "
              f"video_output={check['video_output']} "
              f"cows={check['total_cows']} alerts={check['total_alerts']}")

        if final_status == "done" and not check["video_output"]:
            print(f"[watch] !!! WARNING: video_output was not persisted for #{pid}")

    except Exception as e:
        print(f"[watch] !!! DB UPDATE FAILED for #{pid}: {e}")
        import traceback; traceback.print_exc()
    finally:
        conn.close()

    print(f"[watch] #{pid} → {final_status} "
          f"(cows={total_cows}, alerts={total_alerts})")

@app.route("/debug/prediction/<int:pid>")
@login_required
def debug_prediction(pid):
    import json as _json
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                        (pid, session["user_id"])).fetchone()
    conn.close()
    if not pred:
        abort(404)

    files = {}
    for key in ("video_input", "left_audio", "right_audio",
                "video_output", "csv_report", "html_report"):
        p = pred[key]
        files[key] = {
            "path": p,
            "exists": bool(p) and os.path.exists(p),
            "size": os.path.getsize(p) if p and os.path.exists(p) else None,
        }

    status_json = os.path.join(LOG_DIR, f"prediction_{pid}.status.json")
    status_info = None
    if os.path.exists(status_json):
        try:
            with open(status_json, "r", encoding="utf-8") as f:
                status_info = _json.load(f)
        except Exception:
            pass

    log_path = os.path.join(LOG_DIR, f"prediction_{pid}.log")
    log_tail = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_tail = f.read()[-8000:]

    return jsonify(dict(
        prediction=dict(pred),
        files=files,
        status_json=status_info,
        log_tail=log_tail,
        python_executable=sys.executable,
        cwd=os.getcwd(),
    ))


# ===============================================================
@app.route("/result/<int:pid>")
@login_required
def result(pid):
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                        (pid, session["user_id"])).fetchone()
    if not pred:
        conn.close()
        abort(404)

    results      = conn.execute("SELECT * FROM prediction_results WHERE prediction_id=?", (pid,)).fetchall()
    alerts       = conn.execute("SELECT * FROM prediction_alerts WHERE prediction_id=? ORDER BY ts", (pid,)).fetchall()
    windows      = conn.execute("SELECT * FROM session_windows WHERE prediction_id=? ORDER BY window_index", (pid,)).fetchall()
    cow_profiles = conn.execute("SELECT * FROM cow_profiles WHERE prediction_id=? ORDER BY CAST(cow_id AS INT)", (pid,)).fetchall()
    events       = conn.execute("SELECT * FROM multimodal_events WHERE prediction_id=? ORDER BY timestamp", (pid,)).fetchall()
    anomalies    = conn.execute("SELECT * FROM anomalies WHERE prediction_id=? ORDER BY timestamp", (pid,)).fetchall()
    interactions = conn.execute("SELECT * FROM cow_interactions WHERE prediction_id=? ORDER BY start_time", (pid,)).fetchall()
    timelines    = conn.execute("SELECT * FROM behavior_timelines WHERE prediction_id=? ORDER BY cow_id, start_time", (pid,)).fetchall()
    conn.close()

    # Pre-compute file-existence flags for the template
    checks = {
        "input_exists":   bool(pred["video_input"])  and os.path.exists(pred["video_input"]),
        "output_exists":  bool(pred["video_output"]) and os.path.exists(pred["video_output"]),
        "left_exists":    bool(pred["left_audio"])   and os.path.exists(pred["left_audio"]),
        "right_exists":   bool(pred["right_audio"])  and os.path.exists(pred["right_audio"]),
        "csv_exists":     bool(pred["csv_report"])   and os.path.exists(pred["csv_report"]),
        "html_exists":    bool(pred["html_report"])  and os.path.exists(pred["html_report"]),
    }

    # Group timelines by cow_id
    cow_timelines_dict = {}
    for t in timelines:
        cow_timelines_dict.setdefault(t["cow_id"], []).append(dict(t))

    return render_template("result.html",
                           pred=pred, results=results, alerts=alerts,
                           windows=windows, cow_profiles=cow_profiles,
                           events=events, anomalies=anomalies,
                           interactions=interactions, cow_timelines=cow_timelines_dict,
                           checks=checks, basename=basename)


@app.route("/api/prediction_status/<int:pid>")
@login_required
def api_status(pid):
    conn = get_db()
    row = conn.execute("""SELECT status, total_cows, total_alerts, alert_flag, overall_risk, sync_status
                          FROM predictions WHERE id=? AND user_id=?""",
                       (pid, session["user_id"])).fetchone()
    conn.close()
    return jsonify(dict(row) if row else {})


@app.route("/api/session_details/<int:pid>")
@login_required
def api_session_details(pid):
    conn = get_db()
    pred = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?",
                        (pid, session["user_id"])).fetchone()
    if not pred:
        conn.close()
        return jsonify({"error": "Session not found"}), 404

    windows      = [dict(r) for r in conn.execute("SELECT * FROM session_windows WHERE prediction_id=?", (pid,)).fetchall()]
    cow_profiles = [dict(r) for r in conn.execute("SELECT * FROM cow_profiles WHERE prediction_id=?", (pid,)).fetchall()]
    events       = [dict(r) for r in conn.execute("SELECT * FROM multimodal_events WHERE prediction_id=?", (pid,)).fetchall()]
    anomalies    = [dict(r) for r in conn.execute("SELECT * FROM anomalies WHERE prediction_id=?", (pid,)).fetchall()]
    interactions = [dict(r) for r in conn.execute("SELECT * FROM cow_interactions WHERE prediction_id=?", (pid,)).fetchall()]
    conn.close()

    return jsonify({
        "session": dict(pred),
        "windows": windows,
        "cow_profiles": cow_profiles,
        "multimodal_events": events,
        "anomalies": anomalies,
        "interactions": interactions
    })


# ===============================================================
# New Multi-Page Platform Routes
# ===============================================================
@app.route("/sessions")
@login_required
def sessions():
    conn = get_db()
    rows = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC",
                        (session["user_id"],)).fetchall()
    conn.close()
    return render_template("sessions.html", predictions=rows, basename=basename)


@app.route("/cows")
@login_required
def cows():
    conn = get_db()
    uid = session["user_id"]
    cow_profiles = conn.execute("""
        SELECT cp.*, p.session_code, p.created as session_created, p.farm_name, p.location
        FROM cow_profiles cp
        JOIN predictions p ON cp.prediction_id = p.id
        WHERE p.user_id = ?
        ORDER BY CAST(cp.cow_id AS INT), cp.prediction_id DESC
    """, (uid,)).fetchall()
    conn.close()
    return render_template("herd.html", cow_profiles=cow_profiles)


@app.route("/cow/<int:pid>/<cow_id>")
@login_required
def cow_detail(pid, cow_id):
    conn = get_db()
    uid = session["user_id"]
    pred = conn.execute("SELECT * FROM predictions WHERE id=? AND user_id=?", (pid, uid)).fetchone()
    if not pred:
        conn.close()
        abort(404)

    profile = conn.execute("SELECT * FROM cow_profiles WHERE prediction_id=? AND cow_id=?", (pid, cow_id)).fetchone()
    timelines = conn.execute("SELECT * FROM behavior_timelines WHERE prediction_id=? AND cow_id=? ORDER BY start_time", (pid, cow_id)).fetchall()
    anomalies = conn.execute("SELECT * FROM anomalies WHERE prediction_id=? AND cow_id=? ORDER BY timestamp", (pid, cow_id)).fetchall()
    events = conn.execute("SELECT * FROM multimodal_events WHERE prediction_id=? AND cow_id=? ORDER BY timestamp", (pid, cow_id)).fetchall()
    baseline = conn.execute("SELECT * FROM cow_baselines WHERE user_id=? AND cow_id=?", (uid, cow_id)).fetchone()
    interactions = conn.execute("SELECT * FROM cow_interactions WHERE prediction_id=? AND (cow_a=? OR cow_b=?)", (pid, cow_id, cow_id)).fetchall()
    conn.close()

    return render_template("cow_profile.html",
                           pred=pred,
                           profile=profile,
                           timelines=timelines,
                           anomalies=anomalies,
                           events=events,
                           baseline=baseline,
                           interactions=interactions,
                           cow_id=cow_id)


@app.route("/alerts")
@login_required
def alerts_page():
    conn = get_db()
    uid = session["user_id"]
    alerts = conn.execute("""
        SELECT pa.*, p.session_code, p.farm_name, p.location
        FROM prediction_alerts pa
        JOIN predictions p ON pa.prediction_id = p.id
        WHERE p.user_id = ?
        ORDER BY pa.ts DESC
    """, (uid,)).fetchall()
    conn.close()
    return render_template("alerts.html", alerts=alerts)


@app.route("/api/alert/<int:aid>/acknowledge", methods=["POST"])
@login_required
def acknowledge_alert(aid):
    conn = get_db()
    uid = session["user_id"]
    conn.execute("""
        UPDATE prediction_alerts 
        SET acknowledged = 1 
        WHERE id = ? AND prediction_id IN (SELECT id FROM predictions WHERE user_id = ?)
    """, (aid, uid))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/analytics")
@login_required
def analytics_page():
    conn = get_db()
    uid = session["user_id"]
    predictions = conn.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY id ASC", (uid,)).fetchall()
    interactions = conn.execute("""
        SELECT ci.*, p.session_code 
        FROM cow_interactions ci 
        JOIN predictions p ON ci.prediction_id = p.id 
        WHERE p.user_id = ?
    """, (uid,)).fetchall()
    conn.close()
    return render_template("analytics.html", predictions=predictions, interactions=interactions)


@app.route("/reports")
@login_required
def reports_page():
    conn = get_db()
    uid = session["user_id"]
    rows = conn.execute("""
        SELECT id, session_code, farm_name, location, created, csv_report, html_report, overall_risk, total_cows, total_alerts
        FROM predictions 
        WHERE user_id=? AND status='done' 
        ORDER BY id DESC
    """, (uid,)).fetchall()
    conn.close()
    return render_template("reports.html", predictions=rows, basename=basename)


@app.route("/settings")
@login_required
def settings_page():
    return render_template("settings.html")


@app.route("/api/overview_stats")
@login_required
def api_overview_stats():
    conn = get_db()
    uid = session["user_id"]

    predictions = conn.execute("SELECT id, session_code, overall_risk, created FROM predictions WHERE user_id=? ORDER BY id ASC", (uid,)).fetchall()
    
    risk_trend = [{"session_code": p["session_code"], "risk": p["overall_risk"] or 0, "date": p["created"]} for p in predictions]
    
    # Calculate behavior sums
    cow_profiles = conn.execute("""
        SELECT cp.* FROM cow_profiles cp
        JOIN predictions p ON cp.prediction_id = p.id
        WHERE p.user_id = ?
    """, (uid,)).fetchall()

    standing_sum = sum(cp["standing_pct"] or 0 for cp in cow_profiles)
    lying_sum    = sum(cp["lying_pct"] or 0 for cp in cow_profiles)
    eating_sum   = sum(cp["eating_pct"] or 0 for cp in cow_profiles)
    drinking_sum = sum(cp["drinking_pct"] or 0 for cp in cow_profiles)
    agg_sum      = sum(cp["aggressive_pct"] or 0 for cp in cow_profiles)
    total_profiles = max(1, len(cow_profiles))

    behavior_avg = {
        "standing": round(standing_sum / total_profiles, 1),
        "lying": round(lying_sum / total_profiles, 1),
        "eating": round(eating_sum / total_profiles, 1),
        "drinking": round(drinking_sum / total_profiles, 1),
        "aggressive": round(agg_sum / total_profiles, 1)
    }

    conn.close()
    return jsonify({
        "risk_trend": risk_trend,
        "behavior_avg": behavior_avg
    })


# ===============================================================
if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
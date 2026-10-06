import os, sys, argparse, sqlite3, traceback, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run as pipeline_run
from config import DB_PATH, LOG_DIR


def write_status(prediction_id, status, **extra):
    """Write a small sidecar JSON so the parent process knows what happened."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"prediction_{prediction_id}.status.json")
        payload = dict(status=status, ts=time.time(), **extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print("Failed to write status JSON:", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_id", type=int, required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--left",  default=None)
    ap.add_argument("--right", default=None)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--show", type=int, default=1)
    args = ap.parse_args()

    print(f"=== pipeline_runner starting for prediction #{args.prediction_id} ===", flush=True)
    print(f"cwd: {os.getcwd()}", flush=True)
    print(f"video: {args.video} exists={os.path.exists(args.video)}", flush=True)
    print(f"out_video target: {args.out_video}", flush=True)

    write_status(args.prediction_id, "running")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    telegram_cb = None
    try:
        from telegram_alert import send_alert
        telegram_cb = send_alert
        print("Telegram callback loaded", flush=True)
    except Exception as e:
        print("Telegram disabled:", e, flush=True)

    try:
        result = pipeline_run(
            video_path=args.video,
            left_audio=args.left,
            right_audio=args.right,
            out_video=args.out_video,
            out_csv=args.out_csv,
            out_html=args.out_html,
            prediction_id=args.prediction_id,
            db_conn=conn,
            telegram_callback=telegram_cb,
            show_window=bool(args.show),
        )
        print("PIPELINE_RESULT:", result, flush=True)

        # give OS a moment to flush
        time.sleep(0.5)

        sizes = {}
        for key, path in (("video", args.out_video),
                          ("csv",   args.out_csv),
                          ("html",  args.out_html)):
            sizes[key] = os.path.getsize(path) if os.path.exists(path) else -1
        print(f"OUTPUT SIZES: {sizes}", flush=True)

        write_status(args.prediction_id, "done", **sizes, result=result)
        # NOTE: do NOT set status='done' here — the watchdog in app.py
        # will do it after verifying files are on disk.

    except Exception as e:
        print(f"!!! PIPELINE CRASHED: {e}", flush=True)
        traceback.print_exc()
        write_status(args.prediction_id, "failed", error=str(e))
        # Let the watchdog mark status='failed' — do nothing here.
        raise
    finally:
        conn.close()
        print("=== pipeline_runner finished ===", flush=True)

if __name__ == "__main__":
    main()
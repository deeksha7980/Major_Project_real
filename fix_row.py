# fix_row.py
import sqlite3, os
from config import DB_PATH, OUTPUT_DIR, REPORT_DIR

pid = 2  # ← change to the prediction id you want to fix

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

out_video = os.path.join(OUTPUT_DIR, f"out_{pid}.mp4")
out_csv   = os.path.join(REPORT_DIR,  f"report_{pid}.csv")
out_html  = os.path.join(REPORT_DIR,  f"dashboard_{pid}.html")

print("video exists:", os.path.exists(out_video), out_video)
print("csv   exists:", os.path.exists(out_csv),   out_csv)
print("html  exists:", os.path.exists(out_html),  out_html)

conn.execute("""
    UPDATE predictions
    SET status='done',
        video_output=?,
        csv_report=?,
        html_report=?
    WHERE id=?
""", (out_video, out_csv, out_html, pid))
conn.commit()

row = conn.execute("SELECT id, status, video_output FROM predictions WHERE id=?", (pid,)).fetchone()
print("after:", dict(row))

conn.close()
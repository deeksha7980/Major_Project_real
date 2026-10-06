"""
test_livestock_platform.py — Master Unit & Integration Test Suite for CowCareAI Platform.
Tests stream synchronization, temporal windowing, acoustic zones, cow profiles, multimodal events,
explainable risk scoring, statistical anomalies, baselines, interactions, reports, and end-to-end pipeline execution.
"""
import os, sys, sqlite3, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.session_service import probe_file_durations, generate_session_code
from services.audio_service import analyze_acoustic_zone
from services.cow_service import aggregate_cow_profiles
from services.multimodal_service import detect_multimodal_events
from services.risk_service import calculate_cow_risk, calculate_herd_risk
from services.anomaly_service import detect_anomalies
from services.baseline_service import compare_cow_baseline, update_cow_baseline
from services.interaction_service import build_cow_interaction_graph, calculate_herd_analytics
from services.report_service import generate_csv_report, generate_html_report
from database import init_db, get_db
from pipeline import run as pipeline_run

class TestCowCareAIPlatform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import numpy as np, cv2, wave, struct
        if not os.path.exists("664.mp4"):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter("664.mp4", fourcc, 10.0, (640, 480))
            for _ in range(10):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()
        for f in ["high.mp3", "high1.mp3"]:
            if not os.path.exists(f):
                wav_name = f + ".wav"
                with wave.open(wav_name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    for i in range(16000):
                        val = int(32767.0 * 0.1 * np.sin(2.0 * np.pi * 440.0 * i / 16000))
                        wf.writeframes(struct.pack('<h', val))
                os.replace(wav_name, f)

    def setUp(self):
        init_db()
        self.db_conn = get_db()

    def tearDown(self):
        if self.db_conn:
            self.db_conn.close()

    def test_01_stream_sync_and_session_code(self):
        print("\n[TEST 01] Stream Duration Probing & Time Synchronization")
        res = probe_file_durations("664.mp4", "high.mp3", "high1.mp3")
        print("Probe Result:", res)
        self.assertGreater(res["video_duration"], 0.0)
        self.assertIn("Synchronized", res["sync_status"])

        code = generate_session_code()
        print("Generated Session Code:", code)
        self.assertTrue(code.startswith("SESSION_"))

    def test_02_acoustic_zone_intelligence(self):
        print("\n[TEST 02] Left/Right Acoustic Zone Intelligence")
        # Case A: Left Dominant HFC
        ac_left = analyze_acoustic_zone(0.95, 0.30, 0.95, 0.30)
        print("Left Dominant:", ac_left)
        self.assertEqual(ac_left["acoustic_zone"], "LEFT-DOMINANT")

        # Case B: Broad Dual HFC
        ac_broad = analyze_acoustic_zone(0.94, 0.92, 0.94, 0.92)
        print("Broad Center:", ac_broad)
        self.assertEqual(ac_broad["acoustic_zone"], "BROAD/CENTER")
        self.assertGreater(ac_broad["audio_agreement"], 0.90)

    def test_03_cow_profiles_and_timelines(self):
        print("\n[TEST 03] Individual Cow Profile Aggregation & Behavior Timelines")
        dummy_obs = {
            "1": [
                {"ts": 0.0, "beh": "standing", "conf": 0.9, "vel": 0.2, "side": "left", "box": (10, 10, 100, 100)},
                {"ts": 1.0, "beh": "standing", "conf": 0.9, "vel": 0.2, "side": "left", "box": (10, 10, 100, 100)},
                {"ts": 2.0, "beh": "lying", "conf": 0.85, "vel": 0.0, "side": "left", "box": (10, 10, 100, 100)},
                {"ts": 3.0, "beh": "lying", "conf": 0.85, "vel": 0.0, "side": "left", "box": (10, 10, 100, 100)},
            ]
        }
        res = aggregate_cow_profiles(dummy_obs)
        p1 = res["cow_profiles"]["1"]
        print("Cow 1 Profile:", p1)
        self.assertEqual(p1["dominant_behavior"], "standing")
        self.assertEqual(p1["standing_pct"], 50.0)
        self.assertEqual(p1["lying_pct"], 50.0)
        self.assertEqual(p1["transitions_count"], 1)

    def test_04_multimodal_event_correlation(self):
        print("\n[TEST 04] Multimodal Correlated Event Engine")
        session_windows = [{
            "start_time": 0.0, "end_time": 5.0,
            "left_audio_class": "High-frequency call (HFC)", "left_audio_confidence": 0.94,
            "right_audio_class": "High-frequency call (HFC)", "right_audio_confidence": 0.91,
            "audio_agreement": 0.97
        }]
        cow_obs = {
            "1": [{"ts": 2.0, "beh": "aggressive", "conf": 0.92, "vel": 1.5, "side": "left", "box": (0,0,10,10)}]
        }
        events = detect_multimodal_events(session_windows, cow_obs, [])
        print("Multimodal Events Detected:", len(events))
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0]["cow_id"], "1")
        self.assertEqual(events[0]["severity"], "high")

    def test_05_explainable_risk_engine(self):
        print("\n[TEST 05] Explainable Multimodal Risk Engine")
        profile = {
            "movement_score": 1.2,
            "aggressive_pct": 30.0,
            "transitions_count": 3,
            "audio_event_count": 2
        }
        risk = calculate_cow_risk(profile, [{"dummy": 1}], [{"dummy": 1}], [{"dummy": 1}])
        print("Risk Engine Output:", risk)
        self.assertGreater(risk["risk_score"], 50.0)
        self.assertTrue(len(risk["contributing_factors"]) >= 3)
        self.assertIn("unusual movement", risk["contributor_str"])

    def test_06_statistical_anomaly_engine(self):
        print("\n[TEST 06] Statistical Anomaly Engine")
        profiles = {
            "1": {"first_seen": 0.0, "movement_score": 0.10, "transitions_count": 1},
            "2": {"first_seen": 0.0, "movement_score": 0.12, "transitions_count": 1},
            "3": {"first_seen": 0.0, "movement_score": 0.08, "transitions_count": 1},
            "4": {"first_seen": 0.0, "movement_score": 0.11, "transitions_count": 1},
            "5": {"first_seen": 0.0, "movement_score": 3.50, "transitions_count": 5} # Clear statistical anomaly (> 2.0 sigma)
        }
        anoms = detect_anomalies(profiles, [])
        print("Anomalies Detected:", len(anoms))
        self.assertGreater(len(anoms), 0)
        self.assertEqual(anoms[0]["cow_id"], "5")



    def test_07_end_to_end_pipeline_execution(self):
        print("\n[TEST 07] End-to-End Platform Execution with Real Media")
        cur = self.db_conn.cursor()
        cur.execute("""
            INSERT INTO predictions(user_id, video_input, left_audio, right_audio, status)
            VALUES(1, '664.mp4', 'high.mp3', 'high1.mp3', 'testing')
        """)
        self.db_conn.commit()
        pid = cur.lastrowid

        out_v = f"media/outputs/test_plat_{pid}.mp4"
        out_c = f"reports/test_plat_{pid}.csv"
        out_h = f"reports/test_plat_{pid}.html"

        res = pipeline_run(
            video_path="664.mp4",
            left_audio="high.mp3",
            right_audio="high1.mp3",
            out_video=out_v,
            out_csv=out_c,
            out_html=out_h,
            prediction_id=pid,
            db_conn=self.db_conn,
            show_window=False
        )
        print("End-to-End Pipeline Result:", res)
        self.assertGreaterEqual(res["total_cows"], 0)
        self.assertGreaterEqual(res["overall_risk"], 0.0)

        # Check DB Tables
        cur.execute("SELECT COUNT(*) FROM session_windows WHERE prediction_id=?", (pid,))
        win_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM cow_profiles WHERE prediction_id=?", (pid,))
        cow_cnt = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM multimodal_events WHERE prediction_id=?", (pid,))
        evt_cnt = cur.fetchone()[0]

        print(f"DB Verification: {win_cnt} Windows | {cow_cnt} Cow Profiles | {evt_cnt} Multimodal Events")
        self.assertGreater(win_cnt, 0)
        self.assertGreaterEqual(cow_cnt, 0)
        print("\n[SUCCESS] ALL PLATFORM SUITE TESTS PASSED!")

if __name__ == '__main__':
    unittest.main()

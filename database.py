import sqlite3
from config import DB_PATH

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            email     TEXT UNIQUE NOT NULL,
            password  TEXT NOT NULL,
            created   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_code    TEXT UNIQUE,
            user_id         INTEGER NOT NULL,
            farm_name       TEXT DEFAULT 'Main Farm',
            location        TEXT DEFAULT 'Barn A',
            camera_id       TEXT DEFAULT 'Cam 1',
            video_input     TEXT,
            left_audio      TEXT,
            right_audio     TEXT,
            video_output    TEXT,
            csv_report      TEXT,
            html_report     TEXT,
            duration        REAL DEFAULT 0.0,
            left_duration   REAL DEFAULT 0.0,
            right_duration  REAL DEFAULT 0.0,
            sync_status     TEXT DEFAULT 'Synchronized',
            total_cows      INTEGER DEFAULT 0,
            total_events    INTEGER DEFAULT 0,
            total_anomalies INTEGER DEFAULT 0,
            total_alerts    INTEGER DEFAULT 0,
            overall_risk    REAL DEFAULT 0.0,
            alert_flag      INTEGER DEFAULT 0,
            status          TEXT DEFAULT 'pending',
            created         TEXT DEFAULT CURRENT_TIMESTAMP,
            finished        TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Safe column migrations for existing databases
    for col, col_def in [
        ("session_code", "TEXT"),
        ("farm_name", "TEXT DEFAULT 'Main Farm'"),
        ("location", "TEXT DEFAULT 'Barn A'"),
        ("camera_id", "TEXT DEFAULT 'Cam 1'"),
        ("duration", "REAL DEFAULT 0.0"),
        ("left_duration", "REAL DEFAULT 0.0"),
        ("right_duration", "REAL DEFAULT 0.0"),
        ("sync_status", "TEXT DEFAULT 'Synchronized'"),
        ("total_events", "INTEGER DEFAULT 0"),
        ("total_anomalies", "INTEGER DEFAULT 0"),
        ("overall_risk", "REAL DEFAULT 0.0"),
    ]:
        try:
            cur.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_def}")
        except Exception:
            pass # Column already exists

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            cow_id        TEXT,
            behaviour     TEXT,
            confidence    REAL,
            health        TEXT,
            history       TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ts            REAL,
            alert_type    TEXT,
            cow_id        TEXT,
            severity      TEXT,
            message       TEXT,
            image_frame   TEXT,
            risk_score    REAL DEFAULT 0.0,
            contributing_factors TEXT DEFAULT '',
            sent_telegram INTEGER DEFAULT 0,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    for col, col_def in [
        ("risk_score", "REAL DEFAULT 0.0"),
        ("contributing_factors", "TEXT DEFAULT ''"),
        ("acknowledged", "INTEGER DEFAULT 0"),
    ]:
        try:
            cur.execute(f"ALTER TABLE prediction_alerts ADD COLUMN {col} {col_def}")
        except Exception:
            pass

    # ---------- Phase 3: Temporal Session Windows ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_windows (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id         INTEGER NOT NULL,
            window_index          INTEGER NOT NULL,
            start_time            REAL NOT NULL,
            end_time              REAL NOT NULL,
            cow_count             INTEGER DEFAULT 0,
            left_audio_class      TEXT,
            left_audio_confidence REAL,
            right_audio_class     TEXT,
            right_audio_confidence REAL,
            audio_agreement       REAL,
            acoustic_zone         TEXT,
            risk_score            REAL,
            window_data           TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    # ---------- Phase 4: Individual Cow Profiles ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cow_profiles (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id         INTEGER NOT NULL,
            cow_id                TEXT NOT NULL,
            first_seen            REAL,
            last_seen             REAL,
            visible_duration      REAL,
            dominant_behavior     TEXT,
            standing_pct          REAL DEFAULT 0.0,
            lying_pct             REAL DEFAULT 0.0,
            eating_pct            REAL DEFAULT 0.0,
            drinking_pct          REAL DEFAULT 0.0,
            aggressive_pct        REAL DEFAULT 0.0,
            rumination_pct        REAL DEFAULT 0.0,
            movement_score        REAL DEFAULT 0.0,
            transitions_count     INTEGER DEFAULT 0,
            interaction_count     INTEGER DEFAULT 0,
            anomaly_count         INTEGER DEFAULT 0,
            audio_event_count     INTEGER DEFAULT 0,
            risk_score            REAL DEFAULT 0.0,
            risk_status           TEXT DEFAULT 'NORMAL',
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    # ---------- Phase 5: Behavior Timelines ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS behavior_timelines (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            cow_id        TEXT NOT NULL,
            start_time    REAL NOT NULL,
            end_time      REAL NOT NULL,
            behavior      TEXT NOT NULL,
            confidence    REAL DEFAULT 0.0,
            movement      REAL DEFAULT 0.0,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    # ---------- Phase 7: Multimodal Events ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS multimodal_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id    INTEGER NOT NULL,
            timestamp        REAL NOT NULL,
            cow_id           TEXT,
            event_type       TEXT NOT NULL,
            visual_evidence  TEXT,
            audio_evidence   TEXT,
            left_confidence  REAL,
            right_confidence REAL,
            audio_agreement  REAL,
            event_confidence REAL,
            severity         TEXT DEFAULT 'info',
            description      TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    # ---------- Phase 11: Statistical Anomalies ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            timestamp     REAL NOT NULL,
            cow_id        TEXT NOT NULL,
            feature       TEXT NOT NULL,
            baseline_val  REAL,
            current_val   REAL,
            deviation_pct REAL,
            severity      TEXT DEFAULT 'medium',
            explanation   TEXT,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    # ---------- Phase 13: Spatial Cow Interaction Graph ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cow_interactions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id    INTEGER NOT NULL,
            cow_a            TEXT NOT NULL,
            cow_b            TEXT NOT NULL,
            start_time       REAL NOT NULL,
            end_time         REAL NOT NULL,
            duration         REAL NOT NULL,
            proximity_score  REAL DEFAULT 0.0,
            interaction_type TEXT DEFAULT 'Proximity Event',
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        )
    """)

    # ---------- Phase 10: Individual Cow Baselines ----------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cow_baselines (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            cow_id         TEXT NOT NULL,
            standing_mean  REAL DEFAULT 0.35,
            lying_mean     REAL DEFAULT 0.30,
            eating_mean    REAL DEFAULT 0.20,
            movement_mean  REAL DEFAULT 0.30,
            sample_count   INTEGER DEFAULT 1,
            updated_at     TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
"""
baseline_service.py — Historical Cow Baseline Comparison Service.
Compares individual cow observations against stored historical baselines.
"""

def compare_cow_baseline(db_conn, user_id, cow_id, current_profile):
    """
    Compares current cow session behavior percentages against stored historical baselines.
    Never fabricates historical data.
    """
    if not db_conn or not user_id:
        return {
            "status": "Unavailable",
            "reason": "Insufficient historical observations.",
            "has_baseline": False
        }

    try:
        row = db_conn.execute("""
            SELECT * FROM cow_baselines WHERE user_id=? AND cow_id=?
        """, (user_id, str(cow_id))).fetchone()

        if not row or row['sample_count'] < 2:
            return {
                "status": "Unavailable",
                "reason": "Insufficient historical observations (sample count < 2).",
                "has_baseline": False
            }

        base_standing = row['standing_mean'] * 100.0
        cur_standing = current_profile.get('standing_pct', 0.0)

        diff_standing = cur_standing - base_standing
        significant_diff = abs(diff_standing) >= 25.0

        return {
            "status": "Behavioral deviation detected" if significant_diff else "Consistent with baseline",
            "has_baseline": True,
            "standing_baseline_pct": round(base_standing, 1),
            "standing_current_pct": round(cur_standing, 1),
            "deviation_pct": round(diff_standing, 1),
            "sample_count": row['sample_count']
        }
    except Exception as e:
        return {
            "status": "Unavailable",
            "reason": f"Baseline lookup error: {e}",
            "has_baseline": False
        }

def update_cow_baseline(db_conn, user_id, cow_id, current_profile):
    """Updates historical baseline running averages for a cow ID."""
    if not db_conn or not user_id or not cow_id:
        return

    try:
        row = db_conn.execute("""
            SELECT * FROM cow_baselines WHERE user_id=? AND cow_id=?
        """, (user_id, str(cow_id))).fetchone()

        c_stand = current_profile.get('standing_pct', 0.0) / 100.0
        c_lying = current_profile.get('lying_pct', 0.0) / 100.0
        c_eating = current_profile.get('eating_pct', 0.0) / 100.0
        c_mov = current_profile.get('movement_score', 0.0)

        if not row:
            db_conn.execute("""
                INSERT INTO cow_baselines(user_id, cow_id, standing_mean, lying_mean, eating_mean, movement_mean, sample_count)
                VALUES(?,?,?,?,?,?,1)
            """, (user_id, str(cow_id), c_stand, c_lying, c_eating, c_mov))
        else:
            n = row['sample_count']
            new_n = n + 1
            new_stand = (row['standing_mean'] * n + c_stand) / new_n
            new_lying = (row['lying_mean'] * n + c_lying) / new_n
            new_eat   = (row['eating_mean'] * n + c_eating) / new_n
            new_mov   = (row['movement_mean'] * n + c_mov) / new_n

            db_conn.execute("""
                UPDATE cow_baselines SET
                    standing_mean = ?, lying_mean = ?, eating_mean = ?, movement_mean = ?,
                    sample_count = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id=? AND cow_id=?
            """, (new_stand, new_lying, new_eat, new_mov, new_n, user_id, str(cow_id)))

        db_conn.commit()
    except Exception as e:
        print("Baseline update warning:", e)

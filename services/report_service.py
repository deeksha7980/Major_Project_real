"""
report_service.py — Upgraded Multimodal Report Generator.
Generates comprehensive temporal CSV exports and rich standalone HTML intelligence reports.
"""
import csv, os

def generate_csv_report(out_csv_path, session_info, session_windows, cow_profiles_dict, multimodal_events, anomalies):
    """Generates an upgraded temporal and individual-cow CSV report."""
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["=== COWCAREAI MULTIMODAL LIVESTOCK INTELLIGENCE REPORT ==="])
        writer.writerow(["Session_Code", session_info.get("session_code", "")])
        writer.writerow(["Farm_Name", session_info.get("farm_name", "Main Farm")])
        writer.writerow(["Location", session_info.get("location", "Barn A")])
        writer.writerow(["Duration_Seconds", session_info.get("duration", 0.0)])
        writer.writerow(["Total_Cows", len(cow_profiles_dict)])
        writer.writerow(["Overall_Risk_Score", session_info.get("overall_risk", 0.0)])
        writer.writerow([])

        # Table 1: Individual Cow Profiles
        writer.writerow(["--- INDIVIDUAL COW PROFILES ---"])
        writer.writerow(["Cow_ID", "Dominant_Behavior", "Standing_Pct", "Lying_Pct", "Eating_Pct", "Movement_Score", "Transitions", "Risk_Score", "Risk_Status", "Risk_Contributors"])
        for cid, p in cow_profiles_dict.items():
            writer.writerow([
                f"Cow {cid}", p['dominant_behavior'], p['standing_pct'], p['lying_pct'],
                p['eating_pct'], p['movement_score'], p['transitions_count'],
                p.get('risk_score', 0.0), p.get('risk_status', 'NORMAL'), p.get('contributor_str', '')
            ])
        writer.writerow([])

        # Table 2: Temporal Windows
        writer.writerow(["--- TEMPORAL SESSION WINDOWS ---"])
        writer.writerow(["Window_Idx", "Start_Time", "End_Time", "Cow_Count", "Left_Audio_Class", "Left_Conf", "Right_Audio_Class", "Right_Conf", "Audio_Agreement", "Acoustic_Zone", "Window_Risk"])
        for w in session_windows:
            writer.writerow([
                w['window_index'], w['start_time'], w['end_time'], w['cow_count'],
                w.get('left_audio_class', 'N/A'), w.get('left_audio_confidence', 0.0),
                w.get('right_audio_class', 'N/A'), w.get('right_audio_confidence', 0.0),
                w.get('audio_agreement', 1.0), w.get('acoustic_zone', 'N/A'), w.get('risk_score', 0.0)
            ])
        writer.writerow([])

        # Table 3: Multimodal Events
        writer.writerow(["--- MULTIMODAL EVENTS ---"])
        writer.writerow(["Timestamp", "Cow_ID", "Event_Type", "Visual_Evidence", "Audio_Evidence", "Event_Confidence", "Severity"])
        for e in multimodal_events:
            writer.writerow([
                e['timestamp'], f"Cow {e['cow_id']}", e['event_type'],
                e['visual_evidence'], e['audio_evidence'], e['event_confidence'], e['severity']
            ])

def generate_html_report(out_html_path, session_info, session_windows, cow_profiles_dict, multimodal_events, anomalies, herd_analytics):
    """Generates a rich, responsive HTML livestock intelligence report."""
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CowCareAI — Session Intelligence Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px; border: 1px solid #334155; }}
        h1, h2, h3 {{ color: #38bdf8; margin-top: 0; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
        .stat-box {{ background: #0f172a; padding: 16px; border-radius: 8px; text-align: center; border: 1px solid #334155; }}
        .stat-val {{ font-size: 28px; font-weight: bold; color: #38bdf8; }}
        .stat-lbl {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #38bdf8; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; display: inline-block; }}
        .badge-normal {{ background: #166534; color: #4ade80; }}
        .badge-attention {{ background: #854d0e; color: #facc15; }}
        .badge-elevated {{ background: #9a3412; color: #fb923c; }}
        .badge-high {{ background: #991b1b; color: #f87171; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🐄 CowCareAI — Livestock Intelligence Report</h1>
        <p><strong>Session ID:</strong> {session_info.get("session_code", "N/A")} | <strong>Farm:</strong> {session_info.get("farm_name", "Main Farm")} | <strong>Location:</strong> {session_info.get("location", "Barn A")}</p>
        <p><strong>Duration:</strong> {session_info.get("duration", 0.0):.1f} seconds | <strong>Synchronization:</strong> {session_info.get("sync_status", "Synchronized")}</p>
    </div>

    <div class="grid">
        <div class="stat-box"><div class="stat-val">{len(cow_profiles_dict)}</div><div class="stat-lbl">Total Cows</div></div>
        <div class="stat-box"><div class="stat-val">{len(multimodal_events)}</div><div class="stat-lbl">Multimodal Events</div></div>
        <div class="stat-box"><div class="stat-val">{len(anomalies)}</div><div class="stat-lbl">Anomalies</div></div>
        <div class="stat-box"><div class="stat-val" style="color:{'#f87171' if session_info.get('overall_risk', 0.0) >= 60 else '#38bdf8'}">{session_info.get("overall_risk", 0.0):.0f} / 100</div><div class="stat-lbl">Herd Risk Score</div></div>
    </div>

    <div class="card">
        <h2>Individual Cow Profiles</h2>
        <table>
            <thead>
                <tr>
                    <th>Cow ID</th>
                    <th>Dominant Behavior</th>
                    <th>Standing %</th>
                    <th>Lying %</th>
                    <th>Movement</th>
                    <th>Transitions</th>
                    <th>Risk Score</th>
                    <th>Status</th>
                    <th>Risk Contributors</th>
                </tr>
            </thead>
            <tbody>"""
    
    for cid, p in cow_profiles_dict.items():
        st = p.get('risk_status', 'NORMAL')
        b_cls = "badge-normal" if st == "NORMAL" else ("badge-attention" if st == "LOW ATTENTION" else ("badge-elevated" if st == "ELEVATED" else "badge-high"))
        html += f"""
                <tr>
                    <td><strong>Cow {cid}</strong></td>
                    <td>{p['dominant_behavior'].capitalize()}</td>
                    <td>{p['standing_pct']}%</td>
                    <td>{p['lying_pct']}%</td>
                    <td>{p['movement_score']:.2f}</td>
                    <td>{p['transitions_count']}</td>
                    <td><strong>{p.get('risk_score', 0.0):.0f}</strong></td>
                    <td><span class="badge {b_cls}">{st}</span></td>
                    <td>{p.get('contributor_str', 'Normal')}</td>
                </tr>"""

    html += """
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Multimodal Events</h2>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Cow ID</th>
                    <th>Visual Evidence</th>
                    <th>Audio Evidence</th>
                    <th>Confidence</th>
                    <th>Severity</th>
                </tr>
            </thead>
            <tbody>"""

    for e in multimodal_events:
        html += f"""
                <tr>
                    <td>{e['timestamp']:.1f}s</td>
                    <td>Cow {e['cow_id']}</td>
                    <td>{e['visual_evidence']}</td>
                    <td>{e['audio_evidence']}</td>
                    <td>{e['event_confidence']*100:.0f}%</td>
                    <td><span class="badge {'badge-high' if e['severity']=='high' else 'badge-attention'}">{e['severity'].upper()}</span></td>
                </tr>"""

    html += """
            </tbody>
        </table>
    </div>
</body>
</html>"""

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)

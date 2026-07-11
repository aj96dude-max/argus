"""
ARGUS Agentic Engine - Dashboard Flask App
Serves the ARGUS Command Dashboard and orchestrates the full agent pipeline:
  Listener Agent -> OSINT NLP Agent -> Threat Synthesizer -> Dashboard UI
"""

from __future__ import annotations
import sys
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask_cors import CORS

# Agentic pipeline imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml_engine"))

from agents.listener import SensorHubListener
from agents.osint_nlp import OSINTNLPAgent
from agents.synthesizer import ThreatSynthesizer

logger = logging.getLogger("ARGUS.Dashboard")

# ?????????????????????????????????????????????????????????????
# FLASK APP
# ?????????????????????????????????????????????????????????????

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ?????????????????????????????????????????????????????????????
# GLOBAL STATE
# ?????????????????????????????????????????????????????????????

_enriched_alerts: List[Dict] = []
_sse_clients: List = []  # SSE event queues for browser push
_agent_stats = {
    "total_processed": 0,
    "nlp_enriched": 0,
    "critical_alerts": 0,
    "high_alerts": 0,
    "system_start": datetime.now(timezone.utc).isoformat(),
}

# Initialize agents (lazy - NLP model loaded in background)
nlp_agent = OSINTNLPAgent()
synthesizer = ThreatSynthesizer()


# ?????????????????????????????????????????????????????????????
# AGENT PIPELINE
# ?????????????????????????????????????????????????????????????

def process_alert(alert_payload: Dict):
    """
    Full agentic pipeline:
    1. Retrieve/simulate subject's digital footprint
    2. Run OSINT NLP analysis (Module D)
    3. Synthesize into Predictive Threat Assessment Report
    4. Store enriched alert and push to all SSE browser clients
    5. PATCH Sensor Hub with enrichment data
    """
    subject_id = alert_payload.get("subject_id", "UNKNOWN")
    alert_id   = alert_payload.get("alert_id", "")

    logger.info(f"[Pipeline] Processing alert {alert_id} for {subject_id}")
    _agent_stats["total_processed"] += 1

    # ?? Step 1: Get digital footprint ?????????????????????????
    # In production: fetch real OSINT data from your data sources
    # Here: use simulated footprint, bias toward high risk for flagged subjects
    is_high_risk = alert_payload.get("composite_threat_index", 0) > 0.85
    footprint = nlp_agent.get_simulated_footprint(subject_id, is_high_risk)

    # ?? Step 2: NLP Analysis ??????????????????????????????????
    nlp_result = nlp_agent.analyze(footprint)
    _agent_stats["nlp_enriched"] += 1

    # ?? Step 3: Synthesize Report ?????????????????????????????
    enriched = synthesizer.synthesize(alert_payload, nlp_result)

    # ?? Step 4: Store and push to dashboard ??????????????????
    _enriched_alerts.insert(0, enriched)  # Newest first
    if len(_enriched_alerts) > 200:
        _enriched_alerts.pop()

    severity = enriched.get("final_severity", "NEGLIGIBLE")
    if severity == "CRITICAL":
        _agent_stats["critical_alerts"] += 1
    elif severity == "HIGH":
        _agent_stats["high_alerts"] += 1

    # Push to all connected browser SSE clients
    _push_to_browsers(enriched)

    # ?? Step 5: Enrich Sensor Hub ?????????????????????????????
    listener.enrich_alert(alert_id, {
        "digital_risk_score": nlp_result.get("digital_risk_score"),
        "flagged_phrases":    nlp_result.get("flagged_phrases", []),
        "report_narrative":   enriched.get("report_narrative"),
        "final_severity":     enriched.get("final_severity"),
        "final_threat_score": enriched.get("final_threat_score"),
    })

    logger.info(
        f"[Pipeline] [OK] Alert {alert_id} processed | "
        f"Final score: {enriched.get('final_threat_score', 0):.3f} | "
        f"Severity: {severity}"
    )


def _push_to_browsers(alert_data: Dict):
    """Push alert to all connected dashboard browser clients via SSE."""
    import queue
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(alert_data)
        except Exception:
            dead.append(q)
    for q in dead:
        _sse_clients.remove(q)


# Start listener in background
listener = SensorHubListener(on_alert=process_alert)
listener.start(blocking=False)


# ?????????????????????????????????????????????????????????????
# FLASK ROUTES
# ?????????????????????????????????????????????????????????????

@app.route("/")
def index():
    """Serve the ARGUS Command Dashboard."""
    return render_template("index.html")


@app.route("/api/alerts")
def get_alerts():
    """Return stored enriched alerts as JSON."""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    severity = request.args.get("severity", "").upper() or None

    filtered = _enriched_alerts
    if severity:
        filtered = [a for a in filtered if a.get("final_severity") == severity]

    start = (page - 1) * per_page
    end = start + per_page

    return jsonify({
        "total": len(filtered),
        "page": page,
        "per_page": per_page,
        "alerts": filtered[start:end],
    })


@app.route("/api/alerts/<alert_id>")
def get_alert(alert_id: str):
    """Return a specific alert by ID."""
    for alert in _enriched_alerts:
        if alert.get("alert_id") == alert_id:
            return jsonify(alert)
    return jsonify({"error": "Alert not found"}), 404


@app.route("/api/stats")
def get_stats():
    """Return dashboard statistics."""
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "NEGLIGIBLE": 0}
    anomaly_counts: Dict[str, int] = {}
    subject_scores: Dict[str, float] = {}

    for alert in _enriched_alerts:
        sev = alert.get("final_severity", "NEGLIGIBLE")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        atype = alert.get("gait_class", "UNKNOWN")
        anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1
        subj = alert.get("subject_id", "")
        score = alert.get("final_threat_score", 0)
        if subj and (subj not in subject_scores or score > subject_scores[subj]):
            subject_scores[subj] = score

    top_subjects = sorted(subject_scores.items(), key=lambda x: -x[1])[:5]

    return jsonify({
        **_agent_stats,
        "total_stored": len(_enriched_alerts),
        "by_severity": severity_counts,
        "by_anomaly_type": anomaly_counts,
        "top_threat_subjects": [
            {"subject_id": s, "score": round(sc, 4)} for s, sc in top_subjects
        ],
        "listener_stats": listener.stats,
    })


@app.route("/api/stream")
def browser_sse_stream():
    """
    SSE endpoint for real-time dashboard updates.
    Browser connects here and receives alert events as they arrive.
    """
    import queue

    def event_gen():
        q: queue.Queue = queue.Queue()
        _sse_clients.append(q)
        try:
            # Send current connection confirmation
            yield f"event: connected\ndata: {json.dumps({'status': 'ok', 'total': len(_enriched_alerts)})}\n\n"

            while True:
                try:
                    alert = q.get(timeout=30)
                    payload = json.dumps(alert, default=str)
                    yield f"event: alert\ndata: {payload}\n\n"
                except queue.Empty:
                    yield "event: heartbeat\ndata: {}\n\n"
        except GeneratorExit:
            _sse_clients.remove(q)

    return Response(
        stream_with_context(event_gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/inject-test", methods=["POST"])
def inject_test_alert():
    """
    Inject a synthetic test alert for dashboard demonstration.
    Called from the UI's 'Simulate Alert' button.
    """
    import uuid
    import random
    ANOMALY_CLASSES = [
        "NORMAL", "PREDATORY_GAIT", "TAILING", "AGGRESSIVE_POSTURE",
        "LOITERING", "RAPID_APPROACH", "CONCEALMENT",
    ]

    anomaly = random.choice(["TAILING", "AGGRESSIVE_POSTURE", "PREDATORY_GAIT", "LOITERING"])
    score = random.uniform(0.86, 0.98)

    dummy_payload = {
        "alert_id": f"ARGUS-DEMO-{uuid.uuid4().hex[:4].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": random.choice(["CAM-01", "CAM-02", "CAM-03", "CAM-NORTH", "CAM-SOUTH"]),
        "subject_id": f"SUBJ-{random.randint(1, 20):04d}",
        "composite_threat_index": round(score, 4),
        "anomaly_type": f"{anomaly}:ROUTE_DEVIATION",
        "anomaly_label": anomaly.replace("_", " ").title(),
        "gait_class": anomaly,
        "gait_confidence": round(random.uniform(0.78, 0.96), 4),
        "anomaly_score": round(score + random.uniform(-0.05, 0.05), 4),
        "face_stress_index": round(random.uniform(0.4, 0.9), 4),
        "route_deviation_sigma": round(random.uniform(2.1, 4.5), 3),
        "gnn_route_score": round(random.uniform(0.6, 0.9), 4),
        "hotspot_risk": round(random.uniform(0.3, 0.8), 4),
        "proximity_alert": random.choice([True, False]),
        "location_coords": [
            round(37.7749 + random.uniform(-0.02, 0.02), 6),
            round(-122.4194 + random.uniform(-0.02, 0.02), 6),
        ],
        "severity": "HIGH" if score < 0.92 else "CRITICAL",
        "all_class_probs": {cls: round(random.uniform(0, 0.15), 4) for cls in ANOMALY_CLASSES},
    }

    threading.Thread(target=process_alert, args=(dummy_payload,), daemon=True).start()
    return jsonify({"status": "injected", "alert_id": dummy_payload["alert_id"]})


@app.route("/api/clear", methods=["DELETE"])
def clear_alerts():
    _enriched_alerts.clear()
    _agent_stats["total_processed"] = 0
    _agent_stats["critical_alerts"] = 0
    _agent_stats["high_alerts"] = 0
    return jsonify({"status": "cleared"})


# ?????????????????????????????????????????????????????????????
# STARTUP
# ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    print("=" * 60)
    print("  ARGUS COMMAND DASHBOARD - Starting")
    print("  Open: http://localhost:5000")
    print("  API:  http://localhost:5000/api/stats")
    print("=" * 60)
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )

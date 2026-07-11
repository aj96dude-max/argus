"""
ARGUS - FastAPI Sensor Hub
The central API hub that receives threat payloads from the ML pipeline
and streams them to the Agentic Engine via Server-Sent Events (SSE).

Endpoints:
  POST /alert        - ML pipeline pushes ThreatPayload JSON
  GET  /stream       - Agentic Engine subscribes to SSE alert stream
  GET  /alerts       - Returns all stored alerts (paginated)
  GET  /status       - Health check
  GET  /stats        - Alert statistics dashboard
"""

from __future__ import annotations
import sys
import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENSOR_HUB_HOST, SENSOR_HUB_PORT

logger = logging.getLogger("ARGUS.SensorHub")

# ?????????????????????????????????????????????????????????????
# FASTAPI APP
# ?????????????????????????????????????????????????????????????

app = FastAPI(
    title="ARGUS Sensor Hub API",
    description="Real-time threat payload relay for Project ARGUS",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ?????????????????????????????????????????????????????????????
# PYDANTIC SCHEMAS
# ?????????????????????????????????????????????????????????????

class ThreatPayload(BaseModel):
    alert_id: str = Field(..., description="Unique alert identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    camera_id: str = Field(default="CAM-01")
    subject_id: str = Field(..., description="ARGUS subject identifier")
    threat_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_type: str = Field(..., description="Primary anomaly classification")
    anomaly_label: str = Field(default="")
    gait_class: str = Field(default="UNKNOWN")
    gait_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    face_stress_index: float = Field(default=0.0, ge=0.0, le=1.0)
    route_deviation_sigma: float = Field(default=0.0)
    gnn_route_score: float = Field(default=0.0)
    hotspot_risk: float = Field(default=0.0)
    proximity_alert: bool = Field(default=False)
    location_coords: List[float] = Field(default=[0.0, 0.0])
    severity: str = Field(default="NEGLIGIBLE")
    composite_threat_index: float = Field(..., ge=0.0, le=1.0)
    all_class_probs: Dict[str, float] = Field(default_factory=dict)
    digital_risk_score: Optional[float] = Field(default=None)
    flagged_phrases: List[str] = Field(default_factory=list)
    report_narrative: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "ARGUS-20260624-A1B2",
                "timestamp": "2026-06-24T18:17:08+05:00",
                "camera_id": "CAM-01",
                "subject_id": "SUBJ-0042",
                "threat_score": 0.91,
                "anomaly_type": "PREDATORY_GAIT:TAILING",
                "anomaly_label": "Predatory Gait: Tailing",
                "gait_class": "TAILING",
                "gait_confidence": 0.88,
                "anomaly_score": 0.91,
                "face_stress_index": 0.73,
                "route_deviation_sigma": 2.8,
                "gnn_route_score": 0.76,
                "hotspot_risk": 0.42,
                "proximity_alert": True,
                "location_coords": [37.7749, -122.4194],
                "severity": "CRITICAL",
                "composite_threat_index": 0.87,
                "all_class_probs": {},
                "digital_risk_score": None,
                "flagged_phrases": [],
                "report_narrative": None,
            }
        }


class AlertsResponse(BaseModel):
    total: int
    page: int
    per_page: int
    alerts: List[Dict[str, Any]]


# ?????????????????????????????????????????????????????????????
# IN-MEMORY ALERT STORE + SSE QUEUE
# ?????????????????????????????????????????????????????????????

_alert_store: List[Dict] = []
_sse_queues: List[asyncio.Queue] = []
_alert_stats = {
    "total_received": 0,
    "by_severity": {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "NEGLIGIBLE": 0},
    "by_anomaly_type": {},
    "first_alert_at": None,
    "last_alert_at": None,
}


def _update_stats(payload: Dict):
    _alert_stats["total_received"] += 1
    severity = payload.get("severity", "NEGLIGIBLE")
    _alert_stats["by_severity"][severity] = _alert_stats["by_severity"].get(severity, 0) + 1
    atype = payload.get("anomaly_type", "UNKNOWN")
    _alert_stats["by_anomaly_type"][atype] = _alert_stats["by_anomaly_type"].get(atype, 0) + 1
    now = datetime.now(timezone.utc).isoformat()
    if _alert_stats["first_alert_at"] is None:
        _alert_stats["first_alert_at"] = now
    _alert_stats["last_alert_at"] = now


async def _broadcast_to_sse(payload: Dict):
    """Push alert payload to all connected SSE subscribers."""
    dead_queues = []
    for q in _sse_queues:
        try:
            await q.put(payload)
        except Exception:
            dead_queues.append(q)
    for q in dead_queues:
        _sse_queues.remove(q)


# ?????????????????????????????????????????????????????????????
# ENDPOINTS
# ?????????????????????????????????????????????????????????????

@app.get("/status")
async def get_status():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "ARGUS Sensor Hub",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_alerts": len(_alert_store),
        "sse_subscribers": len(_sse_queues),
    }


@app.post("/alert", status_code=200)
async def receive_alert(payload: ThreatPayload):
    """
    Receives a ThreatPayload from the ML pipeline.
    Stores it and broadcasts to all SSE subscribers.
    """
    data = payload.model_dump()

    # Enrich with server-side receive timestamp
    data["received_at"] = datetime.now(timezone.utc).isoformat()

    # Store
    _alert_store.append(data)
    _update_stats(data)

    # Broadcast to SSE stream
    await _broadcast_to_sse(data)

    logger.info(
        f"ALERT received: {data['alert_id']} | "
        f"Subject: {data['subject_id']} | "
        f"Score: {data['composite_threat_index']:.3f} | "
        f"Severity: {data['severity']}"
    )

    return {
        "status": "received",
        "alert_id": data["alert_id"],
        "queued_for": f"{len(_sse_queues)} subscribers",
    }


@app.get("/stream")
async def sse_stream(request: Any = None):
    """
    Server-Sent Events stream.
    Agentic Engine subscribes here to receive real-time threat alerts.
    """
    queue: asyncio.Queue = asyncio.Queue()
    _sse_queues.append(queue)

    async def event_generator() -> AsyncGenerator:
        # Send current connection event
        yield {
            "event": "connected",
            "data": json.dumps({
                "status": "subscribed",
                "pending_alerts": len(_alert_store),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
        }

        try:
            while True:
                # Wait for new alert
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": "alert",
                        "id": payload.get("alert_id", ""),
                        "data": json.dumps(payload),
                    }
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"ts": datetime.now(timezone.utc).isoformat()}),
                    }
        except asyncio.CancelledError:
            _sse_queues.remove(queue)

    return EventSourceResponse(event_generator())


@app.get("/alerts", response_model=AlertsResponse)
async def get_alerts(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    severity: Optional[str] = Query(default=None),
    subject_id: Optional[str] = Query(default=None),
):
    """Retrieve stored alerts with optional filtering and pagination."""
    filtered = _alert_store.copy()

    if severity:
        filtered = [a for a in filtered if a.get("severity") == severity.upper()]
    if subject_id:
        filtered = [a for a in filtered if a.get("subject_id") == subject_id]

    # Sort by timestamp descending (newest first)
    filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    start = (page - 1) * per_page
    end = start + per_page

    return AlertsResponse(
        total=len(filtered),
        page=page,
        per_page=per_page,
        alerts=filtered[start:end],
    )


@app.get("/alerts/{alert_id}")
async def get_alert_by_id(alert_id: str):
    """Retrieve a specific alert by its ID."""
    for alert in _alert_store:
        if alert.get("alert_id") == alert_id:
            return alert
    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@app.patch("/alerts/{alert_id}")
async def enrich_alert(alert_id: str, enrichment: Dict):
    """
    Agentic Engine calls this to enrich an alert with NLP analysis results.
    Updates digital_risk_score, flagged_phrases, and report_narrative.
    """
    for i, alert in enumerate(_alert_store):
        if alert.get("alert_id") == alert_id:
            _alert_store[i].update(enrichment)
            # Broadcast updated alert to SSE subscribers
            await _broadcast_to_sse({**_alert_store[i], "event_type": "enriched"})
            return {"status": "enriched", "alert_id": alert_id}
    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@app.get("/stats")
async def get_stats():
    """Return alert statistics for dashboard display."""
    return {
        **_alert_stats,
        "active_subjects": len(set(a.get("subject_id") for a in _alert_store)),
        "avg_threat_score": (
            sum(a.get("composite_threat_index", 0) for a in _alert_store)
            / max(len(_alert_store), 1)
        ),
    }


@app.delete("/alerts")
async def clear_alerts():
    """Clear all stored alerts (admin/testing use)."""
    _alert_store.clear()
    _alert_stats["total_received"] = 0
    _alert_stats["by_severity"] = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0, "NEGLIGIBLE": 0}
    _alert_stats["by_anomaly_type"] = {}
    return {"status": "cleared"}


# ?????????????????????????????????????????????????????????????
# STARTUP
# ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    print("=" * 60)
    print("  ARGUS SENSOR HUB - Starting")
    print(f"  API:    http://{SENSOR_HUB_HOST}:{SENSOR_HUB_PORT}")
    print(f"  Docs:   http://localhost:{SENSOR_HUB_PORT}/docs")
    print(f"  Stream: http://localhost:{SENSOR_HUB_PORT}/stream")
    print("=" * 60)
    uvicorn.run(
        "sensor_hub:app",
        host=SENSOR_HUB_HOST,
        port=SENSOR_HUB_PORT,
        reload=False,
        log_level="info",
    )

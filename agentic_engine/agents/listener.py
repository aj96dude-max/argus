"""
ARGUS Agentic Engine - SSE Listener Agent
Connects to the ML Engine's FastAPI Sensor Hub via Server-Sent Events.
On receiving a threat payload, triggers the OSINT NLP agent pipeline.
"""

from __future__ import annotations
import sys
import json
import time
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import httpx
from httpx_sse import connect_sse

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ml_engine"))

# Sensor Hub connection settings
SENSOR_HUB_STREAM_URL = "http://localhost:8001/stream"
SENSOR_HUB_ALERTS_URL = "http://localhost:8001/alerts"
SENSOR_HUB_ENRICH_URL = "http://localhost:8001/alerts/{alert_id}"
RECONNECT_DELAY_SEC   = 5
MAX_RECONNECT_RETRIES = 20

logger = logging.getLogger("ARGUS.Listener")


class SensorHubListener:
    """
    Long-running SSE client that listens to the ARGUS Sensor Hub.
    Calls registered handler callbacks on each received threat alert.
    """

    def __init__(self, on_alert: Optional[Callable] = None):
        self._on_alert = on_alert or self._default_handler
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._retry_count = 0
        self._total_received = 0

    def _default_handler(self, payload: dict):
        logger.info(f"[Listener] Alert received: {payload.get('alert_id')} "
                    f"| Score: {payload.get('composite_threat_index')} "
                    f"| Severity: {payload.get('severity')}")

    def start(self, blocking: bool = True):
        """Start listening - can run in background thread."""
        self._running = True
        if blocking:
            self._listen_loop()
        else:
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            logger.info("[Listener] Started in background thread")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _listen_loop(self):
        """Main reconnecting SSE loop."""
        while self._running and self._retry_count < MAX_RECONNECT_RETRIES:
            try:
                logger.info(f"[Listener] Connecting to Sensor Hub: {SENSOR_HUB_STREAM_URL}")
                with httpx.Client(timeout=None) as client:
                    with connect_sse(client, "GET", SENSOR_HUB_STREAM_URL) as event_source:
                        self._retry_count = 0  # Reset on successful connection
                        logger.info("[Listener] [OK] Connected to SSE stream")

                        for sse_event in event_source.iter_sse():
                            if not self._running:
                                break

                            if sse_event.event == "connected":
                                data = json.loads(sse_event.data)
                                logger.info(
                                    f"[Listener] Subscribed | "
                                    f"Pending alerts: {data.get('pending_alerts', 0)}"
                                )

                            elif sse_event.event == "alert":
                                try:
                                    payload = json.loads(sse_event.data)
                                    self._total_received += 1
                                    logger.info(
                                        f"[Listener] -> Alert #{self._total_received}: "
                                        f"{payload.get('alert_id')} | "
                                        f"Severity: {payload.get('severity')}"
                                    )
                                    # Trigger handler in separate thread to not block SSE
                                    threading.Thread(
                                        target=self._safe_handle,
                                        args=(payload,),
                                        daemon=True,
                                    ).start()
                                except json.JSONDecodeError as e:
                                    logger.error(f"[Listener] JSON parse error: {e}")

                            elif sse_event.event == "heartbeat":
                                logger.debug("[Listener] ? Heartbeat received")

                            elif sse_event.event == "enriched":
                                logger.debug("[Listener] Alert enriched by NLP agent")

            except httpx.ConnectError:
                self._retry_count += 1
                logger.warning(
                    f"[Listener] Sensor Hub not reachable. "
                    f"Retry {self._retry_count}/{MAX_RECONNECT_RETRIES} "
                    f"in {RECONNECT_DELAY_SEC}s..."
                )
                time.sleep(RECONNECT_DELAY_SEC)

            except httpx.ReadError:
                logger.warning("[Listener] SSE connection dropped. Reconnecting...")
                time.sleep(2)

            except Exception as e:
                logger.error(f"[Listener] Unexpected error: {e}")
                time.sleep(RECONNECT_DELAY_SEC)

        if self._retry_count >= MAX_RECONNECT_RETRIES:
            logger.error("[Listener] Max retries exceeded. Listener stopped.")

    def _safe_handle(self, payload: dict):
        """Call handler with error isolation."""
        try:
            self._on_alert(payload)
        except Exception as e:
            logger.error(f"[Listener] Handler error: {e}", exc_info=True)

    def enrich_alert(self, alert_id: str, enrichment: dict) -> bool:
        """
        PATCH the Sensor Hub to update an alert with NLP enrichment data.
        Called after OSINT NLP analysis completes.
        """
        try:
            url = SENSOR_HUB_ENRICH_URL.format(alert_id=alert_id)
            with httpx.Client(timeout=5.0) as client:
                resp = client.patch(url, json=enrichment)
                if resp.status_code == 200:
                    logger.info(f"[Listener] Alert {alert_id} enriched [OK]")
                    return True
        except Exception as e:
            logger.error(f"[Listener] Enrich error: {e}")
        return False

    @property
    def stats(self) -> dict:
        return {
            "total_received": self._total_received,
            "retry_count": self._retry_count,
            "is_running": self._running,
        }


# ?????????????????????????????????????????????????????????????
# STANDALONE TEST
# ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    def test_handler(payload: dict):
        print(f"\n{'='*60}")
        print(f"  ALERT RECEIVED")
        print(f"  ID:       {payload.get('alert_id')}")
        print(f"  Subject:  {payload.get('subject_id')}")
        print(f"  Score:    {payload.get('composite_threat_index')}")
        print(f"  Severity: {payload.get('severity')}")
        print(f"  Type:     {payload.get('anomaly_type')}")
        print(f"{'='*60}\n")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-payload", action="store_true",
                        help="Inject a synthetic test payload")
    args = parser.parse_args()

    if args.test_payload:
        # Inject a dummy alert directly via POST to test the pipeline
        import uuid
        from datetime import datetime, timezone
        dummy = {
            "alert_id": f"ARGUS-TEST-{uuid.uuid4().hex[:4].upper()}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": "CAM-TEST",
            "subject_id": "SUBJ-0000",
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
            "all_class_probs": {"NORMAL": 0.05, "TAILING": 0.88, "LOITERING": 0.07},
            "digital_risk_score": None,
            "flagged_phrases": [],
            "report_narrative": None,
        }
        try:
            resp = httpx.post("http://localhost:8001/alert", json=dummy, timeout=3.0)
            print(f"Test payload injected: {resp.status_code} | {resp.json()}")
        except Exception as e:
            print(f"Could not reach Sensor Hub: {e}. Start sensor_hub.py first.")
    else:
        listener = SensorHubListener(on_alert=test_handler)
        print("[Listener] Connecting to ARGUS Sensor Hub. Ctrl+C to stop.")
        listener.start(blocking=True)

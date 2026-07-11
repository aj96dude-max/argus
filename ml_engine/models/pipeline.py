"""
ARGUS - Unified ML Pipeline
Orchestrates Module A -> B -> C -> Alert API.
Processes live video frames and fires threat alerts when anomaly_score > 0.85.
"""

from __future__ import annotations
import sys
import time
import json
import uuid
import logging
import asyncio
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import cv2
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANOMALY_THRESHOLD, ALERT_ENDPOINT, SENSOR_HUB_URL,
    FRAME_SKIP, VIDEO_FPS, CHECKPOINTS_DIR, DEVICE,
    SEVERITY_LEVELS
)
from models.spatial import SpatialExtractor, SpatialFrame
from models.temporal import TemporalAnalyzer
from models.spatiotemporal import SpatiotemporalTracker

logger = logging.getLogger("ARGUS.Pipeline")


def get_severity(score: float) -> str:
    for level, (low, high) in SEVERITY_LEVELS.items():
        if low <= score < high:
            return level
    return "NEGLIGIBLE"


def compute_composite_score(
    anomaly_score: float,
    gnn_score: float,
    deviation_score: float,
    face_stress: float,
    proximity_alert: bool,
) -> float:
    """
    Weighted fusion of all module scores into composite threat index.
    Weights reflect module reliability and evidence strength.
    """
    proximity_bonus = 0.15 if proximity_alert else 0.0

    composite = (
        0.40 * anomaly_score       +   # CNN-LSTM gait (primary)
        0.20 * gnn_score           +   # GNN route
        0.15 * deviation_score     +   # Route deviation sigma
        0.10 * face_stress         +   # Facial stress
        proximity_bonus                 # Proximity overlap
    )
    return float(np.clip(composite, 0.0, 1.0))


def build_threat_payload(
    subject_id: str,
    gait_result: Dict,
    spatial_result: Optional[SpatialFrame],
    route_result: Dict,
    composite_score: float,
    camera_id: str = "CAM-01",
) -> Dict:
    """Construct the standardized ARGUS threat JSON payload."""
    subject_data = None
    if spatial_result:
        subject_data = spatial_result.get_subject(subject_id)

    face_stress = 0.0
    if subject_data:
        face_stress = subject_data.face_stress_indicators.get("stress_index", 0.0)

    anomaly_type = gait_result.get("predicted_class", "UNKNOWN")
    gait_conf = gait_result.get("class_confidence", 0.0)

    # Humanize anomaly type label
    anomaly_label = anomaly_type.replace("_", " ").title()
    if route_result.get("route_deviation_sigma", 0) > 2.0:
        anomaly_type += ":ROUTE_DEVIATION"

    return {
        "alert_id": f"ARGUS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "subject_id": subject_id,
        "threat_score": round(composite_score, 4),
        "anomaly_type": anomaly_type,
        "anomaly_label": anomaly_label,
        "gait_class": gait_result.get("predicted_class", "UNKNOWN"),
        "gait_confidence": round(gait_conf, 4),
        "anomaly_score": round(gait_result.get("anomaly_score", 0.0), 4),
        "face_stress_index": round(face_stress, 4),
        "route_deviation_sigma": round(route_result.get("route_deviation_sigma", 0.0), 3),
        "gnn_route_score": round(route_result.get("gnn_anomaly_score", 0.0), 4),
        "hotspot_risk": round(route_result.get("hotspot_risk", 0.0), 4),
        "proximity_alert": route_result.get("proximity_alert", False),
        "location_coords": [
            round(route_result.get("lat", 0.0), 6),
            round(route_result.get("lon", 0.0), 6),
        ],
        "severity": get_severity(composite_score),
        "composite_threat_index": round(composite_score, 4),
        "all_class_probs": gait_result.get("all_class_probs", {}),
        "digital_risk_score": None,  # Will be enriched by Agentic Engine
        "flagged_phrases": [],        # Will be enriched by Agentic Engine
        "report_narrative": None,     # Will be enriched by Agentic Engine
    }


class ARGUSPipeline:
    """
    Main ARGUS inference pipeline.
    Chains: Video -> Module A (Spatial) -> Module B (Temporal) -> Module C (Spatiotemporal)
    -> Threat Payload -> FastAPI Sensor Hub -> Agentic Engine
    """

    def __init__(
        self,
        video_source=0,
        camera_id: str = "CAM-01",
        use_simulated_gps: bool = True,
    ):
        self.video_source = video_source
        self.camera_id = camera_id
        self.use_simulated_gps = use_simulated_gps

        logger.info("Initializing ARGUS Pipeline...")
        self.spatial = SpatialExtractor()
        self.temporal = TemporalAnalyzer(
            model_checkpoint=CHECKPOINTS_DIR / "cnn_lstm_best.pt"
        )
        self.spatiotemporal = SpatiotemporalTracker(
            model_checkpoint=CHECKPOINTS_DIR / "gnn_best.pt"
        )

        self._frame_idx = 0
        self._alert_count = 0
        self._last_alert_time: Dict[str, float] = {}
        self._alert_cooldown_sec = 10.0  # Avoid spamming same subject

        # Simulated GPS base (San Francisco) - used when real GPS unavailable
        self._sim_gps_base = (37.7749, -122.4194)

        logger.info(f"Pipeline ready. Video: {video_source} | Camera: {camera_id}")

    def _simulate_gps(self, subject_id: str) -> tuple:
        """Generate drifting GPS coordinates for demo/testing."""
        import math
        seed = hash(subject_id) % 1000
        t = time.time()
        lat = self._sim_gps_base[0] + math.sin(t * 0.01 + seed) * 0.005
        lon = self._sim_gps_base[1] + math.cos(t * 0.01 + seed) * 0.005
        return lat, lon

    def _should_alert(self, subject_id: str) -> bool:
        """Rate-limit alerts per subject."""
        now = time.time()
        last = self._last_alert_time.get(subject_id, 0.0)
        if now - last >= self._alert_cooldown_sec:
            self._last_alert_time[subject_id] = now
            return True
        return False

    def _push_alert(self, payload: Dict) -> bool:
        """Push threat payload to FastAPI Sensor Hub."""
        try:
            response = httpx.post(
                ALERT_ENDPOINT,
                json=payload,
                timeout=3.0,
            )
            if response.status_code == 200:
                self._alert_count += 1
                logger.info(
                    f"[Pipeline] ALERT #{self._alert_count} pushed | "
                    f"{payload['subject_id']} | Score: {payload['composite_threat_index']:.3f} "
                    f"| Severity: {payload['severity']}"
                )
                return True
            else:
                logger.warning(f"Alert push failed: HTTP {response.status_code}")
        except httpx.ConnectError:
            logger.warning("Sensor Hub not reachable. Alert dropped. Start sensor_hub.py first.")
        except Exception as e:
            logger.error(f"Alert push error: {e}")
        return False

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process one video frame through the full ARGUS pipeline.
        Returns threat payload if alert fired, else None.
        """
        # ?? Module A: Spatial Extraction ?????????????????????
        spatial_frame = self.spatial.process_frame(
            frame,
            frame_idx=self._frame_idx,
            timestamp=self._frame_idx / max(VIDEO_FPS, 1),
        )

        fired_payload = None

        # ?? Per-Subject Analysis ??????????????????????????????
        for subject in spatial_frame.subjects:
            # ?? Module B: Temporal/Gait ???????????????????????
            gait_result = self.temporal.update(subject)
            if gait_result is None:
                continue  # Window not yet full

            # ?? Module C: Spatiotemporal ??????????????????????
            lat, lon = self._simulate_gps(subject.subject_id) if self.use_simulated_gps else (0, 0)
            now = datetime.now()
            route_result = self.spatiotemporal.update_subject(
                subject_id=subject.subject_id,
                lat=lat, lon=lon,
                hour=now.hour,
                dayofweek=now.weekday(),
            )
            route_result["lat"] = lat
            route_result["lon"] = lon

            # ?? Composite Threat Score ????????????????????????
            face_stress = subject.face_stress_indicators.get("stress_index", 0.0)
            composite = compute_composite_score(
                anomaly_score=gait_result["anomaly_score"],
                gnn_score=route_result["gnn_anomaly_score"],
                deviation_score=route_result["route_deviation_score"],
                face_stress=face_stress,
                proximity_alert=route_result["proximity_alert"],
            )

            # ?? Threshold Gate -> Alert ????????????????????????
            if composite >= ANOMALY_THRESHOLD and self._should_alert(subject.subject_id):
                payload = build_threat_payload(
                    subject_id=subject.subject_id,
                    gait_result=gait_result,
                    spatial_result=spatial_frame,
                    route_result=route_result,
                    composite_score=composite,
                    camera_id=self.camera_id,
                )
                self._push_alert(payload)
                fired_payload = payload

        self._frame_idx += 1
        return fired_payload

    def run(self, display: bool = True):
        """Main run loop - processes live video stream or file."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.video_source}")
            return

        logger.info(f"ARGUS Pipeline running | Source: {self.video_source}")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(self.video_source, str):
                        logger.info("Video file ended.")
                        break
                    continue

                frame_count += 1
                if frame_count % (FRAME_SKIP + 1) != 0:
                    continue

                # Process frame
                payload = self.process_frame(frame)

                # ?? Visualization ?????????????????????????????
                if display:
                    # Re-process for drawing (lightweight)
                    sf = self.spatial.process_frame(frame, self._frame_idx)
                    annotated = self.spatial.draw_annotations(sf)

                    # Overlay status
                    status = f"ARGUS | Frame:{self._frame_idx} | Alerts:{self._alert_count}"
                    cv2.putText(annotated, status, (10, annotated.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    if payload:
                        severity = payload.get("severity", "")
                        score = payload.get("composite_threat_index", 0)
                        alert_text = f"[WARN] ALERT: {severity} | Score: {score:.3f}"
                        cv2.putText(annotated, alert_text, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("ARGUS - Live Threat Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.spatial.release()
            logger.info(f"Pipeline shutdown. Total alerts fired: {self._alert_count}")


# ?????????????????????????????????????????????????????????????
# DEMO MODE - Synthetic video without real datasets
# ?????????????????????????????????????????????????????????????

def run_demo():
    """Run ARGUS pipeline on synthetic video with forced anomaly for testing."""
    from data.synthetic import generate_synthetic_video, SYNTHETIC_VIDEO_DIR

    print("=" * 60)
    print("  ARGUS - Demo Mode (Synthetic Video)")
    print("=" * 60)

    demo_video = SYNTHETIC_VIDEO_DIR / "TAILING" / "demo.mp4"
    generate_synthetic_video(demo_video, label="TAILING", n_frames=120)

    pipeline = ARGUSPipeline(
        video_source=str(demo_video),
        camera_id="CAM-DEMO",
        use_simulated_gps=True,
    )
    pipeline.run(display=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ARGUS ML Pipeline")
    parser.add_argument("--input", type=str, default="0",
                        help="Video source: 0=webcam, path=video file, 'synthetic'=demo mode")
    parser.add_argument("--camera", type=str, default="CAM-01")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if args.input == "synthetic":
        run_demo()
    else:
        source = int(args.input) if args.input.isdigit() else args.input
        pipeline = ARGUSPipeline(
            video_source=source,
            camera_id=args.camera,
            use_simulated_gps=True,
        )
        pipeline.run(display=not args.no_display)

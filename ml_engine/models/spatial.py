"""
ARGUS - Module A: Spatial Feature Extraction
YOLOv8 person detection + MediaPipe 3D skeletal pose + facial landmarks.
Processes each frame and returns structured SpatialFrame data.
"""

from __future__ import annotations
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    YOLO_MODEL_SIZE, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    MEDIAPIPE_COMPLEXITY, POSE_LANDMARKS, FACE_LANDMARKS,
    FRAME_WIDTH, FRAME_HEIGHT, MAX_SUBJECTS, DEVICE
)

logger = logging.getLogger("ARGUS.ModuleA")

# ?????????????????????????????????????????????????????????????
# DATA STRUCTURES
# ?????????????????????????????????????????????????????????????

@dataclass
class SubjectSpatialData:
    """Spatial data extracted for one tracked subject in one frame."""
    subject_id: str
    bbox: Tuple[int, int, int, int]             # (x1, y1, x2, y2)
    detection_confidence: float
    pose_landmarks: Optional[np.ndarray] = None  # (33, 3) - x, y, z
    face_landmarks: Optional[np.ndarray] = None  # (468, 3)
    pose_confidence: float = 0.0
    face_detected: bool = False
    face_stress_indicators: Dict[str, float] = field(default_factory=dict)

    @property
    def pose_flat(self) -> np.ndarray:
        """Flatten pose landmarks to (99,) vector for CNN-LSTM input."""
        if self.pose_landmarks is not None:
            return self.pose_landmarks.flatten()
        return np.zeros(POSE_LANDMARKS * 3, dtype=np.float32)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class SpatialFrame:
    """All spatial data extracted from one video frame."""
    frame_idx: int
    timestamp: float                              # Seconds from video start
    subjects: List[SubjectSpatialData]
    raw_frame: Optional[np.ndarray] = None        # BGR frame for visualization

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    def get_subject(self, subject_id: str) -> Optional[SubjectSpatialData]:
        for s in self.subjects:
            if s.subject_id == subject_id:
                return s
        return None


# ?????????????????????????????????????????????????????????????
# MODULE A - SPATIAL EXTRACTOR
# ?????????????????????????????????????????????????????????????

class SpatialExtractor:
    """
    Module A: Detects humans with YOLOv8, then extracts
    3D skeletal pose and facial landmarks via MediaPipe Holistic.
    """

    def __init__(self):
        self.yolo_model = None
        self.mp_holistic = None
        self.mp_drawing = None
        self._subject_tracker: Dict[str, str] = {}  # track_id -> subject_id
        self._next_subject_id = 0
        self._init_models()

    def _init_models(self):
        """Lazy-load YOLOv8 and MediaPipe."""
        logger.info("Initializing Module A: YOLOv8 + MediaPipe...")

        # YOLOv8
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(YOLO_MODEL_SIZE)
            self.yolo_model.to(DEVICE)
            logger.info(f"YOLOv8 ({YOLO_MODEL_SIZE}) loaded on {DEVICE}")
        except Exception as e:
            logger.error(f"YOLOv8 load failed: {e}. Using bounding box fallback.")
            self.yolo_model = None

        # MediaPipe
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=MEDIAPIPE_COMPLEXITY,
                enable_segmentation=False,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            logger.info("MediaPipe Holistic loaded")
        except Exception as e:
            logger.error(f"MediaPipe load failed: {e}.")
            self.mp_holistic = None

    def _assign_subject_id(self, track_id: Optional[int]) -> str:
        """Map YOLO track ID to stable ARGUS subject ID."""
        key = str(track_id) if track_id is not None else "unknown"
        if key not in self._subject_tracker:
            self._subject_tracker[key] = f"SUBJ-{self._next_subject_id:04X}"
            self._next_subject_id += 1
        return self._subject_tracker[key]

    def _extract_pose_landmarks(
        self, frame_rgb: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Run MediaPipe Holistic on a cropped subject ROI.
        Returns: (pose_landmarks (33,3), face_landmarks (468,3), confidence)
        """
        if self.mp_holistic is None:
            return None, None, 0.0

        x1, y1, x2, y2 = bbox
        # Expand ROI slightly for better holistic results
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame_rgb.shape[1], x2 + pad)
        y2 = min(frame_rgb.shape[0], y2 + pad)

        roi = frame_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, 0.0

        roi_resized = cv2.resize(roi, (256, 512))
        results = self.mp_holistic.process(roi_resized)

        pose_landmarks = None
        face_landmarks = None
        confidence = 0.0

        if results.pose_landmarks:
            pose_landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ], dtype=np.float32)
            confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])

        if results.face_landmarks:
            face_landmarks = np.array([
                [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
            ], dtype=np.float32)

        return pose_landmarks, face_landmarks, float(confidence)

    def _compute_face_stress_indicators(
        self, face_landmarks: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute micro-expression proxy indicators from facial landmarks.
        These are geometric approximations of AU (Action Units):
          - brow_furrow: inner brow elevation difference
          - lip_compression: lip separation distance
          - jaw_tension: jaw-to-chin distance
        """
        if face_landmarks is None or len(face_landmarks) < 468:
            return {}

        indicators = {}

        # Brow furrow - landmarks 19, 24 (inner brow points)
        try:
            left_inner_brow = face_landmarks[19]
            right_inner_brow = face_landmarks[24]
            brow_separation = abs(left_inner_brow[1] - right_inner_brow[1])
            indicators["brow_furrow"] = float(np.clip(brow_separation * 10, 0, 1))

            # Lip compression - landmarks 13, 14 (upper/lower lip center)
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            lip_dist = np.linalg.norm(upper_lip - lower_lip)
            indicators["lip_compression"] = float(np.clip(1 - lip_dist * 20, 0, 1))

            # Jaw tension - landmark 152 (chin) vs 0 (top of face)
            chin = face_landmarks[152]
            top = face_landmarks[10]
            jaw_dist = np.linalg.norm(chin - top)
            indicators["jaw_tension"] = float(np.clip(jaw_dist * 3, 0, 1))

            # Aggregate stress index (weighted)
            indicators["stress_index"] = (
                0.4 * indicators["brow_furrow"] +
                0.35 * indicators["lip_compression"] +
                0.25 * indicators["jaw_tension"]
            )
        except (IndexError, ValueError):
            indicators["stress_index"] = 0.0

        return indicators

    def process_frame(
        self, frame: np.ndarray, frame_idx: int = 0, timestamp: float = 0.0
    ) -> SpatialFrame:
        """
        Main entry point: process one BGR frame.
        Returns a SpatialFrame with all detected subjects.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        subjects: List[SubjectSpatialData] = []

        # ?? YOLOv8 Detection + Tracking ??????????????????????
        detections = self._run_yolo(frame)

        for det in detections[:MAX_SUBJECTS]:
            bbox = det["bbox"]
            confidence = det["confidence"]
            track_id = det.get("track_id")
            subject_id = self._assign_subject_id(track_id)

            # ?? MediaPipe Pose + Face ?????????????????????????
            pose_lm, face_lm, pose_conf = self._extract_pose_landmarks(frame_rgb, bbox)
            stress = self._compute_face_stress_indicators(face_lm)

            subject = SubjectSpatialData(
                subject_id=subject_id,
                bbox=bbox,
                detection_confidence=confidence,
                pose_landmarks=pose_lm,
                face_landmarks=face_lm,
                pose_confidence=pose_conf,
                face_detected=face_lm is not None,
                face_stress_indicators=stress,
            )
            subjects.append(subject)

        return SpatialFrame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            subjects=subjects,
            raw_frame=frame.copy(),
        )

    def _run_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLOv8 person detection with tracking."""
        detections = []
        if self.yolo_model is None:
            return self._fallback_detection(frame)

        try:
            results = self.yolo_model.track(
                frame,
                persist=True,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU_THRESHOLD,
                classes=[0],  # Person class only
                verbose=False,
            )
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else None
                    detections.append({
                        "bbox": tuple(xyxy),
                        "confidence": conf,
                        "track_id": track_id,
                    })
        except Exception as e:
            logger.warning(f"YOLOv8 inference error: {e}")
            return self._fallback_detection(frame)

        return detections

    def _fallback_detection(self, frame: np.ndarray) -> List[Dict]:
        """
        Simple MOG2 background subtraction fallback when YOLO is unavailable.
        Returns approximate bounding boxes for moving objects.
        """
        if not hasattr(self, "_bg_subtractor"):
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=50, detectShadows=False
            )
        mask = self._bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    "bbox": (x, y, x + w, y + h),
                    "confidence": 0.6,
                    "track_id": None,
                })
        return detections[:MAX_SUBJECTS]

    def draw_annotations(self, spatial_frame: SpatialFrame) -> np.ndarray:
        """Draw YOLO boxes, pose landmarks, and stress indicators on frame."""
        frame = spatial_frame.raw_frame.copy()

        for subj in spatial_frame.subjects:
            x1, y1, x2, y2 = subj.bbox
            stress = subj.face_stress_indicators.get("stress_index", 0.0)

            # Color based on stress level
            if stress > 0.7:
                color = (0, 0, 255)   # Red - high stress
            elif stress > 0.4:
                color = (0, 165, 255)  # Orange - moderate
            else:
                color = (0, 255, 0)   # Green - normal

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{subj.subject_id} | conf:{subj.detection_confidence:.2f}"
            if stress > 0:
                label += f" | stress:{stress:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def release(self):
        if self.mp_holistic:
            self.mp_holistic.close()


# ?????????????????????????????????????????????????????????????
# STANDALONE TEST
# ?????????????????????????????????????????????????????????????

if __name__ == "__main__":
    import time
    from data.synthetic import generate_synthetic_video

    print("[Module A] Running standalone test with synthetic video...")
    test_video = Path("data/synthetic_videos/test_moduleA.mp4")
    generate_synthetic_video(test_video, label="TAILING", n_frames=60)

    extractor = SpatialExtractor()
    cap = cv2.VideoCapture(str(test_video))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        spatial_frame = extractor.process_frame(frame, frame_idx=frame_idx)
        elapsed = time.time() - t0

        print(f"Frame {frame_idx:03d}: {spatial_frame.n_subjects} subjects | "
              f"Processing: {elapsed*1000:.1f}ms")

        for subj in spatial_frame.subjects:
            stress = subj.face_stress_indicators.get("stress_index", 0.0)
            print(f"  {subj.subject_id}: pose_conf={subj.pose_confidence:.2f} "
                  f"stress={stress:.2f} pose_shape={subj.pose_flat.shape}")

        annotated = extractor.draw_annotations(spatial_frame)
        cv2.imshow("ARGUS Module A - Spatial", annotated)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    extractor.release()
    print("[Module A] Test complete.")

"""
ARGUS ML Engine - Synthetic Data Generator
Generates realistic dummy data for testing without real datasets.
Run standalone: python data/synthetic.py
"""

import sys
import json
import random
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEMPORAL_WINDOW, FRAME_WIDTH, FRAME_HEIGHT, ANOMALY_CLASSES
)

DATA_DIR = Path(__file__).parent
SYNTHETIC_VIDEO_DIR = DATA_DIR / "synthetic_videos"
SYNTHETIC_GIS_FILE  = DATA_DIR / "synthetic_gis.json"
SYNTHETIC_OSINT_FILE = DATA_DIR / "synthetic_osint.json"


# ?????????????????????????????????????????????????????????????
# SYNTHETIC VIDEO GENERATOR
# ?????????????????????????????????????????????????????????????

def generate_synthetic_video(
    output_path: Path,
    label: str = "NORMAL",
    n_frames: int = 64,
    fps: int = 15,
):
    """
    Generates a synthetic surveillance video with a moving blob as a subject.
    Labels influence movement patterns:
      - NORMAL: smooth random walk
      - TAILING: blob follows another blob closely
      - AGGRESSIVE_POSTURE: rapid erratic motion
      - LOITERING: blob circles in a small radius
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    # Subject start position
    x, y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    target_x, target_y = FRAME_WIDTH // 4, FRAME_HEIGHT // 3

    for i in range(n_frames):
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 40, dtype=np.uint8)

        # Simulate background noise
        noise = np.random.randint(0, 15, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Draw some static background elements
        cv2.rectangle(frame, (50, 50), (150, FRAME_HEIGHT-50), (60, 60, 60), -1)
        cv2.rectangle(frame, (FRAME_WIDTH-200, 80), (FRAME_WIDTH-50, FRAME_HEIGHT-80), (55, 55, 70), -1)

        # Movement logic based on label
        if label == "NORMAL":
            x += random.randint(-5, 5)
            y += random.randint(-3, 3)
        elif label == "TAILING":
            # Follow target with slight lag
            x += (target_x - x) * 0.05 + random.randint(-2, 2)
            y += (target_y - y) * 0.05 + random.randint(-1, 1)
            target_x += random.randint(-8, 8)
            target_y += random.randint(-5, 5)
            # Draw target (victim)
            target_x = np.clip(target_x, 30, FRAME_WIDTH-30)
            target_y = np.clip(target_y, 30, FRAME_HEIGHT-30)
            cv2.ellipse(frame, (int(target_x), int(target_y)), (10, 25), 0, 0, 360, (0, 200, 100), -1)
        elif label == "AGGRESSIVE_POSTURE":
            x += random.randint(-15, 15)
            y += random.randint(-15, 15)
        elif label == "LOITERING":
            angle = (i / n_frames) * 2 * math.pi * 3
            radius = 40
            x = FRAME_WIDTH // 2 + int(radius * math.cos(angle))
            y = FRAME_HEIGHT // 2 + int(radius * math.sin(angle))
        elif label == "PREDATORY_GAIT":
            # Slow deliberate movement toward a corner
            x += 2
            y += 1
        else:
            x += random.randint(-4, 4)
            y += random.randint(-4, 4)

        # Clamp to frame
        x = int(np.clip(x, 20, FRAME_WIDTH - 20))
        y = int(np.clip(y, 20, FRAME_HEIGHT - 20))

        # Draw subject (person silhouette approximation)
        cv2.ellipse(frame, (x, y - 15), (8, 10), 0, 0, 360, (200, 150, 100), -1)  # head
        cv2.ellipse(frame, (x, y + 5), (10, 18), 0, 0, 360, (180, 130, 80), -1)   # body

        # Timestamp overlay
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"CAM-01  {ts}  FRAME:{i:04d}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        writer.write(frame)

    writer.release()
    return output_path


def generate_all_synthetic_videos(n_per_class: int = 5) -> List[Path]:
    """Generate sample videos for each anomaly class."""
    SYNTHETIC_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    for cls in ANOMALY_CLASSES:
        for i in range(n_per_class):
            out = SYNTHETIC_VIDEO_DIR / cls / f"sample_{i:03d}.mp4"
            if not out.exists():
                generate_synthetic_video(out, label=cls, n_frames=TEMPORAL_WINDOW * 4)
            generated.append(out)
    print(f"[Synthetic] Generated {len(generated)} videos in {SYNTHETIC_VIDEO_DIR}")
    return generated


# ?????????????????????????????????????????????????????????????
# SYNTHETIC GIS DATA
# ?????????????????????????????????????????????????????????????

def generate_synthetic_gis_trajectories(n_subjects: int = 20) -> List[Dict]:
    """
    Generate fake subject trajectories around a simulated urban grid.
    Returns list of trajectory dicts with GPS coordinates and anomaly flags.
    """
    base_lat, base_lon = 37.7749, -122.4194  # San Francisco center

    trajectories = []
    for subj_id in range(n_subjects):
        is_anomalous = subj_id % 5 == 0  # 20% anomalous subjects
        points = []

        lat = base_lat + random.uniform(-0.02, 0.02)
        lon = base_lon + random.uniform(-0.02, 0.02)
        start_time = datetime.now() - timedelta(minutes=random.randint(5, 60))

        for step in range(random.randint(20, 80)):
            ts = start_time + timedelta(seconds=step * 10)

            if is_anomalous:
                # Deviate toward a "target" location (simulated victim route)
                lat += random.uniform(-0.0002, 0.0005)
                lon += random.uniform(-0.0002, 0.0003)
            else:
                lat += random.uniform(-0.0003, 0.0003)
                lon += random.uniform(-0.0003, 0.0003)

            points.append({
                "timestamp": ts.isoformat(),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "speed_kmh": round(random.uniform(0.5, 6.0), 2),
            })

        trajectories.append({
            "subject_id": f"SUBJ-{subj_id:04d}",
            "is_anomalous": is_anomalous,
            "anomaly_type": "ROUTE_DEVIATION" if is_anomalous else "NONE",
            "trajectory": points,
        })

    SYNTHETIC_GIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTHETIC_GIS_FILE, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"[Synthetic] GIS trajectories -> {SYNTHETIC_GIS_FILE}")
    return trajectories


# ?????????????????????????????????????????????????????????????
# SYNTHETIC OSINT / DIGITAL FOOTPRINT DATA
# ?????????????????????????????????????????????????????????????

NORMAL_QUERIES = [
    "best pizza near me", "weather forecast tomorrow",
    "how to fix a leaky faucet", "movie showtimes tonight",
    "python tutorial beginners", "gym workout plan",
    "best coffee shops downtown", "car insurance quotes",
    "how to cook pasta carbonara", "cheap flights to dubai",
]

PREDATORY_QUERIES = [
    "how to follow someone without being noticed",
    "best parks near schools late at night",
    "how to disable a security camera without being seen",
    "anonymous browsing methods tor vpn",
    "how to track someone's location without them knowing",
    "self-defense tools legal to carry",
    "how to approach someone from behind quietly",
    "areas with low police presence",
    "how to avoid facial recognition cameras",
    "predatory behavior psychological patterns",
]


def generate_synthetic_osint(n_subjects: int = 20) -> List[Dict]:
    """Generate simulated digital footprint data for subjects."""
    footprints = []
    for subj_id in range(n_subjects):
        is_flagged = subj_id % 5 == 0

        if is_flagged:
            queries = random.sample(PREDATORY_QUERIES, k=random.randint(4, 8))
            queries += random.sample(NORMAL_QUERIES, k=random.randint(2, 4))
        else:
            queries = random.sample(NORMAL_QUERIES, k=random.randint(5, 10))

        random.shuffle(queries)

        footprints.append({
            "subject_id": f"SUBJ-{subj_id:04d}",
            "search_history": queries,
            "forum_posts": [
                f"Looking for advice on {random.choice(queries[:3])}"
            ] if is_flagged else [],
            "is_flagged_ground_truth": is_flagged,
        })

    SYNTHETIC_OSINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNTHETIC_OSINT_FILE, "w") as f:
        json.dump(footprints, f, indent=2)
    print(f"[Synthetic] OSINT footprints -> {SYNTHETIC_OSINT_FILE}")
    return footprints


# ?????????????????????????????????????????????????????????????
# SYNTHETIC POSE SEQUENCES (for CNN-LSTM testing without video)
# ?????????????????????????????????????????????????????????????

def generate_synthetic_pose_sequence(
    label: str = "NORMAL",
    n_frames: int = TEMPORAL_WINDOW
) -> torch.Tensor:
    """
    Generate a fake pose landmark sequence tensor.
    Shape: (n_frames, 33 landmarks * 3 coords) = (16, 99)
    """
    base = np.random.randn(33 * 3)

    frames = []
    for i in range(n_frames):
        if label == "NORMAL":
            delta = np.random.randn(33 * 3) * 0.01
        elif label == "AGGRESSIVE_POSTURE":
            delta = np.random.randn(33 * 3) * 0.05
        elif label == "LOITERING":
            angle = (i / n_frames) * 2 * math.pi
            delta = np.array([math.sin(angle) * 0.03] * (33 * 3))
        else:
            delta = np.random.randn(33 * 3) * 0.02
        base = base + delta
        frames.append(base.copy())

    return torch.tensor(np.stack(frames), dtype=torch.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("ARGUS - Synthetic Data Generator")
    print("=" * 60)
    generate_all_synthetic_videos(n_per_class=3)
    generate_synthetic_gis_trajectories(n_subjects=15)
    generate_synthetic_osint(n_subjects=15)
    print("\n[DONE] All synthetic data generated.")
    print(f"  Videos  -> {SYNTHETIC_VIDEO_DIR}")
    print(f"  GIS     -> {SYNTHETIC_GIS_FILE}")
    print(f"  OSINT   -> {SYNTHETIC_OSINT_FILE}")

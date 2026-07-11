"""
ARGUS ML Engine - Dataset Loaders
Handles UCF-Crime, CASME II, DCSASS, and SF Crime datasets.
Includes class balancing, missing-data handling, and frame validation.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Import config (relative)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    UCF_CRIME_ROOT, CASME2_ROOT, SF_CRIME_CSV,
    TEMPORAL_WINDOW, FRAME_WIDTH, FRAME_HEIGHT,
    ANOMALY_CLASSES, DEVICE
)

logger = logging.getLogger("ARGUS.DataLoaders")

# ?????????????????????????????????????????????????????????????
# UCF-CRIME DATASET LOADER
# ?????????????????????????????????????????????????????????????

# Official UCF-Crime 13 anomaly classes + Normal
UCF_CRIME_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
    "Normal"
]

# Mapping to ARGUS internal anomaly classes
UCF_TO_ARGUS = {
    "Abuse":          "AGGRESSIVE_POSTURE",
    "Arrest":         "AGGRESSIVE_POSTURE",
    "Arson":          "CONCEALMENT",
    "Assault":        "AGGRESSIVE_POSTURE",
    "Burglary":       "CONCEALMENT",
    "Explosion":      "AGGRESSIVE_POSTURE",
    "Fighting":       "AGGRESSIVE_POSTURE",
    "RoadAccidents":  "NORMAL",
    "Robbery":        "RAPID_APPROACH",
    "Shooting":       "AGGRESSIVE_POSTURE",
    "Shoplifting":    "CONCEALMENT",
    "Stealing":       "PREDATORY_GAIT",
    "Vandalism":      "LOITERING",
    "Normal":         "NORMAL",
}


class UCFCrimeDataset(Dataset):
    """
    Loads UCF-Crime surveillance videos as temporal frame sequences.
    
    Expected directory structure:
        ucf_crime/
            Abuse/
                Abuse001_x264.mp4
                ...
            Normal_Videos/
                Normal_Videos001_x264.mp4
                ...
    """

    def __init__(
        self,
        root: Path = UCF_CRIME_ROOT,
        split: str = "train",
        temporal_window: int = TEMPORAL_WINDOW,
        transform=None,
        max_samples_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.temporal_window = temporal_window
        self.transform = transform or self._default_transform()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(ANOMALY_CLASSES)

        self.samples: List[Tuple[Path, str]] = []
        self._load_dataset(max_samples_per_class)

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load_dataset(self, max_samples_per_class: Optional[int]):
        """Walk UCF-Crime directory and build sample list."""
        if not self.root.exists():
            logger.warning(
                f"UCF-Crime root not found at {self.root}. "
                "Use synthetic data for testing. Run: python data/synthetic.py"
            )
            return

        for cls_name in UCF_CRIME_CLASSES:
            # Handle 'Normal' class which may have a different folder name
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                cls_dir = self.root / f"{cls_name}_Videos"
            if not cls_dir.exists():
                continue

            videos = list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi"))
            argus_label = UCF_TO_ARGUS.get(cls_name, "NORMAL")

            if max_samples_per_class:
                videos = videos[:max_samples_per_class]

            # 80/20 train/val split
            split_idx = int(len(videos) * 0.8)
            if self.split == "train":
                videos = videos[:split_idx]
            else:
                videos = videos[split_idx:]

            for v in videos:
                self.samples.append((v, argus_label))

        logger.info(f"UCF-Crime [{self.split}]: {len(self.samples)} samples loaded")
        self._log_class_distribution()

    def _log_class_distribution(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        logger.info(f"Class distribution: {dict(counts)}")

    def _extract_frames(self, video_path: Path) -> Optional[np.ndarray]:
        """Extract TEMPORAL_WINDOW evenly-spaced frames from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.temporal_window:
            cap.release()
            return None

        # Sample evenly across the video
        indices = np.linspace(0, total_frames - 1, self.temporal_window, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                # Interpolate missing frame with last valid frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frames.append(frame)

        cap.release()
        return np.stack(frames)  # (T, H, W, C)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)

        if frames is None:
            # Return zero tensor for corrupt samples
            frames_tensor = torch.zeros(self.temporal_window, 3, 224, 224)
        else:
            # Apply transforms to each frame
            frame_tensors = [self.transform(f) for f in frames]
            frames_tensor = torch.stack(frame_tensors)  # (T, C, H, W)

        label_idx = self.label_encoder.transform([label])[0]
        return frames_tensor, torch.tensor(label_idx, dtype=torch.long)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler to handle class imbalance."""
        from collections import Counter
        label_counts = Counter(label for _, label in self.samples)
        total = len(self.samples)
        weights = []
        for _, label in self.samples:
            weight = total / (len(label_counts) * label_counts[label])
            weights.append(weight)
        return WeightedRandomSampler(weights, num_samples=total, replacement=True)


# ?????????????????????????????????????????????????????????????
# CASME II - MICRO-EXPRESSION DATASET LOADER
# ?????????????????????????????????????????????????????????????

CASME2_EMOTION_CLASSES = [
    "disgust", "fear", "happiness", "repression",
    "sadness", "surprise", "tense", "others"
]

# Map micro-expressions to high-stress indicator (binary)
CASME2_STRESS_LABELS = {
    "disgust":    1,
    "fear":       1,
    "repression": 1,
    "tense":      1,
    "sadness":    0,
    "happiness":  0,
    "surprise":   0,
    "others":     0,
}


class CASME2Dataset(Dataset):
    """
    Loads CASME II micro-expression image sequences.
    
    Expected structure:
        casme2/
            sub01/
                EP02_01f/
                    img001.jpg
                    img002.jpg
                    ...
            coding.xlsx  # Label file
    """

    def __init__(
        self,
        root: Path = CASME2_ROOT,
        split: str = "train",
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
        self.samples: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        if not self.root.exists():
            logger.warning(f"CASME II root not found at {self.root}. Using synthetic fallback.")
            return

        coding_file = self.root / "coding.xlsx"
        if not coding_file.exists():
            coding_file = self.root / "CASME2-coding-20140508.xlsx"

        if not coding_file.exists():
            logger.warning("CASME II coding file not found. Using folder names as labels.")
            self._load_from_folders()
            return

        df = pd.read_excel(coding_file)
        for _, row in df.iterrows():
            subject = f"sub{int(row.get('Subject', 0)):02d}"
            sequence = str(row.get('Filename', ''))
            emotion = str(row.get('Emotion', 'others')).lower()

            seq_path = self.root / subject / sequence
            if not seq_path.exists():
                continue

            frames = sorted(seq_path.glob("*.jpg"))
            if len(frames) < 3:
                continue

            self.samples.append({
                "frames": frames,
                "emotion": emotion,
                "stress": CASME2_STRESS_LABELS.get(emotion, 0),
            })

        # 80/20 split
        split_idx = int(len(self.samples) * 0.8)
        self.samples = self.samples[:split_idx] if self.split == "train" else self.samples[split_idx:]
        logger.info(f"CASME II [{self.split}]: {len(self.samples)} sequences")

    def _load_from_folders(self):
        for subject_dir in sorted(self.root.iterdir()):
            if not subject_dir.is_dir():
                continue
            for seq_dir in sorted(subject_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                frames = sorted(seq_dir.glob("*.jpg"))
                if len(frames) < 3:
                    continue
                self.samples.append({
                    "frames": frames,
                    "emotion": "unknown",
                    "stress": 0,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        frames = sample["frames"]

        # Use apex frame (most expressive) ? context
        apex_idx = len(frames) // 2
        selected = frames[max(0, apex_idx-1): apex_idx+2]

        tensors = []
        for fp in selected:
            img = cv2.imread(str(fp))
            if img is None:
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(img))

        # Pad to 3 frames if needed
        while len(tensors) < 3:
            tensors.append(tensors[-1])

        frames_tensor = torch.stack(tensors[:3])  # (3, C, H, W)
        stress_label = torch.tensor(sample["stress"], dtype=torch.long)
        return frames_tensor, stress_label


# ?????????????????????????????????????????????????????????????
# SF CRIME INCIDENTS - SPATIOTEMPORAL LOADER
# ?????????????????????????????????????????????????????????????

class SFCrimeDataset(Dataset):
    """
    Loads SF Crime Incident Reports for GNN hotspot training.
    CSV must have columns: DateTime, Category, Latitude, Longitude
    """

    def __init__(self, csv_path: Path = SF_CRIME_CSV, split: str = "train"):
        self.csv_path = Path(csv_path)
        self.split = split
        self.records: List[Dict] = []
        self._load()

    def _load(self):
        if not self.csv_path.exists():
            logger.warning(f"SF Crime CSV not found at {self.csv_path}. Using synthetic GIS data.")
            return

        df = pd.read_csv(self.csv_path, parse_dates=["DateTime"])
        df = df.dropna(subset=["Latitude", "Longitude"])
        df = df[df["Latitude"].between(-90, 90)]
        df = df[df["Longitude"].between(-180, 180)]

        split_idx = int(len(df) * 0.8)
        df = df.iloc[:split_idx] if self.split == "train" else df.iloc[split_idx:]

        for _, row in df.iterrows():
            self.records.append({
                "lat": float(row["Latitude"]),
                "lon": float(row["Longitude"]),
                "category": str(row.get("Category", "OTHER")),
                "hour": row["DateTime"].hour if pd.notnull(row.get("DateTime")) else 0,
                "dayofweek": row["DateTime"].dayofweek if pd.notnull(row.get("DateTime")) else 0,
            })
        logger.info(f"SF Crime [{self.split}]: {len(self.records)} incidents")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        features = torch.tensor(
            [r["lat"], r["lon"], r["hour"] / 24.0, r["dayofweek"] / 7.0],
            dtype=torch.float32
        )
        return features


# ?????????????????????????????????????????????????????????????
# DATALOADER FACTORY
# ?????????????????????????????????????????????????????????????

def get_ucf_crime_loaders(batch_size: int = 8, balanced: bool = True):
    train_ds = UCFCrimeDataset(split="train")
    val_ds = UCFCrimeDataset(split="val")

    sampler = train_ds.get_weighted_sampler() if balanced and len(train_ds) > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def get_casme2_loaders(batch_size: int = 32):
    train_ds = CASME2Dataset(split="train")
    val_ds = CASME2Dataset(split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

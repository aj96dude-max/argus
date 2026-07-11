"""
ARGUS - Module B: Temporal & Gait Analysis
CNN-LSTM model for classifying anomalous behavior from pose sequences.
Analyzes 16-frame sliding windows to detect predatory gait patterns.
"""

from __future__ import annotations
import sys
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TEMPORAL_WINDOW, POSE_LANDMARKS, CNN_FEATURE_DIM,
    LSTM_HIDDEN_DIM, LSTM_LAYERS, LSTM_DROPOUT,
    NUM_ANOMALY_CLASSES, ANOMALY_CLASSES, ANOMALY_THRESHOLD,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, WEIGHT_DECAY,
    CHECKPOINTS_DIR, DEVICE
)
from models.spatial import SubjectSpatialData

logger = logging.getLogger("ARGUS.ModuleB")

INPUT_DIM = POSE_LANDMARKS * 3  # 99 - flattened 3D pose landmarks


# ?????????????????????????????????????????????????????????????
# CNN ENCODER
# ?????????????????????????????????????????????????????????????

class PoseCNNEncoder(nn.Module):
    """
    1D CNN that extracts spatial features from a single pose frame.
    Input:  (batch, INPUT_DIM) = (B, 99)
    Output: (batch, CNN_FEATURE_DIM) = (B, 256)
    """

    def __init__(self, input_dim: int = INPUT_DIM, feature_dim: int = CNN_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # Reshape to (B, 1, input_dim) for Conv1d
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ?????????????????????????????????????????????????????????????
# CNN-LSTM CLASSIFIER
# ?????????????????????????????????????????????????????????????

class CNNLSTMClassifier(nn.Module):
    """
    Module B: CNN-LSTM architecture for temporal gait/behavior classification.

    Architecture:
        1. CNN Encoder processes each frame's pose vector independently
        2. LSTM analyzes the temporal sequence of CNN features
        3. Classification head predicts anomaly class

    Input:  (batch, temporal_window, pose_dim) = (B, 16, 99)
    Output: (batch, num_classes) = (B, 7)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        cnn_feature_dim: int = CNN_FEATURE_DIM,
        lstm_hidden: int = LSTM_HIDDEN_DIM,
        lstm_layers: int = LSTM_LAYERS,
        num_classes: int = NUM_ANOMALY_CLASSES,
        dropout: float = LSTM_DROPOUT,
    ):
        super().__init__()
        self.temporal_window = TEMPORAL_WINDOW
        self.cnn_encoder = PoseCNNEncoder(input_dim, cnn_feature_dim)

        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,  # Bidirectional for better context
        )

        lstm_output_dim = lstm_hidden * 2  # Bidirectional

        # Attention mechanism over time steps
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        # Anomaly confidence head (binary: normal vs anomalous)
        self.anomaly_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, pose_dim)
        Returns dict with 'logits', 'anomaly_score', 'attention_weights'
        """
        B, T, D = x.shape

        # Apply CNN encoder to each time step
        x_flat = x.view(B * T, D)
        cnn_features = self.cnn_encoder(x_flat)
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, CNN_FEATURE_DIM)

        # LSTM temporal processing
        lstm_out, (h_n, _) = self.lstm(cnn_features)  # lstm_out: (B, T, lstm_hidden*2)

        # Attention-weighted pooling
        attn_scores = self.attention(lstm_out)          # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)    # (B, T, 1)
        context = (attn_weights * lstm_out).sum(dim=1)  # (B, lstm_hidden*2)

        logits = self.classifier(context)
        anomaly_score = self.anomaly_head(context).squeeze(-1)

        return {
            "logits": logits,
            "anomaly_score": anomaly_score,
            "attention_weights": attn_weights.squeeze(-1),
            "context": context,
        }


# ?????????????????????????????????????????????????????????????
# TEMPORAL ANALYZER (Inference + Sliding Window)
# ?????????????????????????????????????????????????????????????

class TemporalAnalyzer:
    """
    Module B runtime wrapper.
    Maintains per-subject sliding windows and runs CNN-LSTM inference.
    """

    def __init__(self, model_checkpoint: Optional[Path] = None):
        self.model = CNNLSTMClassifier().to(DEVICE)
        self.model.eval()

        # Per-subject sliding window buffers
        self._windows: Dict[str, deque] = {}

        if model_checkpoint and model_checkpoint.exists():
            self._load_checkpoint(model_checkpoint)
            logger.info(f"CNN-LSTM loaded from {model_checkpoint}")
        else:
            logger.warning(
                "No checkpoint found. Model will run with random weights. "
                "Train with: python models/temporal.py --train"
            )

    def _load_checkpoint(self, path: Path):
        checkpoint = torch.load(str(path), map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _get_window(self, subject_id: str) -> deque:
        if subject_id not in self._windows:
            self._windows[subject_id] = deque(maxlen=TEMPORAL_WINDOW)
        return self._windows[subject_id]

    def update(self, subject: SubjectSpatialData) -> Optional[Dict]:
        """
        Feed one frame of pose data for a subject.
        Returns classification result when window is full, else None.
        """
        window = self._get_window(subject.subject_id)
        window.append(subject.pose_flat)

        if len(window) < TEMPORAL_WINDOW:
            return None  # Window not yet full

        return self._infer(subject.subject_id, np.stack(list(window)))

    @torch.no_grad()
    def _infer(self, subject_id: str, pose_sequence: np.ndarray) -> Dict:
        """Run CNN-LSTM on a (TEMPORAL_WINDOW, 99) pose sequence."""
        x = torch.tensor(pose_sequence, dtype=torch.float32)
        x = x.unsqueeze(0).to(DEVICE)  # (1, T, D)

        output = self.model(x)

        logits = output["logits"][0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        anomaly_score = float(output["anomaly_score"][0].cpu().numpy())
        pred_class_idx = int(probs.argmax())
        pred_class = ANOMALY_CLASSES[pred_class_idx]
        confidence = float(probs[pred_class_idx])

        result = {
            "subject_id": subject_id,
            "predicted_class": pred_class,
            "class_confidence": confidence,
            "anomaly_score": anomaly_score,
            "all_class_probs": {cls: float(p) for cls, p in zip(ANOMALY_CLASSES, probs)},
            "is_anomalous": anomaly_score > ANOMALY_THRESHOLD,
        }

        if result["is_anomalous"]:
            logger.warning(
                f"[ModuleB] ANOMALY DETECTED | Subject: {subject_id} | "
                f"Class: {pred_class} | Score: {anomaly_score:.3f}"
            )

        return result


# ?????????????????????????????????????????????????????????????
# TRAINING LOOP
# ?????????????????????????????????????????????????????????????

def train_cnn_lstm():
    """Full training loop using UCF-Crime dataset or synthetic fallback."""
    from torch.utils.data import TensorDataset, DataLoader
    from data.synthetic import generate_synthetic_pose_sequence

    print("[Module B] Starting CNN-LSTM Training...")
    model = CNNLSTMClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # Try to load UCF-Crime; fall back to synthetic
    try:
        from data.loaders import get_ucf_crime_loaders
        train_loader, val_loader = get_ucf_crime_loaders(BATCH_SIZE)
        if len(train_loader) == 0:
            raise ValueError("Empty loader")
        use_real_data = True
        print("[Module B] Using UCF-Crime dataset")
    except Exception:
        print("[Module B] UCF-Crime not available. Using synthetic pose sequences.")
        use_real_data = False

    if not use_real_data:
        # Build synthetic dataset from pose sequences
        X_list, y_list = [], []
        for cls_idx, cls in enumerate(ANOMALY_CLASSES):
            for _ in range(50):
                seq = generate_synthetic_pose_sequence(cls, TEMPORAL_WINDOW)
                X_list.append(seq)
                y_list.append(cls_idx)

        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.long)
        dataset = TensorDataset(X, y)
        split = int(len(dataset) * 0.8)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # ?? TRAIN ?????????????????????????????????????????????
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in train_loader:
            if use_real_data:
                frames, labels = batch
                # Flatten spatial dims: (B, T, C, H, W) -> need pose vectors
                # For real UCF-Crime, we'd need spatial.py preprocessing first
                # Here we use mean pooling of frame pixels as proxy
                B, T, C, H, W = frames.shape
                pose_proxy = frames.view(B, T, -1).mean(dim=-1, keepdim=True)
                pose_proxy = pose_proxy.expand(-1, -1, INPUT_DIM).float()
                x, y_true = pose_proxy.to(DEVICE), labels.to(DEVICE)
            else:
                x, y_true = batch[0].to(DEVICE), batch[1].to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out["logits"], y_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = out["logits"].argmax(dim=-1)
            train_correct += (preds == y_true).sum().item()
            train_total += len(y_true)

        scheduler.step()
        train_acc = train_correct / max(train_total, 1)

        # ?? VALIDATE ??????????????????????????????????????????
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if use_real_data:
                    frames, labels = batch
                    B, T, C, H, W = frames.shape
                    pose_proxy = frames.view(B, T, -1).mean(dim=-1, keepdim=True)
                    pose_proxy = pose_proxy.expand(-1, -1, INPUT_DIM).float()
                    x, y_true = pose_proxy.to(DEVICE), labels.to(DEVICE)
                else:
                    x, y_true = batch[0].to(DEVICE), batch[1].to(DEVICE)

                out = model(x)
                loss = criterion(out["logits"], y_true)
                val_loss += loss.item()
                preds = out["logits"].argmax(dim=-1)
                val_correct += (preds == y_true).sum().item()
                val_total += len(y_true)

        val_acc = val_correct / max(val_total, 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.3f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = CHECKPOINTS_DIR / "cnn_lstm_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, str(ckpt_path))
            print(f"  [OK] Checkpoint saved -> {ckpt_path}")

    print("[Module B] Training complete.")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARGUS Module B - CNN-LSTM")
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--test", action="store_true", help="Run inference test")
    args = parser.parse_args()

    if args.train:
        train_cnn_lstm()
    elif args.test:
        from data.synthetic import generate_synthetic_pose_sequence
        analyzer = TemporalAnalyzer()
        print("\n[Module B] Inference Test with synthetic pose sequences:")
        for cls in ANOMALY_CLASSES:
            for t in range(TEMPORAL_WINDOW + 2):
                fake_subj = type("FakeSubject", (), {
                    "subject_id": f"TEST-{cls}",
                    "pose_flat": generate_synthetic_pose_sequence(cls, 1)[0].numpy()
                })()
                result = analyzer.update(fake_subj)
                if result:
                    print(f"  {cls:25s} -> Predicted: {result['predicted_class']:25s} "
                          f"| Anomaly Score: {result['anomaly_score']:.3f}")
                    break
    else:
        print("Usage: python models/temporal.py --train | --test")

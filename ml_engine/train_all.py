"""
ARGUS - Training Orchestrator
Trains all ML models in the correct dependency order.
Run this once after setting up your datasets.

Usage:
    python train_all.py                   # Train all models
    python train_all.py --module B        # Train only Module B (CNN-LSTM)
    python train_all.py --synthetic       # Force synthetic data (no real datasets)
    python train_all.py --epochs 20       # Override epoch count
"""

import sys
import time
import argparse
import logging
from pathlib import Path

# Ensure ml_engine root is on path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ARGUS.TrainAll")


def print_banner(text: str):
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def train_module_b(synthetic: bool, epochs: int):
    """Train CNN-LSTM (Module B) - core anomaly classifier."""
    print_banner("MODULE B - CNN-LSTM TEMPORAL CLASSIFIER")
    from config import EPOCHS
    import config
    if epochs:
        config.EPOCHS = epochs

    from models.temporal import train_cnn_lstm
    if synthetic:
        logger.info("Forcing synthetic data mode for Module B")
        # Temporarily hide real dataset path
        import config as cfg
        orig = cfg.UCF_CRIME_ROOT
        cfg.UCF_CRIME_ROOT = Path("__nonexistent__")
        model = train_cnn_lstm()
        cfg.UCF_CRIME_ROOT = orig
    else:
        model = train_cnn_lstm()
    return model


def train_module_c(synthetic: bool):
    """Train GNN Route Model (Module C)."""
    print_banner("MODULE C - GNN SPATIOTEMPORAL ROUTE TRACKER")
    import torch
    from config import CHECKPOINTS_DIR, DEVICE, GNN_HIDDEN_DIM, GNN_LAYERS
    from models.spatiotemporal import RouteGNN
    from data.synthetic import generate_synthetic_gis_trajectories
    import numpy as np

    logger.info("Training GNN on synthetic GIS trajectories...")
    trajectories = generate_synthetic_gis_trajectories(n_subjects=50)

    model = RouteGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    EPOCHS_GNN = 30
    for epoch in range(1, EPOCHS_GNN + 1):
        model.train()
        total_loss = 0
        for traj in trajectories:
            points = traj["trajectory"]
            if len(points) < 5:
                continue

            coords = torch.tensor(
                [[p["lat"], p["lon"], 12 / 24.0, 2 / 7.0] for p in points[-20:]],
                dtype=torch.float32,
                device=DEVICE,
            )
            label = torch.tensor(
                [float(traj["is_anomalous"])], dtype=torch.float32, device=DEVICE
            )

            optimizer.zero_grad()
            pred = model(coords)
            loss = criterion(pred.squeeze(), label.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            logger.info(f"GNN Epoch {epoch}/{EPOCHS_GNN} | Loss: {total_loss/len(trajectories):.4f}")

    ckpt = CHECKPOINTS_DIR / "gnn_best.pt"
    torch.save({"gnn_state_dict": model.state_dict()}, str(ckpt))
    logger.info(f"GNN checkpoint saved -> {ckpt}")
    return model


def generate_synthetic_data():
    """Generate all synthetic datasets."""
    print_banner("GENERATING SYNTHETIC DATA")
    from data.synthetic import (
        generate_all_synthetic_videos,
        generate_synthetic_gis_trajectories,
        generate_synthetic_osint,
    )
    generate_all_synthetic_videos(n_per_class=5)
    generate_synthetic_gis_trajectories(n_subjects=30)
    generate_synthetic_osint(n_subjects=30)
    logger.info("Synthetic data generation complete.")


def main():
    parser = argparse.ArgumentParser(description="ARGUS Training Orchestrator")
    parser.add_argument(
        "--module", choices=["B", "C", "all"], default="all",
        help="Which module to train (B=CNN-LSTM, C=GNN, all=both)"
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data even if real datasets are available")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Override training epochs (0 = use config default)")
    parser.add_argument("--data-only", action="store_true",
                        help="Only generate synthetic data, do not train")
    args = parser.parse_args()

    print_banner("ARGUS - TRAINING ORCHESTRATOR")
    logger.info(f"Module: {args.module} | Synthetic: {args.synthetic} | Epochs: {args.epochs or 'default'}")

    start = time.time()

    # Always generate synthetic data first
    generate_synthetic_data()
    if args.data_only:
        logger.info("--data-only flag set. Skipping training.")
        return

    if args.module in ("B", "all"):
        train_module_b(synthetic=args.synthetic, epochs=args.epochs)

    if args.module in ("C", "all"):
        train_module_c(synthetic=args.synthetic)

    elapsed = time.time() - start
    print_banner(f"TRAINING COMPLETE - {elapsed:.1f}s")
    logger.info("All models saved to checkpoints/")
    logger.info("Run: python models/pipeline.py --input synthetic")


if __name__ == "__main__":
    main()

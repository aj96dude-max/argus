"""
ARGUS - Module C: Spatiotemporal Route Tracking
Graph Neural Network (GNN) + Dynamic Mode Decomposition (DMD)
for Pattern of Life analysis, route deviation detection, and hotspot prediction.
"""

from __future__ import annotations
import sys
import math
import logging
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GNN_HIDDEN_DIM, GNN_LAYERS, PATTERN_OF_LIFE_FRAMES,
    ROUTE_DEVIATION_SIGMA, PROXIMITY_THRESHOLD_M,
    HOTSPOT_DECAY_RATE, DEVICE
)

logger = logging.getLogger("ARGUS.ModuleC")

# ?????????????????????????????????????????????????????????????
# GIS UTILITIES
# ?????????????????????????????????????????????????????????????

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two GPS coordinates."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ?????????????????????????????????????????????????????????????
# GNN MODEL
# ?????????????????????????????????????????????????????????????

class GraphConvLayer(nn.Module):
    """Simple graph convolution: aggregates neighbor features via adjacency."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (N, in_dim)   - node features
        adj: (N, N)        - normalized adjacency matrix
        """
        # Aggregate: A * X
        agg = torch.mm(adj, x)
        out = self.linear(agg)
        return F.relu(self.norm(out))


class RouteGNN(nn.Module):
    """
    GNN for trajectory anomaly detection.
    Nodes = GPS waypoints, Edges = path connections weighted by distance/time.

    Input:  (N, 4) - [lat, lon, hour, dayofweek_norm]
    Output: (1,)   - route anomaly score 0?1
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = GNN_HIDDEN_DIM,
        n_layers: int = GNN_LAYERS,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _build_adjacency(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Build normalized adjacency matrix from coordinate proximity.
        Sequential nodes in trajectory get edge weight = 1.
        """
        N = coords.shape[0]
        A = torch.zeros(N, N, device=coords.device)

        # Sequential edges (trajectory path)
        for i in range(N - 1):
            A[i, i+1] = 1.0
            A[i+1, i] = 1.0

        # Self-loops
        A += torch.eye(N, device=coords.device)

        # Row-normalize
        row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return A / row_sum

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 4)"""
        adj = self._build_adjacency(coords)
        x = F.relu(self.input_proj(coords))

        for conv in self.conv_layers:
            x = conv(x, adj)

        # Global mean pooling
        graph_repr = x.mean(dim=0, keepdim=True)
        return self.readout(graph_repr)


# ?????????????????????????????????????????????????????????????
# DYNAMIC MODE DECOMPOSITION (DMD) - HOTSPOT PREDICTION
# ?????????????????????????????????????????????????????????????

class DMDHotspotPredictor:
    """
    Dynamic Mode Decomposition for spatiotemporal hotspot prediction.
    Decomposes a crime density matrix over time to predict future hotspots.
    """

    def __init__(self, grid_size: int = 20, n_modes: int = 5):
        self.grid_size = grid_size
        self.n_modes = n_modes
        self.snapshots: List[np.ndarray] = []
        self._dmd_modes = None
        self._dmd_eigenvalues = None

    def update(self, lat: float, lon: float, lat_range: Tuple, lon_range: Tuple):
        """Add an event point and update the density grid."""
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            return

        grid = self.snapshots[-1].copy() if self.snapshots else np.zeros(
            (self.grid_size, self.grid_size)
        )

        # Apply temporal decay
        grid *= HOTSPOT_DECAY_RATE

        # Map coordinates to grid cell
        row = int((lat - lat_min) / (lat_max - lat_min) * (self.grid_size - 1))
        col = int((lon - lon_min) / (lon_max - lon_min) * (self.grid_size - 1))
        row = np.clip(row, 0, self.grid_size - 1)
        col = np.clip(col, 0, self.grid_size - 1)
        grid[row, col] += 1.0

        self.snapshots.append(grid)

    def fit(self) -> bool:
        """Fit DMD on accumulated snapshots."""
        if len(self.snapshots) < self.n_modes + 2:
            return False

        # Build snapshot matrices X and X'
        data = np.array(self.snapshots)
        flat = data.reshape(len(data), -1).T  # (features, time)

        X = flat[:, :-1]
        X_prime = flat[:, 1:]

        # SVD-based DMD
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            r = min(self.n_modes, len(s))
            U_r = U[:, :r]
            S_r = np.diag(s[:r])
            V_r = Vt[:r, :].T

            A_tilde = U_r.T @ X_prime @ V_r @ np.linalg.inv(S_r)
            eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

            self._dmd_modes = X_prime @ V_r @ np.linalg.inv(S_r) @ eigenvectors
            self._dmd_eigenvalues = eigenvalues
            return True
        except np.linalg.LinAlgError:
            return False

    def predict_hotspots(self, steps_ahead: int = 3) -> Optional[np.ndarray]:
        """
        Predict hotspot density grid N steps into the future.
        Returns (grid_size, grid_size) density array.
        """
        if self._dmd_modes is None or not self.fit():
            if self.snapshots:
                return self.snapshots[-1]
            return None

        current = self.snapshots[-1].flatten()
        b = np.linalg.lstsq(self._dmd_modes, current, rcond=None)[0]

        future = np.zeros_like(current, dtype=complex)
        for k in range(len(b)):
            mode_k = self._dmd_modes[:, k]
            eig_k = self._dmd_eigenvalues[k]
            future += b[k] * mode_k * (eig_k ** steps_ahead)

        return np.abs(future).reshape(self.grid_size, self.grid_size)


# ?????????????????????????????????????????????????????????????
# PATTERN OF LIFE TRACKER
# ?????????????????????????????????????????????????????????????

@dataclass
class SubjectRoute:
    """Trajectory and Pattern of Life state for one subject."""
    subject_id: str
    trajectory: deque = field(default_factory=lambda: deque(maxlen=500))
    baseline_positions: List[Tuple[float, float]] = field(default_factory=list)
    baseline_established: bool = False
    frames_observed: int = 0
    last_known_lat: float = 0.0
    last_known_lon: float = 0.0

    def add_point(self, lat: float, lon: float):
        self.trajectory.append((lat, lon))
        self.last_known_lat = lat
        self.last_known_lon = lon
        self.frames_observed += 1

        if not self.baseline_established and self.frames_observed >= PATTERN_OF_LIFE_FRAMES:
            self.baseline_positions = list(self.trajectory)
            self.baseline_established = True
            logger.debug(f"[ModuleC] Pattern of Life baseline established for {self.subject_id}")

    @property
    def baseline_centroid(self) -> Tuple[float, float]:
        if not self.baseline_positions:
            return (0.0, 0.0)
        lats = [p[0] for p in self.baseline_positions]
        lons = [p[1] for p in self.baseline_positions]
        return (np.mean(lats), np.mean(lons))

    @property
    def baseline_std(self) -> float:
        if len(self.baseline_positions) < 2:
            return 0.0
        centroid = self.baseline_centroid
        distances = [
            haversine_distance(p[0], p[1], centroid[0], centroid[1])
            for p in self.baseline_positions
        ]
        return float(np.std(distances))


class SpatiotemporalTracker:
    """
    Module C runtime orchestrator.
    Tracks subjects' GPS trajectories, establishes Pattern of Life baselines,
    detects route deviations using GNN, and predicts hotspots via DMD.
    """

    def __init__(self, model_checkpoint: Optional[Path] = None):
        self.gnn = RouteGNN().to(DEVICE)
        self.gnn.eval()
        self.dmd = DMDHotspotPredictor()
        self._routes: Dict[str, SubjectRoute] = {}

        if model_checkpoint and model_checkpoint.exists():
            checkpoint = torch.load(str(model_checkpoint), map_location=DEVICE)
            self.gnn.load_state_dict(checkpoint["gnn_state_dict"])
            logger.info(f"GNN loaded from {model_checkpoint}")
        else:
            logger.warning("No GNN checkpoint. Using untrained model.")

        # Geographic bounds (default: San Francisco)
        self.lat_range = (37.70, 37.83)
        self.lon_range = (-122.52, -122.36)

    def update_subject(
        self, subject_id: str, lat: float, lon: float,
        hour: int = 0, dayofweek: int = 0
    ) -> Dict:
        """
        Feed one GPS observation for a subject.
        Returns spatiotemporal analysis results.
        """
        if subject_id not in self._routes:
            self._routes[subject_id] = SubjectRoute(subject_id=subject_id)

        route = self._routes[subject_id]
        route.add_point(lat, lon)

        # Update DMD density grid
        self.dmd.update(lat, lon, self.lat_range, self.lon_range)

        result = {
            "subject_id": subject_id,
            "route_deviation_score": 0.0,
            "route_deviation_sigma": 0.0,
            "proximity_alert": False,
            "gnn_anomaly_score": 0.0,
            "baseline_established": route.baseline_established,
            "hotspot_risk": 0.0,
        }

        # ?? Route Deviation Analysis ??????????????????????????
        if route.baseline_established and len(route.trajectory) > 5:
            centroid = route.baseline_centroid
            current_dist = haversine_distance(lat, lon, centroid[0], centroid[1])
            sigma = route.baseline_std

            if sigma > 0:
                deviation_sigmas = current_dist / sigma
                result["route_deviation_sigma"] = deviation_sigmas
                result["route_deviation_score"] = float(
                    np.clip(deviation_sigmas / (ROUTE_DEVIATION_SIGMA * 3), 0, 1)
                )
                if deviation_sigmas > ROUTE_DEVIATION_SIGMA:
                    logger.warning(
                        f"[ModuleC] Route deviation: {subject_id} "
                        f"| {deviation_sigmas:.1f}? from baseline"
                    )

        # ?? GNN Route Anomaly Score ???????????????????????????
        if len(route.trajectory) >= 5:
            result["gnn_anomaly_score"] = self._run_gnn(
                list(route.trajectory)[-20:], hour, dayofweek
            )

        # ?? Proximity Check (vs all other subjects) ???????????
        for other_id, other_route in self._routes.items():
            if other_id == subject_id:
                continue
            dist = haversine_distance(
                lat, lon,
                other_route.last_known_lat, other_route.last_known_lon
            )
            if dist < PROXIMITY_THRESHOLD_M:
                result["proximity_alert"] = True
                result["proximity_subject"] = other_id
                result["proximity_distance_m"] = dist
                break

        # ?? Hotspot Risk ??????????????????????????????????????
        hotspot_grid = self.dmd.predict_hotspots(steps_ahead=2)
        if hotspot_grid is not None:
            # Map current position to grid
            row = int(
                (lat - self.lat_range[0]) / (self.lat_range[1] - self.lat_range[0])
                * (self.dmd.grid_size - 1)
            )
            col = int(
                (lon - self.lon_range[0]) / (self.lon_range[1] - self.lon_range[0])
                * (self.dmd.grid_size - 1)
            )
            row = np.clip(row, 0, self.dmd.grid_size - 1)
            col = np.clip(col, 0, self.dmd.grid_size - 1)
            max_density = hotspot_grid.max() or 1e-8
            result["hotspot_risk"] = float(hotspot_grid[row, col] / max_density)

        return result

    @torch.no_grad()
    def _run_gnn(
        self, trajectory: List[Tuple[float, float]],
        hour: int, dayofweek: int
    ) -> float:
        """Run GNN on a trajectory window."""
        try:
            coords = np.array([
                [lat, lon, hour / 24.0, dayofweek / 7.0]
                for lat, lon in trajectory
            ], dtype=np.float32)
            x = torch.tensor(coords, device=DEVICE)
            score = self.gnn(x)
            return float(score.cpu().item())
        except Exception as e:
            logger.debug(f"GNN inference error: {e}")
            return 0.0

    def get_all_routes_summary(self) -> List[Dict]:
        """Return summary of all tracked subject routes."""
        return [
            {
                "subject_id": r.subject_id,
                "total_points": len(r.trajectory),
                "baseline_established": r.baseline_established,
                "baseline_std_m": r.baseline_std,
                "last_lat": r.last_known_lat,
                "last_lon": r.last_known_lon,
            }
            for r in self._routes.values()
        ]


if __name__ == "__main__":
    from data.synthetic import generate_synthetic_gis_trajectories
    import json

    print("[Module C] Running GNN spatiotemporal test...")
    tracker = SpatiotemporalTracker()
    trajectories = generate_synthetic_gis_trajectories(n_subjects=5)

    for traj in trajectories:
        subj_id = traj["subject_id"]
        for point in traj["trajectory"]:
            result = tracker.update_subject(
                subject_id=subj_id,
                lat=point["lat"],
                lon=point["lon"],
            )

        print(f"\nSubject: {subj_id} (GT anomalous: {traj['is_anomalous']})")
        print(f"  Route Deviation: {result['route_deviation_sigma']:.2f}?")
        print(f"  GNN Anomaly Score: {result['gnn_anomaly_score']:.3f}")
        print(f"  Hotspot Risk: {result['hotspot_risk']:.3f}")
        print(f"  Proximity Alert: {result['proximity_alert']}")

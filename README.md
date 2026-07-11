# PROJECT ARGUS
### Predictive Threat & Spatiotemporal Anomaly Detection System
**Lead Architect:** Abdullah Javed (vagabond)

---

## System Overview

ARGUS is a multi-modal AI framework fusing:
- **Computer Vision** (YOLOv8 + MediaPipe) → real-time human detection & 3D skeletal pose
- **Temporal Analysis** (CNN-LSTM) → gait pattern classification from pose sequences
- **Spatiotemporal Tracking** (GNN + DMD) → route deviation, Pattern of Life, hotspot prediction
- **NLP OSINT** (RoBERTa) → digital footprint predatory language analysis
- **Agentic Synthesis** → automated Predictive Threat Assessment Reports
- **Command Dashboard** → professional law enforcement browser UI

```
┌─────────────────────────────────────────────────────────────┐
│                    ARGUS SYSTEM TOPOLOGY                    │
├──────────────────────────┬──────────────────────────────────┤
│   ML VISION ENGINE       │   AGENTIC SYNTHESIS ENGINE       │
│   (ml_engine/)           │   (agentic_engine/)              │
│                          │                                  │
│  [Video Feed]            │  [SSE Listener Agent]            │
│       ↓                  │          ↓                       │
│  [YOLOv8 Detection]      │  [OSINT NLP Agent]               │
│       ↓                  │     (RoBERTa)                    │
│  [MediaPipe 3D Pose]     │          ↓                       │
│       ↓                  │  [Threat Synthesizer]            │
│  [CNN-LSTM Temporal] ────┼──→ FastAPI /alert POST           │
│       ↓                  │          ↓                       │
│  [GNN Route Tracker]     │  [ARGUS Dashboard UI]            │
│       ↓                  │     http://localhost:5000        │
│  [FastAPI Sensor Hub]────┼──→ SSE Stream → Browser          │
│   :8001                  │                                  │
└──────────────────────────┴──────────────────────────────────┘
```

---

## Directory Structure

```
argus/
├── ml_engine/                   ← ML Vision & Processing Engine
│   ├── config.py                ← Central configuration (CUDA, paths, hyperparams)
│   ├── requirements.txt
│   ├── checkpoints/             ← Model weights saved here
│   ├── data/
│   │   ├── loaders.py           ← UCF-Crime, CASME II, SF Crime dataset loaders
│   │   └── synthetic.py        ← Synthetic video + GIS + OSINT data generator
│   ├── models/
│   │   ├── spatial.py           ← Module A: YOLOv8 + MediaPipe
│   │   ├── temporal.py          ← Module B: CNN-LSTM
│   │   ├── spatiotemporal.py    ← Module C: GNN + DMD
│   │   └── pipeline.py          ← Unified pipeline orchestrator
│   └── api/
│       └── sensor_hub.py        ← FastAPI Sensor Hub (port 8001)
│
└── agentic_engine/              ← Agentic Synthesis & Alert Engine
    ├── requirements.txt
    ├── agents/
    │   ├── listener.py           ← SSE listener agent
    │   ├── osint_nlp.py          ← Module D: RoBERTa OSINT agent
    │   └── synthesizer.py        ← Threat Assessment Report generator
    └── dashboard/
        ├── app.py                ← Flask dashboard (port 5000)
        ├── templates/
        │   └── index.html        ← ARGUS Command Dashboard UI
        └── static/
            ├── style.css
            └── argus.js
```

---

## Quick Start

### Step 1 — Install Dependencies

```powershell
# ML Engine
cd argus/ml_engine
pip install -r requirements.txt

# For PyTorch with CUDA (adjust version for your CUDA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For torch-geometric:
pip install torch-geometric

# Agentic Engine
cd ../agentic_engine
pip install -r requirements.txt
```

### Step 2 — Generate Synthetic Test Data

```powershell
cd argus/ml_engine
python data/synthetic.py
```

This creates:
- `data/synthetic_videos/` — sample surveillance videos per anomaly class
- `data/synthetic_gis.json` — simulated GPS trajectory data
- `data/synthetic_osint.json` — simulated digital footprint records

### Step 3 — Start the FastAPI Sensor Hub

```powershell
cd argus/ml_engine/api
python sensor_hub.py
# API running at: http://localhost:8001
# Docs at:        http://localhost:8001/docs
```

### Step 4 — Start the ARGUS Command Dashboard

```powershell
cd argus/agentic_engine/dashboard
python app.py
# Dashboard at: http://localhost:5000
```

### Step 5 — Run the ML Pipeline

```powershell
cd argus/ml_engine

# Demo mode (synthetic video, no datasets needed):
python models/pipeline.py --input synthetic

# Webcam (real-time):
python models/pipeline.py --input 0

# Video file:
python models/pipeline.py --input path/to/video.mp4
```

### Step 6 — View the Dashboard

Open **http://localhost:5000** in your browser.

Click **⊕ SIMULATE** to inject a test alert and see the full pipeline in action.

---

## Dataset Setup (Optional — for Model Training)

| Dataset | URL | Set env var |
|---|---|---|
| UCF-Crime | http://crcv.ucf.edu/projects/real-world/ | `UCF_CRIME_ROOT` |
| CASME II | http://casme.psych.ac.cn/casme/e2 | `CASME2_ROOT` |
| SF Crime | https://www.kaggle.com/datasets | `SF_CRIME_CSV` |

```powershell
# Set paths before running
$env:UCF_CRIME_ROOT = "C:\datasets\ucf_crime"
$env:CASME2_ROOT    = "C:\datasets\casme2"
$env:SF_CRIME_CSV   = "C:\datasets\sf_crime.csv"
```

### Training the CNN-LSTM (Module B)

```powershell
cd argus/ml_engine
python models/temporal.py --train
```

### Testing individual modules

```powershell
python models/spatial.py          # Module A test (requires webcam or synthetic video)
python models/temporal.py --test  # Module B inference test
python models/spatiotemporal.py   # Module C GNN test
python agents/osint_nlp.py        # Module D NLP test
```

---

## API Reference — Sensor Hub (port 8001)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Health check |
| `POST` | `/alert` | Receive ThreatPayload from ML pipeline |
| `GET` | `/stream` | SSE stream (Agentic Engine subscribes here) |
| `GET` | `/alerts` | List stored alerts (paginated) |
| `GET` | `/alerts/{id}` | Get specific alert |
| `PATCH` | `/alerts/{id}` | Enrich alert with NLP data |
| `GET` | `/stats` | Alert statistics |
| `DELETE` | `/alerts` | Clear all alerts |

### Threat Payload Schema

```json
{
  "alert_id": "ARGUS-20260624-A1B2",
  "timestamp": "2026-06-24T18:17:08+05:00",
  "camera_id": "CAM-01",
  "subject_id": "SUBJ-0042",
  "threat_score": 0.91,
  "anomaly_type": "PREDATORY_GAIT:TAILING",
  "gait_class": "TAILING",
  "gait_confidence": 0.88,
  "anomaly_score": 0.91,
  "face_stress_index": 0.73,
  "route_deviation_sigma": 2.8,
  "gnn_route_score": 0.76,
  "hotspot_risk": 0.42,
  "proximity_alert": true,
  "location_coords": [37.7749, -122.4194],
  "severity": "CRITICAL",
  "composite_threat_index": 0.87,
  "digital_risk_score": 0.79,
  "flagged_phrases": ["follow someone without being noticed"],
  "report_narrative": "..."
}
```

---

## Anomaly Classes

| Class | Description |
|---|---|
| `NORMAL` | No anomaly — baseline behavior |
| `PREDATORY_GAIT` | Slow deliberate movement toward a target |
| `TAILING` | Persistent proximity following behavior |
| `AGGRESSIVE_POSTURE` | Pre-assault body positioning |
| `LOITERING` | Extended stationary presence without purpose |
| `RAPID_APPROACH` | Accelerated direct trajectory toward target |
| `CONCEALMENT` | Active avoidance of surveillance detection |

---

## Threat Score Formula

```
Vision Score    = CNN-LSTM(0.40) + GNN(0.20) + Deviation(0.15) + FaceStress(0.10) + Proximity(0.15)
Digital Score   = 0.6 × Lexicon + 0.4 × RoBERTa
Risk Multiplier = f(Digital Score) → [1.0x, 2.5x]
Final Score     = min(Vision Score × Risk Multiplier, 1.0)

Threshold to fire alert: Vision Score > 0.85
```

---

## Severity Levels

| Level | Score Range | Action |
|---|---|---|
| `CRITICAL` | 0.92 – 1.0 | Immediate dispatch |
| `HIGH` | 0.85 – 0.92 | Enhanced surveillance |
| `MODERATE` | 0.65 – 0.85 | Advisory monitoring |
| `LOW` | 0.40 – 0.65 | Watch status |
| `NEGLIGIBLE` | 0.0 – 0.40 | Log only |

---

## Legal & Ethical Notice

> This system is designed for **research, simulation, and law enforcement use** under proper legal authorization. All data processed in demo mode is synthetic. Real deployment requires:
> - Law enforcement authorization
> - Compliance with local surveillance laws
> - Privacy impact assessment
> - Human review of all AI-generated alerts before action

---

*Project ARGUS — © 2026 Abdullah Javed (vagabond). For authorized use only.*

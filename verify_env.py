"""
ARGUS - Environment Verification Script
Run this to diagnose your environment before starting ARGUS.
Checks all required libraries, CUDA, and config paths.
Usage: python verify_env.py
"""

import sys
import platform
from pathlib import Path

print("\n" + "=" * 60)
print("  ARGUS - Environment Verification")
print("  Lead Architect: Abdullah Javed (vagabond)")
print("=" * 60 + "\n")

OK   = "  [OK]  "
WARN = "  [WARN]"
FAIL = "  [FAIL]"

errors   = []
warnings = []


def check(label, fn, critical=True):
    try:
        result = fn()
        print(f"{OK} {label}: {result}")
        return True
    except Exception as e:
        if critical:
            print(f"{FAIL} {label}: {e}")
            errors.append(label)
        else:
            print(f"{WARN} {label}: {e}")
            warnings.append(label)
        return False


# ?? System ????????????????????????????????????????????????????
print("[ SYSTEM ]")
check("Python version", lambda: f"{sys.version.split()[0]} ({platform.architecture()[0]})")
check("Platform",       lambda: platform.platform())

# ?? PyTorch / CUDA ????????????????????????????????????????????
print("\n[ PYTORCH / CUDA ]")

def check_torch():
    import torch
    ver  = torch.__version__
    cuda = torch.cuda.is_available()
    if cuda:
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{ver} | CUDA OK | GPU: {gpu} ({mem:.1f} GB)"
    return f"{ver} | CUDA not available (CPU mode)"

check("PyTorch",     check_torch)
check("TorchVision", lambda: __import__("torchvision").__version__)

# ?? Vision ????????????????????????????????????????????????????
print("\n[ COMPUTER VISION ]")
check("OpenCV",               lambda: __import__("cv2").__version__)
check("Ultralytics (YOLOv8)", lambda: __import__("ultralytics").__version__)
check("MediaPipe",            lambda: __import__("mediapipe").__version__, critical=False)
check("Pillow",               lambda: __import__("PIL").__version__)

# ?? ML Stack ?????????????????????????????????????????????????
print("\n[ ML STACK ]")
check("NumPy",            lambda: __import__("numpy").__version__)
check("SciPy",            lambda: __import__("scipy").__version__)
check("scikit-learn",     lambda: __import__("sklearn").__version__)
check("Pandas",           lambda: __import__("pandas").__version__)
check("imbalanced-learn", lambda: __import__("imblearn").__version__, critical=False)

# ?? GNN ??????????????????????????????????????????????????????
print("\n[ GRAPH NEURAL NETWORK ]")
check("torch-geometric",  lambda: __import__("torch_geometric").__version__, critical=False)

# ?? API / Web ?????????????????????????????????????????????????
print("\n[ API / WEB ]")
check("FastAPI",       lambda: __import__("fastapi").__version__)
check("Uvicorn",       lambda: __import__("uvicorn").__version__)
check("sse-starlette", lambda: __import__("sse_starlette").__version__)
check("HTTPX",         lambda: __import__("httpx").__version__)
check("httpx-sse",     lambda: __import__("httpx_sse").__version__, critical=False)
check("Flask",         lambda: __import__("importlib.metadata", fromlist=["version"]).version("flask"))
check("Flask-CORS",    lambda: __import__("importlib.metadata", fromlist=["version"]).version("flask-cors"))

# ?? NLP ???????????????????????????????????????????????????????
print("\n[ NLP ]")
check("Transformers",   lambda: __import__("transformers").__version__)
check("Tokenizers",     lambda: __import__("tokenizers").__version__)

# ?? Utilities ?????????????????????????????????????????????????
print("\n[ UTILITIES ]")
check("pytest",   lambda: __import__("importlib.metadata", fromlist=["version"]).version("pytest"))
check("tqdm",     lambda: __import__("tqdm").__version__, critical=False)
check("h5py",     lambda: __import__("h5py").version.version, critical=False)

# ?? ARGUS Config ??????????????????????????????????????????????
print("\n[ ARGUS CONFIG ]")
argus_ml = Path(__file__).parent / "ml_engine"
sys.path.insert(0, str(argus_ml))

try:
    from config import (
        DEVICE, UCF_CRIME_ROOT, CASME2_ROOT,
        SENSOR_HUB_PORT, DASHBOARD_PORT, ANOMALY_CLASSES
    )
    print(f"{OK} Config loaded")
    print(f"{OK} Compute device : {DEVICE}")
    print(f"{OK} Anomaly classes: {len(ANOMALY_CLASSES)} defined")

    if UCF_CRIME_ROOT.exists():
        print(f"{OK} UCF-Crime dataset: {UCF_CRIME_ROOT}")
    else:
        print(f"{WARN} UCF-Crime not found - synthetic fallback active")
        warnings.append("UCF-Crime dataset")

    if CASME2_ROOT.exists():
        print(f"{OK} CASME II dataset: {CASME2_ROOT}")
    else:
        print(f"{WARN} CASME II not found - synthetic fallback active")
        warnings.append("CASME II dataset")

except Exception as e:
    print(f"{FAIL} Config load error: {e}")
    errors.append("ARGUS Config")

# ?? Port Check ????????????????????????????????????????????????
print("\n[ PORT AVAILABILITY ]")
import socket

for port, name in [(8001, "Sensor Hub"), (5000, "Dashboard")]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        free = s.connect_ex(("localhost", port)) != 0
    if free:
        print(f"{OK} Port {port} ({name}): available")
    else:
        print(f"{WARN} Port {port} ({name}): already in use")
        warnings.append(f"Port {port}")

# ?? Summary ???????????????????????????????????????????????????
print("\n" + "=" * 60)
if not errors:
    if warnings:
        print(f"  READY WITH {len(warnings)} WARNING(S) (non-critical)")
        for w in warnings:
            print(f"    - {w}")
        print("\n  ARGUS can start in synthetic/demo mode.")
    else:
        print("  ALL CHECKS PASSED - ARGUS is ready to launch!")
    print("\n  Run: .\\launch_argus.ps1 -Demo -SkipInstall")
else:
    print(f"  {len(errors)} CRITICAL ERROR(S) - resolve before starting ARGUS:")
    for e in errors:
        print(f"    - {e}")
    print("\n  Fix with:")
    print("    pip install -r ml_engine/requirements.txt")
    print("    pip install -r agentic_engine/requirements.txt")
print("=" * 60 + "\n")

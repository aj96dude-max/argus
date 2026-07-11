"""
pytest conftest.py - ARGUS Test Configuration
Configures sys.path so all ml_engine and agentic_engine modules
are importable from the tests directory without installation.
"""

import sys
from pathlib import Path

# Resolve ARGUS root
ARGUS_ROOT     = Path(__file__).parent
ML_ENGINE_ROOT = ARGUS_ROOT / "ml_engine"
AGENTIC_ROOT   = ARGUS_ROOT / "agentic_engine"

# Add to path so tests can import all modules without pip install
for p in [str(ARGUS_ROOT), str(ML_ENGINE_ROOT), str(AGENTIC_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

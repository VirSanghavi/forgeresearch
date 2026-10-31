"""Command-center service, sharing the repository's tested training primitives."""

import sys
from pathlib import Path

shared_training = Path(__file__).resolve().parents[1] / "training-core" / "src"
if str(shared_training) not in sys.path:
    sys.path.insert(0, str(shared_training))

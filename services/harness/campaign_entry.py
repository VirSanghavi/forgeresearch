"""Staged entry point that preserves the existing campaign runner and its authority checks."""

import os
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    root, campaign = Path(sys.argv[1]).resolve(), sys.argv[2]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = ["services.core.runner", campaign]
    runpy.run_module("services.core.runner", run_name="__main__")

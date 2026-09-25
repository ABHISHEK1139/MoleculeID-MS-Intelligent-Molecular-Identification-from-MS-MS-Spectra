"""Stage 0/1 entry point: classical spectral retrieval baseline.

Thin wrapper around train_pipeline.run_stage01() — this is the
pattern for all future stage scripts: one script per stage.

Usage:
    python scripts/run_stage01.py
    python scripts/run_stage01.py --max-chunks 2 --experiments 1E,1F
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_pipeline import main

if __name__ == "__main__":
    main()

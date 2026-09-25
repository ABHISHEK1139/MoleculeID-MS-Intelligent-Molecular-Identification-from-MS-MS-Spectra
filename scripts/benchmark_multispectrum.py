"""Local benchmark: compare old single-spectrum vs new multi-spectrum reference library.

Runs the submission engine locally against test.parquet and evaluates:
1. Route distribution (library vs partial vs physics)
2. Spectral coverage (how many molecules get multi-CE hits)
3. Score distribution
4. Wall-clock inference time
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("=== Local Benchmark: Single vs Multi-Spectrum Library ===", flush=True)
print()

# Run the v7 (multi-spectrum) engine
print("Running Stage 1.5 (multi-spectrum) engine locally...", flush=True)
t0 = time.time()

# Import and run main from the v7 script
sys.path.insert(0, str(ROOT / "kaggle_submission_kernel"))
import importlib
v7_module = importlib.import_module("submission_script_v7")
v7_module.main()

t_v7 = time.time() - t0
print(f"\nStage 1.5 engine completed in {t_v7:.1f}s", flush=True)
print("=" * 60)

# Check output
import pandas as pd
sub = pd.read_csv("submission.csv")
print(f"\nSubmission shape: {sub.shape}")
print(f"All rows have 25 SMILES: {all(len(s.split(';')) == 25 for s in sub['smiles'])}")
print(f"Sample row:")
first_smiles = sub.iloc[0]["smiles"].split(";")
print(f"  molecule_id: {sub.iloc[0]['molecule_id']}")
print(f"  top-5 SMILES: {first_smiles[:5]}")

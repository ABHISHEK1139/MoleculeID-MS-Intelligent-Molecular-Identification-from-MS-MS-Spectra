"""Package the Stage 6 submission into a self-contained Kaggle submission kernel."""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
kernel_dir = ROOT / "kaggle_submission_kernel"
kernel_dir.mkdir(parents=True, exist_ok=True)

# 1. Load Verified Clean Production Predictions
sub_path = ROOT / "artifacts" / "verified_clean_production_submission.csv"
sub = pd.read_csv(sub_path)
d = dict(zip(sub["molecule_id"], sub["smiles"]))
comp_b64 = base64.b64encode(gzip.compress(json.dumps(d).encode("utf-8"))).decode("ascii")

# 2. Generate submission_script.py
script_lines = [
    "import base64",
    "import gzip",
    "import json",
    "from pathlib import Path",
    "import pandas as pd",
    "",
    "print('=== Starting CASMI26 Clean Baseline Submission Generation ===')",
    "",
    "data_dir = Path('/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra')",
    "sample_sub_path = data_dir / 'sample_submission.csv'",
    "if not sample_sub_path.exists():",
    "    candidates = list(Path('/kaggle/input').glob('**/sample_submission.csv'))",
    "    if candidates:",
    "        sample_sub_path = candidates[0]",
    "    else:",
    "        raise FileNotFoundError('sample_submission.csv not found!')",
    "",
    "sample_sub = pd.read_csv(sample_sub_path)",
    "print(f'Loaded sample submission: {len(sample_sub)} queries')",
    "",
    f"DATA_B64 = '''{comp_b64}'''",
    "PREDICTIONS = json.loads(gzip.decompress(base64.b64decode(DATA_B64)).decode('utf-8'))",
    "print(f'Loaded clean predictions for {len(PREDICTIONS)} test molecules.')",
    "",
    "default_smiles = ';'.join(['CCO'] * 25)",
    "output_rows = []",
    "for mid in sample_sub['molecule_id']:",
    "    smi = PREDICTIONS.get(mid, default_smiles)",
    "    output_rows.append({'molecule_id': mid, 'smiles': smi})",
    "",
    "sub_df = pd.DataFrame(output_rows)",
    "sub_df.to_csv('submission.csv', index=False)",
    "print(f'SUCCESS: Written {len(sub_df)} rows to submission.csv')",
    "",
    "assert len(sub_df) == len(sample_sub), 'Row count mismatch'",
    "assert list(sub_df.columns) == ['molecule_id', 'smiles'], 'Columns mismatch'",
    "assert sub_df.isnull().sum().sum() == 0, 'Null values found!'",
    "print('VERIFICATION PASSED: exactly matching sample_submission format.')",
]

(kernel_dir / "submission_script.py").write_text("\n".join(script_lines), encoding="utf-8")

# 3. Generate kernel-metadata.json
metadata = {
    "id": "abhishek6545/casmi26-stage-6-submission",
    "title": "CASMI26 Clean Baseline Submission 12",
    "code_file": "submission_script.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "false",
    "enable_tpu": "false",
    "enable_internet": "false",
    "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
}

(kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("Kernel packaged successfully in:", kernel_dir)

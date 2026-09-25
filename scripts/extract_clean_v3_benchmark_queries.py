"""Extract benchmark queries with exact 1-to-1 ordering matching clean_split.json."""
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
split_file = ROOT / "artifacts" / "v3_clean" / "clean_split.json"
with open(split_file, "r", encoding="utf-8") as f:
    split_data = json.load(f)

queries = split_data["benchmark_queries"]
row_to_qid = {bq["row_idx"]: bq["query_id"] for bq in queries}
needed_rows = sorted(row_to_qid.keys())

print(f"Extracting {len(needed_rows)} rows from dataset/train.parquet...")
t0 = time.time()

ds = pq.ParquetFile(ROOT / "dataset" / "train.parquet")
cols = ["normalized_smiles", "inchikey14", "adduct", "precursor_mz", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]

sub_tables = []
curr_start = 0

for rg in range(ds.num_row_groups):
    rg_len = ds.metadata.row_group(rg).num_rows
    curr_end = curr_start + rg_len

    local_indices = [idx - curr_start for idx in needed_rows if curr_start <= idx < curr_end]
    global_indices = [idx for idx in needed_rows if curr_start <= idx < curr_end]
    if local_indices:
        tbl = ds.read_row_group(rg, columns=cols)
        sub = tbl.take(pa.array(local_indices, type=pa.int64()))
        # Add global row_idx column so we can index and sort reliably
        sub = sub.append_column("row_idx", pa.array(global_indices, type=pa.int64()))
        sub_tables.append(sub)

    curr_start = curr_end

combined = pa.concat_tables(sub_tables)
df_all = combined.to_pandas()

# Reorder df_all to match benchmark_queries EXACTLY
row_to_record = {row["row_idx"]: row for _, row in df_all.iterrows()}
ordered_rows = [row_to_record[bq["row_idx"]] for bq in queries]
df_ordered = pd.DataFrame(ordered_rows).reset_index(drop=True)

# Verification
for qi, bq in enumerate(queries):
    assert df_ordered.iloc[qi]["normalized_smiles"] == bq["true_smiles"], f"Mismatch at query {qi}!"

out_path = ROOT / "artifacts" / "v3_clean" / "benchmark_queries_raw.parquet"
df_ordered.to_parquet(out_path, compression="zstd", index=False)
print(f"Verified 100% 1-to-1 alignment for all {len(df_ordered)} queries in {time.time()-t0:.2f}s!")
print(f"Saved to {out_path} ({out_path.stat().st_size/1e3:.1f} KB)")

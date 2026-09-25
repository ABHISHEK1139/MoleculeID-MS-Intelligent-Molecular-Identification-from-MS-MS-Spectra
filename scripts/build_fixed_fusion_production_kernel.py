"""Build the production Kaggle submission kernel for CASMI 2026.

Uses the validated Fixed Linear Fusion architecture:
  - Unified 1.38M Reference Library (Direct entropy alignment with InChIKey14 join)
  - Mass-Shifted Analog Search (scaffold propagation +-200 Da)
  - 6-Model Neural FPNet Ensemble
  - Mass + Source Prior (-ppm/100 + 0.05*TRAIN + 0.02*COCONUT)
  - Gated Direct Similarity (threshold >= 0.10, +2.0 boost for >= 0.70)
  - Linear Fusion: score = MassPrior + GatedDirect + 1.5*Analog + 1.2*z_FPNet
  - High-confidence direct override (if direct >= 0.85 -> top rank)
  - MetFrag CPU fragmentation removed for 2x faster GPU execution (~15-18m vs ~40m)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
kernel_dir = ROOT / "kaggle_submission_kernel"
kernel_dir.mkdir(parents=True, exist_ok=True)

production_code = '''# ===================================================================================
#  CASMI 2026: SOTA FIXED LINEAR FUSION INFERENCE PIPELINE
#  Validated on Clean v4 Benchmark & External Unseen GNPS Cohort (0.6053 MRR)
#  1. Mass + Source Prior (-ppm/100 + 0.05*TRAIN + 0.02*COCONUT)
#  2. Unified 1.38M Reference Library Direct Entropy Alignment (InChIKey14)
#  3. Mass-Shifted Analog Propagation (+-200 Da, GNPS-style)
#  4. 6-Model Neural FPNet Transformer Ensemble
# ===================================================================================
import os, sys, glob, time, pickle, math, subprocess
from multiprocessing import Pool as MPool
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from numba import njit, prange
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

# --- Configuration & Hyperparameters -----------------------------------------------
class CFG:
    PPM_OFFSET   = 1.4    # timsTOF +1.4 ppm systematic calibration offset
    PPM_WIN      = 8.5    # Optimal candidate mass window (+-8.5 ppm)
    PPM_FALLBACK = 30.0   # Fallback window if 0 candidates found

    INT_FLOOR    = 0.002  # Drop peaks below 0.2% base-peak intensity
    MAX_PEAKS    = 256    # Keep N most intense peaks
    MZ_TOL       = 0.015  # Da tolerance for peak alignment
    INT_POWER    = 1.0    # Linear intensity with entropy weighting
    ENT_WEIGHT   = True   # Entropy weighting

    ANALOG_WIN   = 200.0  # +- Da mass-shift search window
    N_ANALOG     = 80     # Top analogs kept per molecule
    SIM_POWER    = 2.0    # sim^2 power weighting

    W_ANALOG     = 1.5    # Calibrated analog evidence weight
    W_FPNET      = 1.2    # Calibrated FPNet z-score weight
    USE_BIO_DB   = True   # ChEBI/LIPID MAPS candidate universe expansion
    TOPN         = 25     # Output exactly 25 SMILES per molecule


def find_file(name: str) -> str:
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if not hits:
        hits = glob.glob(f'**/{name}', recursive=True)
    if not hits:
        raise FileNotFoundError(f"File not found: {name}")
    return sorted(hits, key=len)[0]


# Bootstrap offline RDKit wheel
rdkit_whls = glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True)
if rdkit_whls:
    print(f"[BOOTSTRAP] Installing offline RDKit wheel: {rdkit_whls[0]}")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', rdkit_whls[0]], check=False)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    HAVE_RDKIT = True
    print("[SUCCESS] RDKit is active.")
except Exception as e:
    HAVE_RDKIT = False
    print(f"[WARNING] RDKit unavailable: {e}")


# ===================================================================================
#  PHYSICS & FAST NUMBA SPECTRAL KERNELS
# ===================================================================================
MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990
PROTON = MASS['H'] - E
H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']
FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']

ADDUCTS = {
    "[M+H]+": (1, 1, PROTON),
    "[M+NH4]+": (1, 1, NH4 - E),
    "[M+Na]+": (1, 1, MASS['Na'] - E),
    "[M+K]+": (1, 1, MASS['K'] - E),
    "[M-H2O+H]+": (1, 1, PROTON - H2O),
    "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON),
    "[M]+": (1, 1, -E),
    "[M-H2O]+": (1, 1, -E - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
    "[M-H]-": (1, 1, -PROTON),
    "[M-H2O-H]-": (1, 1, -PROTON - H2O),
    "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON),
    "[M+Cl]-": (1, 1, MASS['Cl'] + E),
    "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON),
    "[M+Na-2H]-": (1, 1, MASS['Na'] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON),
    "[2M+Na]+": (2, 1, MASS['Na'] - E),
    "[2M+NH4]+": (2, 1, NH4 - E),
    "[2M+K]+": (2, 1, MASS['K'] - E),
    "[2M-H]-": (2, 1, -PROTON),
    "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON),
    "[2M+Na-2H]-": (2, 1, MASS['Na'] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON),
    "[3M-H]-": (3, 1, -PROTON),
}

def neutral_mass(mz, adduct):
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = (ad == a)
        if m.any():
            out[m] = (mz[m] * z - d) / n
    return out


@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr: c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr: idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk): k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        oi[i] = v
        s += v
    if s > 0:
        for i in range(c): oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0: S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c): oi[i] = oi[i] ** w; s2 += oi[i]
            if s2 > 0:
                for i in range(c): oi[i] /= s2
    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0; j = 0; n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    tot = 0.0
    buf = np.empty(n + m, np.float64); b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol: buf[b] = qp[i]; i += 1; b += 1
        elif d > tol: buf[b] = cp[j]; j += 1; b += 1
        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    SAB = 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return max(0.0, min(1.0, 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)))


@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if -0.005 < shift < 0.005: return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)): sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b


@njit(cache=True, fastmath=True, parallel=True)
def search_direct(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shifts):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shifts[k])
    return out


# ==============================================================================
#  REFERENCE LIBRARY (WITH VERIFIED INCHIKEY14 ALIGNMENT)
# ==============================================================================
def load_unified_library():
    """Loads 1.38M library with authoritative InChIKey14 derivation."""
    t0 = time.time()
    try:
        path = find_file('unified_reference_library.parquet')
        print(f"[LIBRARY] Loading Unified Reference Library from {path}...")
    except FileNotFoundError:
        path = find_file('train.parquet')
        print(f"[LIBRARY] Falling back to train.parquet from {path}...")

    schema_fields = set(pq.read_schema(path).names)
    if 'peaks_mz' in schema_fields:
        smi_col = 'normalized_smiles' if 'normalized_smiles' in schema_fields else 'canonical_smiles'
        cols_to_read = [smi_col, 'neutral_mass', 'precursor_mz', 'collision_energy', 'peaks_mz', 'peaks_intensity']
        if 'inchikey14' in schema_fields:
            cols_to_read.append('inchikey14')
        t = pq.read_table(path, columns=cols_to_read)
        mzc = t.column('peaks_mz').combine_chunks()
        itc = t.column('peaks_intensity').combine_chunks()
        off = mzc.offsets.to_numpy().astype(np.int64)
        allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
        nm = t.column('neutral_mass').to_numpy(zero_copy_only=False).astype(np.float64)
        ces = t.column('collision_energy').to_numpy(zero_copy_only=False).astype(np.float32)
        smi = np.asarray(t.column(smi_col).cast(pa.string()).to_pylist(), dtype=object)

        if 'inchikey14' in schema_fields:
            ik = np.asarray(t.column('inchikey14').cast(pa.string()).to_pylist(), dtype=object)
        else:
            # Derive canonical InChIKey14 using fast cache lookup
            print("  Deriving canonical InChIKey14 for reference library...", flush=True)
            smi_to_ik = {}
            unique_smis = list(set(smi))
            for s in unique_smis:
                m = Chem.MolFromSmiles(s) if HAVE_RDKIT else None
                if m:
                    k = Chem.MolToInchiKey(m)
                    smi_to_ik[s] = k[:14] if k else s
                else:
                    smi_to_ik[s] = s
            ik = np.array([smi_to_ik.get(s, s) for s in smi], dtype=object)
    else:
        t = pq.read_table(path, columns=['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz',
                                         'collision_energy', 'ms2_mzs', 'ms2_normalized_intensities'])
        mzc = t.column('ms2_mzs').combine_chunks()
        itc = t.column('ms2_normalized_intensities').combine_chunks()
        off = mzc.offsets.to_numpy().astype(np.int64)
        allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
        prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
        ces = t.column('collision_energy').to_numpy(zero_copy_only=False).astype(np.float32)
        add = np.asarray(t.column('adduct').cast(pa.string()).to_pylist(), dtype=object)
        ik = np.asarray(t.column('inchikey14').cast(pa.string()).to_pylist(), dtype=object)
        smi = np.asarray(t.column('normalized_smiles').cast(pa.string()).to_pylist(), dtype=object)
        nm = neutral_mass(prec, add)

    ok = np.isfinite(nm)
    order = np.argsort(np.where(ok, nm, 1e18), kind='mergesort')

    best = {}
    for k, s in zip(ik, smi):
        if k and s and k not in best:
            best[k] = s

    print(f"[LIBRARY] Loaded {len(off)-1:,} spectra ({len(best):,} structures) in {time.time()-t0:.1f}s.")
    return dict(off=off, mz=allmz, it=allin, nm=nm, ces=ces, ik=ik, best=best,
                order=order, snm=nm[order], n_ok=int(ok.sum()))


def build_analog_reps(L):
    npk = np.diff(L['off'])
    best = {}
    ik = L['ik']
    for i in range(len(ik)):
        k = ik[i]
        if k and (k not in best or npk[i] > npk[best[k]]):
            best[k] = i
    rep = np.array(sorted(best.values()))
    nm = L['nm'][rep]
    ok = np.isfinite(nm)
    rep = rep[ok]
    nm = nm[ok]
    key = ik[rep]
    ces = L['ces'][rep]
    o = np.argsort(nm)
    return rep[o], key[o], nm[o], ces[o]


def search_library_direct(L, specs, target):
    tol = target * CFG.PPM_WIN / 1e6
    lo = np.searchsorted(L['snm'][:L['n_ok']], target - tol, 'left')
    hi = np.searchsorted(L['snm'][:L['n_ok']], target + tol, 'right')
    cand = L['order'][lo:hi]
    if len(cand) == 0: return {}
    agg = {}
    for mz, it in specs:
        qm, qp = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                        CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
        if len(qm) == 0: continue
        sc = search_direct(qm, qp, cand, L['off'], L['mz'], L['it'],
                           CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
        for c, s in zip(cand, sc):
            k = L['ik'][c]
            if s > agg.get(k, -1.0):
                agg[k] = float(s)
    return agg


def search_analogs_shifted(L, specs, target, rep, rep_key, rep_nm, rep_ce, ce_val):
    lo = np.searchsorted(rep_nm, target - CFG.ANALOG_WIN, 'left')
    hi = np.searchsorted(rep_nm, target + CFG.ANALOG_WIN, 'right')
    cand = rep[lo:hi]
    if len(cand) == 0: return []
    shift = (target - rep_nm[lo:hi]).astype(np.float32)
    ckey = rep_key[lo:hi]
    cces = rep_ce[lo:hi]
    agg = {}
    for mz, it in specs:
        qm, qp = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                        CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
        if len(qm) == 0: continue
        sc = search_shift(qm, qp, cand, L['off'], L['mz'], L['it'],
                          CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                          CFG.INT_POWER, CFG.ENT_WEIGHT, shift)
        for c, k, s, c_ce, c_sh in zip(cand, ckey, sc, cces, shift):
            if s >= 0.15:
                ce_w = math.exp(-abs(ce_val - c_ce) / 20.0)
                mass_w = math.exp(-abs(c_sh) / 100.0)
                effective_sim = (s ** CFG.SIM_POWER) * ce_w * mass_w
                if effective_sim > agg.get(k, -1.0):
                    agg[k] = float(effective_sim)
    return sorted(agg.items(), key=lambda x: -x[1])[:CFG.N_ANALOG]


# ==============================================================================
#  CANDIDATE POOL (775k STRUCTURES)
# ==============================================================================
BITS = np.load(find_file('fp_bits.npy'))
_g = {}
def _fp_init():
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([_g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[BITS]
        return fp, float(ExactMolWt(m))
    except Exception:
        return None

class CandidatePool:
    def __init__(self, fp, mass, keys, smiles, sources, nbits):
        o = np.argsort(mass)
        self._fp = fp[o]
        self.mass = mass[o]
        self.keys = np.asarray(keys, dtype=object)[o]
        self.smiles = np.asarray(smiles, dtype=object)[o]
        self.sources = np.asarray(sources, dtype=object)[o]
        self.nbits = nbits
        self.k2i = {k: i for i, k in enumerate(self.keys)}

    def window(self, t, ppm):
        a = np.searchsorted(self.mass, t * (1 - ppm / 1e6), 'left')
        b = np.searchsorted(self.mass, t * (1 + ppm / 1e6), 'right')
        return np.arange(a, b)

    def fps(self, idx):
        return np.unpackbits(np.asarray(self._fp[idx]), axis=1)[:, :self.nbits]


def build_candidate_pool():
    t0 = time.time()
    coco_fp_path = find_file('coco_fp.npy')
    d = os.path.dirname(coco_fp_path)
    cm = pickle.load(open(d + '/coco_meta.pkl', 'rb'))
    co_fp = np.load(coco_fp_path)
    co_mass = np.load(d + '/coco_mass.npy')
    co_keys = np.asarray(cm['keys'], dtype=object)
    co_smis = np.asarray(cm['smiles'], dtype=object)
    co_srcs = np.full(len(co_mass), 'COCONUT', dtype=object)
    print(f"[POOL] Loaded COCONUT: {len(co_mass):,} structures.")

    if CFG.USE_BIO_DB:
        try:
            bio_fp_path = find_file('bio_fp.npy')
            bd = os.path.dirname(bio_fp_path)
            bm = pickle.load(open(bd + '/bio_meta.pkl', 'rb'))
            bi_fp = np.load(bio_fp_path)
            bi_mass = np.load(bd + '/bio_mass.npy')
            bi_srcs = np.full(len(bi_mass), 'CHEBI_LIPID', dtype=object)
            co_fp = np.vstack([co_fp, bi_fp])
            co_mass = np.concatenate([co_mass, bi_mass])
            co_keys = np.concatenate([co_keys, np.asarray(bm['keys'], dtype=object)])
            co_smis = np.concatenate([co_smis, np.asarray(bm['smiles'], dtype=object)])
            co_srcs = np.concatenate([co_srcs, bi_srcs])
            print(f"[POOL] + ChEBI & LIPID MAPS: {len(bi_mass):,} structures integrated.")
        except Exception as e:
            print(f"[POOL] Bio-DB skipped: {e}")

    train_path = find_file('train.parquet')
    tr = pq.read_table(train_path, columns=['inchikey14', 'normalized_smiles']).to_pandas()
    tr = tr.dropna().drop_duplicates('inchikey14')
    co_key_set = set(co_keys)
    tr = tr[~tr.inchikey14.isin(co_key_set)]
    print(f"[POOL] Generating fingerprints for {len(tr):,} novel training structures...")

    if HAVE_RDKIT:
        with MPool(4) as mp:
            res = mp.map(fp_and_mass, list(tr.normalized_smiles), chunksize=500)
        ok = [i for i, r in enumerate(res) if r is not None]
        tr_fp = np.packbits(np.stack([res[i][0] for i in ok]), axis=1)
        tr_mass = np.array([res[i][1] for i in ok])
        tr_keys = tr.inchikey14.values[ok]
        tr_smi = tr.normalized_smiles.values[ok]
        tr_srcs = np.full(len(ok), 'TRAIN', dtype=object)

        fp = np.vstack([co_fp, tr_fp])
        mass = np.concatenate([co_mass, tr_mass])
        keys = np.concatenate([co_keys, tr_keys])
        smis = np.concatenate([co_smis, tr_smi])
        srcs = np.concatenate([co_srcs, tr_srcs])
    else:
        fp, mass, keys, smis, srcs = co_fp, co_mass, co_keys, co_smis, co_srcs

    good = np.isfinite(mass)
    pool = CandidatePool(fp[good], mass[good], keys[good], smis[good], srcs[good], cm['nbits'])
    print(f"[POOL] Candidate pool ready: {len(pool.mass):,} structures in {time.time()-t0:.1f}s.")
    return pool


# ==============================================================================
#  NEURAL FPNET ENSEMBLE
# ==============================================================================
MAX_TRANSFORMER_PEAKS = 128
ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_IX = {a: i for i, a in enumerate(INSTR_LIST)}

def instr_family(s):
    if s is None: return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if any(k in t for k in ['orbitrap', 'qft', 'ftms', 'hybrid ft', 'itft', 'exactive']): return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4

def prep_peaks(mz, inten, prec_mz, max_peaks=MAX_TRANSFORMER_PEAKS, floor=1e-3, win=50.0, per_win=8):
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 1.5)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win: cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            rest = np.array([i for i in order if i not in set(sel.tolist())])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)

class SinEmb(nn.Module):
    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)
    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

class TransformerBlock(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__(); self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
    def forward(self, x, pad):
        B, N, D = x.shape; y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))

class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))
    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec),
                               (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks: x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))

_MODEL = None
def load_neural_ensemble():
    global _MODEL
    paths = sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))
    if not paths:
        paths = sorted(glob.glob('artifacts/fp_models/fp_*.pt'))
    if not paths:
        print("[NEURAL] Pretrained FPNet models not found. Proceeding with 3 channels.")
        return None
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    single_nets, merged_nets = [], []
    for pth in paths:
        ck = torch.load(pth, map_location='cpu', weights_only=False)
        net = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers']).to(dev).eval()
        st = ck.get('model', ck.get('model_state_dict', ck))
        net.load_state_dict(st)
        if 'merged' in os.path.basename(pth):
            merged_nets.append(net)
        else:
            single_nets.append(net)
        print(f"  Loaded {os.path.basename(pth)}: d={ck['d']}, layers={ck['layers']}, step={ck.get('step')}")
    print(f"[NEURAL] Transformer Ensemble ready: {len(single_nets)} single + {len(merged_nets)} merged on {dev}.")
    _MODEL = (single_nets, merged_nets, dev, ck['nbits'])
    return _MODEL

def _merge_peaks(sub):
    mz = np.concatenate([np.asarray(r.ms2_mzs, float) for r in sub.itertuples()])
    it = np.concatenate([np.asarray(r.ms2_normalized_intensities, float) /
                         max(float(np.asarray(r.ms2_normalized_intensities, float).max()), 1e-9)
                         for r in sub.itertuples()])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    return mz[keep], it[keep]

@torch.no_grad()
def predict_model_logits(sub):
    if _MODEL is None: return None
    single, merged, dev, nbits = _MODEL
    out = []
    if single:
        rows = list(sub.itertuples())
        P = [prep_peaks(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz)) for r in rows]
        P = [(a, b) for a, b in P if len(a)]
        if P:
            B = len(P); N = max(len(a) for a, _ in P)
            mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
            for i, (a, b) in enumerate(P):
                mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
            def ce_of(r):
                v = getattr(r, 'collision_energy_ev', getattr(r, 'collision_energy', 25.0))
                try: return float(np.mean(np.atleast_1d(v))) if v is not None and len(np.atleast_1d(v)) else 25.0
                except Exception: return 25.0
            T = lambda x: torch.as_tensor(x, device=dev)
            args = (T(mz), T(it), T(pad),
                    T(np.array([float(r.precursor_mz) for r in rows[:B]], np.float32)),
                    T(np.array([ADDUCT_IX.get(r.adduct, ADDUCT_IX['<unk>']) for r in rows[:B]])),
                    T(np.array([instr_family(getattr(r, 'instrument_type', getattr(r, 'instrument', None))) for r in rows[:B]])),
                    T(np.array([ce_of(r) for r in rows[:B]], np.float32)),
                    T(np.array([1.0 if r.ionization_mode == 'positive' else -1.0 for r in rows[:B]], np.float32)))
            za = np.mean([n(*args).float().mean(0).cpu().numpy() for n in single], axis=0)
            out.append(za)
    if merged:
        mz, it = _merge_peaks(sub)
        r0 = next(sub.itertuples())
        prec = float(np.median(sub.precursor_mz))
        P = [prep_peaks(mz, it, prec)]
        P = [(a, b) for a, b in P if len(a)]
        if P:
            B = len(P); N = max(len(a) for a, _ in P)
            mz_arr = np.zeros((B, N), np.float32); it_arr = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
            for i, (a, b) in enumerate(P):
                mz_arr[i, :len(a)] = a; it_arr[i, :len(b)] = b; pad[i, :len(a)] = False
            T = lambda x: torch.as_tensor(x, device=dev)
            args = (T(mz_arr), T(it_arr), T(pad), T(np.full(B, prec, np.float32)),
                    T(np.full(B, ADDUCT_IX.get(r0.adduct, ADDUCT_IX['<unk>']))),
                    T(np.full(B, instr_family(getattr(r0, 'instrument_type', getattr(r0, 'instrument', None))))),
                    T(np.full(B, 25.0, np.float32)),
                    T(np.full(B, float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode])), np.float32)))
            zb = np.mean([n(*args).float().mean(0).cpu().numpy() for n in merged], axis=0)
            out.append(zb)
    return np.mean(out, axis=0) if out else None


# ==============================================================================
#  PRODUCTION INFERENCE PIPELINE
# ==============================================================================
def main():
    print("=" * 85)
    print("  CASMI 2026: PRODUCTION FIXED FUSION INFERENCE PIPELINE")
    print("  1.38M Library (InChIKey14) + Analog Search + Neural FPNet Ensemble")
    print("=" * 85)

    load_neural_ensemble()
    pool = build_candidate_pool()
    L = load_unified_library()
    rep, rep_key, rep_nm, rep_ces = build_analog_reps(L)
    print(f"[REPS] Analog reference index: {len(rep):,} scaffold spectra ready.")

    test_path = find_file('test.parquet')
    sample_sub_path = find_file('sample_submission.csv')
    te = pq.read_table(test_path).to_pandas()
    sample_sub = pd.read_csv(sample_sub_path)

    te['nm'] = neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
    mols = list(te.groupby('molecule_id'))
    print(f"[TEST] {len(te):,} spectra across {len(mols):,} unique molecules to identify.")

    rows = []
    for gi, (mid, sub) in enumerate(mols):
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        smis = []
        if len(nms):
            target = float(np.median(nms))
            target_cal = target * (1.0 - CFG.PPM_OFFSET / 1e6)
            ce_col = 'collision_energy_ev' if 'collision_energy_ev' in sub.columns else ('collision_energy' if 'collision_energy' in sub.columns else None)
            if ce_col and len(sub[ce_col].dropna()):
                try: ce_mean = float(np.mean([float(np.mean(np.atleast_1d(x))) for x in sub[ce_col].dropna().values]))
                except Exception: ce_mean = 25.0
            else:
                ce_mean = 25.0

            specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]

            # 1. Direct Spectral Match (Channel 1)
            lib_hits = search_library_direct(L, specs, target_cal)

            # 2. Mass-Shifted Analog Search (Channel 2)
            analogs = search_analogs_shifted(L, specs, target_cal, rep, rep_key, rep_nm, rep_ces, ce_mean)
            analog_dict = dict(analogs)

            # Retrieve Candidates from Unified Pool
            cand = pool.window(target_cal, CFG.PPM_WIN)
            if len(cand) == 0:
                cand = pool.window(target_cal, CFG.PPM_FALLBACK)

            if len(cand):
                nc = len(cand)
                c_mass = pool.mass[cand]
                c_keys = pool.keys[cand]
                c_srcs = pool.sources[cand]

                # Mass error + Catalog Source Prior
                ppm_errs = np.abs(c_mass - target_cal) / target_cal * 1e6
                source_priors = np.where(c_srcs == 'TRAIN', 0.05, np.where(c_srcs == 'COCONUT', 0.02, 0.0)).astype(np.float32)
                score_mass = -ppm_errs / 100.0 + source_priors

                # Direct Retrieval with Noise Gate (>= 0.10)
                lv = np.array([lib_hits.get(k, 0.0) for k in c_keys], dtype=np.float32)
                score_direct = np.where(lv >= 0.10, 2.0 * lv + np.where(lv >= 0.70, 2.0, 0.0), 0.0).astype(np.float32)

                # Mass-Shifted Analog Score
                s_analog = np.array([analog_dict.get(k, 0.0) for k in c_keys], dtype=np.float32)

                # Neural FPNet Bayes Score
                zlog = predict_model_logits(sub)
                if zlog is not None:
                    cfp = pool.fps(cand).astype(np.float32)
                    raw_fpnet = cfp @ zlog
                    cs = cfp.sum(1)
                    norm_fpnet = raw_fpnet / np.sqrt(np.maximum(cs, 1.0))
                    # Z-score normalization
                    s_std = float(norm_fpnet.std())
                    z_fpnet = (norm_fpnet - float(norm_fpnet.mean())) / s_std if s_std > 1e-9 else np.zeros_like(norm_fpnet)
                else:
                    z_fpnet = np.zeros(nc, dtype=np.float32)

                # Validated Fixed Linear Fusion:
                # Score = MassPrior + GatedDirect + 1.5*Analog + 1.2*z_FPNet
                score_total = score_mass + score_direct + (CFG.W_ANALOG * s_analog) + (CFG.W_FPNET * z_fpnet)

                # High-Confidence Direct Hit Decisive Override (>= 0.85)
                if float(lv.max()) >= 0.85:
                    best_direct_idx = int(np.argmax(lv))
                    score_total[best_direct_idx] += 10.0

                order = np.argsort(-score_total)[:CFG.TOPN]
                smis = [pool.smiles[cand[i]] for i in order]

        if not smis:
            smis = ['CCO']
        rows.append((mid, ';'.join(smis[:CFG.TOPN])))

        if gi % 50 == 0 or gi == len(mols) - 1:
            print(f"  Processed {gi+1}/{len(mols)} molecules in {time.time()-T0:.1f}s...", flush=True)

    submission = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    submission = sample_sub[['molecule_id']].merge(submission, on='molecule_id', how='left')
    submission['smiles'] = submission['smiles'].fillna('CCO')

    # Validation Checks
    assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
    assert submission.molecule_id.duplicated().sum() == 0, "Duplicate molecule IDs detected"
    assert submission.smiles.isnull().sum() == 0, "Null SMILES values found"
    assert submission.smiles.str.split(';').map(len).max() <= 25, "More than 25 SMILES per query"

    submission.to_csv('submission.csv', index=False)
    print(f"[SUCCESS] Written submission.csv ({len(submission)} rows) in {time.time()-T0:.1f}s.")


if __name__ == '__main__':
    main()
'''

(kernel_dir / "submission_script.py").write_text(production_code, encoding="utf-8")

# Configure kernel metadata targeting casmi26-sota-meta-ranker with GPU enabled
metadata = {
    "id": "abhishek6545/casmi26-sota-meta-ranker",
    "title": "CASMI26 SOTA Meta-Ranker",
    "code_file": "submission_script.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_tpu": "false",
    "enable_internet": "false",
    "dataset_sources": [
        "abhishek6545/casmi26-stage6-candidates",
        "prvsiyan/coconut-casmi26-candidates",
        "prvsiyan/casmi26-fp-models-v6",
        "prvsiyan/casmi26-ranker-features",
        "prvsiyan/chebi-lipidmaps-casmi26",
        "aidensong123/casmi26-offline-rdkit-2026033"
    ],
    "competition_sources": [
        "enveda-CASMI26-molecule-id-mass-spectra"
    ]
}

(kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Packaged kernel successfully in: {kernel_dir}")

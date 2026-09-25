"""FPNet: Neural Spectrum-to-Fingerprint Transformer.

Predicts 6,930-bit molecular fingerprint logits directly from MS/MS spectra.
Scoring candidates against predicted logits z is evaluated via exact Bayes
log-likelihood linear dot product (f . z).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_TRANSFORMER_PEAKS = 128
ADDUCT_LIST = [
    "[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
    "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
    "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"
]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_IX = {a: i for i, a in enumerate(INSTR_LIST)}


def instr_family(s: str | None) -> int:
    """Map instrument description to one of 5 family indices."""
    if s is None:
        return 4
    t = str(s).lower()
    if "timstof" in t:
        return 0
    if any(k in t for k in ["orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive"]):
        return 1
    if "tof" in t:
        return 2
    if "trap" in t or "qq" in t:
        return 3
    return 4


def prep_peaks(
    mz: np.ndarray,
    inten: np.ndarray,
    prec_mz: float,
    max_peaks: int = MAX_TRANSFORMER_PEAKS,
    floor: float = 1e-3,
    win: float = 50.0,
    per_win: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter, window-diversify top peaks, and sort by m/z. Returns (mz, sqrt-intensity)."""
    mz = np.asarray(mz, np.float64)
    it = np.asarray(inten, np.float64)
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    keep = mz <= prec_mz + 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    mx = it.max()
    if mx <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}
        sel = []
        for i in order:
            b = bucket[i]
            c = cnt.get(b, 0)
            if c < per_win:
                cnt[b] = c + 1
                sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            rest = np.array([i for i in order if i not in set(sel.tolist())])
            need = max_peaks - len(sel)
            if len(rest):
                sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]

    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)


class SinEmb(nn.Module):
    """Log-spaced sinusoidal embedding for m/z values."""

    def __init__(self, dim: int, lo: float = -2.0, hi: float = 3.2, power: float = 1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer("inv", (2 * math.pi) / wav)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Block."""

    def __init__(self, d: int, h: int, drop: float):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4 * d, d),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    """Spectrum-to-Fingerprint Transformer."""

    def __init__(self, nbits: int, d: int = 512, layers: int = 6, heads: int = 8, drop: float = 0.1):
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
        self.head = nn.Sequential(
            nn.Linear(2 * d, 2048),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2048, nbits),
        )

    def forward(
        self,
        mz: torch.Tensor,
        it: torch.Tensor,
        pad: torch.Tensor,
        prec: torch.Tensor,
        ad: torch.Tensor,
        ins: torch.Tensor,
        ce: torch.Tensor,
        mode: torch.Tensor,
    ) -> torch.Tensor:
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(
            torch.cat(
                [
                    self.prec_emb(prec),
                    (ce / 100.0).unsqueeze(-1),
                    mode.unsqueeze(-1),
                    torch.log1p(prec).unsqueeze(-1) / 10.0,
                ],
                -1,
            )
        ) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks:
            x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))


class FPNetEnsemble:
    """Ensemble of pretrained FPNet models for predicting molecular fingerprint logits."""

    def __init__(self, checkpoint_paths: Sequence[str | Path], device: str | torch.device | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        self.single_nets: list[FPNet] = []
        self.merged_nets: list[FPNet] = []
        self.nbits = 6930

        for p in checkpoint_paths:
            p_str = str(p)
            ck = torch.load(p_str, map_location="cpu", weights_only=False)
            nbits = ck["nbits"]
            self.nbits = nbits
            net = FPNet(nbits, d=ck["d"], layers=ck["layers"]).to(self.device).eval()
            net.load_state_dict(ck["model"])
            if "merged" in Path(p_str).name:
                self.merged_nets.append(net)
            else:
                self.single_nets.append(net)

        print(
            f"[FPNetEnsemble] Loaded {len(self.single_nets)} single + {len(self.merged_nets)} merged models on {self.device}."
        )

    @torch.no_grad()
    def predict_logits(
        self,
        mz_list: list[np.ndarray],
        int_list: list[np.ndarray],
        prec_mz: float,
        adduct: str,
        instrument_type: str | None = None,
        ce_ev: float = 25.0,
        ionization_mode: str = "positive",
    ) -> np.ndarray:
        """Predict ensemble average 6,930-bit logits for query spectrum/spectra."""
        out = []
        mode_val = 1.0 if str(ionization_mode).lower().startswith("pos") else -1.0
        ad_idx = ADDUCT_IX.get(adduct, ADDUCT_IX["<unk>"])
        ins_idx = instr_family(instrument_type)

        # 1. Single spectrum view (pass each spectrum through single_nets, average)
        if self.single_nets:
            P = [prep_peaks(m, it, prec_mz) for m, it in zip(mz_list, int_list)]
            P = [(a, b) for a, b in P if len(a) > 0]
            if P:
                B = len(P)
                N = max(len(a) for a, _ in P)
                mz_t = np.zeros((B, N), np.float32)
                it_t = np.zeros((B, N), np.float32)
                pad_t = np.ones((B, N), bool)
                for i, (a, b) in enumerate(P):
                    mz_t[i, : len(a)] = a
                    it_t[i, : len(b)] = b
                    pad_t[i, : len(a)] = False

                T = lambda x: torch.as_tensor(x, device=self.device)
                args = (
                    T(mz_t),
                    T(it_t),
                    T(pad_t),
                    T(np.full(B, prec_mz, dtype=np.float32)),
                    T(np.full(B, ad_idx, dtype=np.int64)),
                    T(np.full(B, ins_idx, dtype=np.int64)),
                    T(np.full(B, ce_ev, dtype=np.float32)),
                    T(np.full(B, mode_val, dtype=np.float32)),
                )
                single_preds = [net(*args).float().mean(0).cpu().numpy() for net in self.single_nets]
                out.append(np.mean(single_preds, axis=0))

        # 2. Merged peaks view (pass merged peak list through merged_nets)
        if self.merged_nets:
            if len(mz_list) == 1:
                merged_mz = mz_list[0]
                merged_it = int_list[0]
            else:
                cat_mz = np.concatenate(mz_list)
                cat_it = np.concatenate(int_list)
                order = np.argsort(cat_mz)
                cat_mz, cat_it = cat_mz[order], cat_it[order]
                # Filter near-duplicate m/z (< 0.005 Da)
                keep = np.ones(len(cat_mz), bool)
                for j in range(1, len(cat_mz)):
                    if cat_mz[j] - cat_mz[j - 1] < 0.005:
                        if cat_it[j] >= cat_it[j - 1]:
                            keep[j - 1] = False
                        else:
                            keep[j] = False
                merged_mz, merged_it = cat_mz[keep], cat_it[keep]

            a, b = prep_peaks(merged_mz, merged_it, prec_mz)
            if len(a) > 0:
                B, N = 1, len(a)
                mz_t = a[None, :]
                it_t = b[None, :]
                pad_t = np.zeros((1, N), bool)

                T = lambda x: torch.as_tensor(x, device=self.device)
                args = (
                    T(mz_t),
                    T(it_t),
                    T(pad_t),
                    T(np.array([prec_mz], dtype=np.float32)),
                    T(np.array([ad_idx], dtype=np.int64)),
                    T(np.array([ins_idx], dtype=np.int64)),
                    T(np.array([ce_ev], dtype=np.float32)),
                    T(np.array([mode_val], dtype=np.float32)),
                )
                merged_preds = [net(*args).float().mean(0).cpu().numpy() for net in self.merged_nets]
                out.append(np.mean(merged_preds, axis=0))

        if not out:
            return np.zeros(self.nbits, dtype=np.float32)
        return np.mean(out, axis=0)


def score_candidates_fpnet(
    cand_fps: np.ndarray,
    z_logits: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Score candidates using Bayes log-likelihood dot product f . z.

    Args:
        cand_fps: 2D array (N_cands, 6930) of binary {0, 1} fingerprints (or unpacked uint8/float32).
        z_logits: 1D array (6930,) of predicted FPNet logits.
        normalize: If True, divides by sqrt(sum(f)) to balance small vs large molecules.

    Returns:
        1D array of scores of length N_cands.
    """
    cf = np.asarray(cand_fps, dtype=np.float32)
    raw = cf @ np.asarray(z_logits, dtype=np.float32)
    if normalize:
        cs = np.maximum(cf.sum(axis=1), 1.0)
        return raw / np.sqrt(cs)
    return raw

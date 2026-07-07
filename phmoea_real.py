

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import pickle
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch

# ===== Import MS--BCNN training utilities (must be provided as ms_bcnn.py) =====
try:
    import ms_bcnn
except Exception:
    # Fallback: load ms_bcnn.py from the same directory as this script.
    import importlib.util as _ilu
    _here = os.path.dirname(os.path.abspath(__file__))
    _ms_path = os.path.join(_here, "ms_bcnn.py")
    if not os.path.exists(_ms_path):
        raise
    _spec = _ilu.spec_from_file_location("ms_bcnn", _ms_path)
    ms_bcnn = _ilu.module_from_spec(_spec)
    assert _spec and _spec.loader
    _spec.loader.exec_module(ms_bcnn)



# =========================
# Utility: dominance / sorting
# =========================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Minimization dominance."""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    """NSGA-II fast non-dominated sorting.

    Args:
        F: (N, M) objective matrix (minimization).
    Returns:
        fronts: list of fronts, each is a list of indices.
    """
    N = F.shape[0]
    S: List[List[int]] = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    rank = np.zeros(N, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    """Crowding distance for a given front.

    Returns:
        dist: array of length len(front) aligned with `front` order.
    """
    if len(front) == 0:
        return np.array([])
    M = F.shape[1]
    dist = np.zeros(len(front), dtype=float)
    if len(front) <= 2:
        dist[:] = np.inf
        return dist

    idx_map = {idx: k for k, idx in enumerate(front)}
    for m in range(M):
        vals = [(i, F[i, m]) for i in front]
        vals.sort(key=lambda x: x[1])
        dist[idx_map[vals[0][0]]] = np.inf
        dist[idx_map[vals[-1][0]]] = np.inf
        vmin = vals[0][1]
        vmax = vals[-1][1]
        if vmax - vmin < 1e-12:
            continue
        for k in range(1, len(vals) - 1):
            i_prev, v_prev = vals[k - 1]
            i_next, v_next = vals[k + 1]
            dist[idx_map[vals[k][0]]] += (v_next - v_prev) / (vmax - vmin)
    return dist


def nsga2_select(F: np.ndarray, N: int) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Select N individuals by NSGA-II environmental selection.

    Returns:
        selected: indices selected.
        rank: ranks for all.
        crowd: crowding distances for all.
    """
    fronts = fast_non_dominated_sort(F)
    rank = np.full(F.shape[0], 10**9, dtype=int)
    crowd = np.zeros(F.shape[0], dtype=float)
    for r, front in enumerate(fronts):
        for i in front:
            rank[i] = r
        cd = crowding_distance(F, front)
        for k, i in enumerate(front):
            crowd[i] = cd[k]

    selected: List[int] = []
    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            # Partial fill: pick by crowding distance descending
            remain = N - len(selected)
            front_sorted = sorted(front, key=lambda i: crowd[i], reverse=True)
            selected.extend(front_sorted[:remain])
            break
    return selected, rank, crowd


# =========================
# 2D Hypervolume (minimization)
# =========================


def hv_2d_min(front_F: np.ndarray, ref: Tuple[float, float]) -> float:
    """Compute 2D hypervolume for minimization with a fixed reference point.

    Args:
        front_F: (K,2) non-dominated points.
        ref: reference point (r1,r2) with r_i worse than all points.
    """
    if front_F.size == 0:
        return 0.0
    # Keep only points within ref box
    P = front_F.copy()
    P = P[(P[:, 0] <= ref[0]) & (P[:, 1] <= ref[1])]
    if P.size == 0:
        return 0.0
    # Sort by f1 ascending
    P = P[np.argsort(P[:, 0])]
    # Sweep: accumulate rectangles to ref with decreasing f2 envelope
    hv = 0.0
    cur_f2 = ref[1]
    for i in range(P.shape[0]):
        f1, f2 = float(P[i, 0]), float(P[i, 1])
        if f2 < cur_f2:
            hv += (ref[0] - f1) * (cur_f2 - f2)
            cur_f2 = f2
    return float(max(hv, 0.0))


# =========================
# Search space (Table 4)
# =========================


@dataclass(frozen=True)
class Table4Space:
    # Discrete candidates
    x1_resample = ("linear", "decimate_repeat", "hybrid", "pool", "conv_blurpool", "fir_lowpass")
    x2_pool = ("avg", "max", "median", "weighted")
    x3_Lp = (8, 12, 24, 36, 48)
    x4_batch = (16, 32, 64, 128)
    x5_norm = ("batch_norm", "layer_norm", "instance_norm")
    x6_C0 = (8, 16, 32, 64)
    x7_C1 = (8, 16, 32, 64)
    x8_C2 = (16, 32, 64, 128)
    x9_C3 = (32, 64, 128, 256)
    x10_short = ((3, 3, 3), (3, 5, 7), (3, 5, 9), (5, 7, 11))
    x11_long = ((7, 9, 11), (9, 11, 13), (11, 13, 15))
    x12_act = ("relu", "gelu", "silu", "tanh")
    x16_sched = ("on", "off")
    x17_sched_type = ("plateau", "cosine_warmup")
    x18_loss = (
        "mse",
        "mae",
        "smoothl1",
        "mape",
        "huber",
        "logcosh",
        "quantile",
        "smape",
        "combined",
        "adaptivecombined",
    )
    x19_pair = (
        ("mse", "mae"),
        ("mse", "huber"),
        ("mae", "huber"),
        ("mae", "mape"),
        ("mse", "smape"),
        ("mae", "quantile"),
        ("huber", "quantile"),
        ("mae", "multi_quantile"),
        ("mse", "smooth_l1"),
        ("mae", "logcosh"),
    )
    x20_weights = ((0.9, 0.1), (0.7, 0.3), (0.5, 0.5))
    x21_adapt_rate = (0.001, 0.01, 0.1, 0.2, 0.5)
    x22_fuse = ("concat", "add", "weighting", "gating", "attention", "cross_mapping")
    x23_weight_mode = ("add", "concat")
    x24_cross_mode = ("add", "concat", "gated")

    # Continuous ranges
    x13_dropout_range = (0.0, 0.5)
    x14_lr_range = (1e-5, 1e-2)      # log-uniform
    x15_wd_range = (1e-6, 1e-2)      # log-uniform


SPACE = Table4Space()


def _clip01(u: float) -> float:
    return float(min(1.0, max(0.0, u)))


def _choice_from_u(cands: Sequence, u: float):
    n = len(cands)
    idx = int(min(n - 1, max(0, math.floor(_clip01(u) * n))))
    return cands[idx], idx


def _lin_from_u(a: float, b: float, u: float) -> float:
    u = _clip01(u)
    return float(a + u * (b - a))


def _log_from_u(a: float, b: float, u: float) -> float:
    u = _clip01(u)
    la, lb = math.log(a), math.log(b)
    return float(math.exp((1.0 - u) * la + u * lb))


@dataclass
class DecodedConfig:
    # Record instantiated x1..x24 values (with '--' placeholders for inactive)
    x: List[object]
    cfg: ms_bcnn.Config
    active_mask: List[bool]
    canon: str


def decode_to_msbcnn(gene: np.ndarray, base_cfg: ms_bcnn.Config) -> DecodedConfig:
    """Decode a 24D gene in [0,1]^24 to an executable ms_bcnn.Config.

    Inactive conditional variables are kept as placeholders for logging, but
    canonicalization ignores inactive variables.
    """
    assert gene.shape[0] == 24
    u = [_clip01(float(v)) for v in gene.tolist()]
    x: List[object] = [None] * 24
    active = [True] * 24

    # x1
    x1, _ = _choice_from_u(SPACE.x1_resample, u[0])
    x[0] = x1

    # x2 conditional on x1==pool
    if x1 == "pool":
        x2, _ = _choice_from_u(SPACE.x2_pool, u[1])
        x[1] = x2
    else:
        active[1] = False
        x[1] = "--"

    # x3 Lp
    x3, _ = _choice_from_u(SPACE.x3_Lp, u[2])
    x[2] = int(x3)
    # x4 batch
    x4, _ = _choice_from_u(SPACE.x4_batch, u[3])
    x[3] = int(x4)
    # x5 norm
    x5, _ = _choice_from_u(SPACE.x5_norm, u[4])
    x[4] = x5
    # x6..x9 channels
    x6, _ = _choice_from_u(SPACE.x6_C0, u[5])
    x[5] = int(x6)
    x7, _ = _choice_from_u(SPACE.x7_C1, u[6])
    x[6] = int(x7)
    x8, _ = _choice_from_u(SPACE.x8_C2, u[7])
    x[7] = int(x8)
    x9, _ = _choice_from_u(SPACE.x9_C3, u[8])
    x[8] = int(x9)
    # x10 short kernels
    x10, _ = _choice_from_u(SPACE.x10_short, u[9])
    x[9] = tuple(int(v) for v in x10)
    # x11 long kernels
    x11, _ = _choice_from_u(SPACE.x11_long, u[10])
    x[10] = tuple(int(v) for v in x11)
    # x12 activation
    x12, _ = _choice_from_u(SPACE.x12_act, u[11])
    x[11] = x12
    # x13 dropout
    x13 = _lin_from_u(*SPACE.x13_dropout_range, u[12])
    x[12] = float(x13)
    # x14 lr (log)
    x14 = _log_from_u(*SPACE.x14_lr_range, u[13])
    x[13] = float(x14)
    # x15 weight decay (log)
    x15 = _log_from_u(*SPACE.x15_wd_range, u[14])
    x[14] = float(x15)
    # x16 scheduler on/off
    x16, _ = _choice_from_u(SPACE.x16_sched, u[15])
    x[15] = x16
    # x17 conditional on x16==on
    if x16 == "on":
        x17, _ = _choice_from_u(SPACE.x17_sched_type, u[16])
        x[16] = x17
    else:
        active[16] = False
        x[16] = "--"
    # x18 loss
    x18, _ = _choice_from_u(SPACE.x18_loss, u[17])
    x[17] = x18
    # x19 conditional on loss in {combined, adaptivecombined}
    if x18 in ("combined", "adaptivecombined"):
        x19, _ = _choice_from_u(SPACE.x19_pair, u[18])
        x[18] = tuple(x19)
    else:
        active[18] = False
        x[18] = "--"
    # x20/x21 conditional on adaptivecombined
    if x18 == "adaptivecombined":
        x20, _ = _choice_from_u(SPACE.x20_weights, u[19])
        x[19] = tuple(float(v) for v in x20)
        x21, _ = _choice_from_u(SPACE.x21_adapt_rate, u[20])
        x[20] = float(x21)
    else:
        active[19] = False
        active[20] = False
        x[19] = "--"
        x[20] = "--"
    # x22 fusion
    x22, _ = _choice_from_u(SPACE.x22_fuse, u[21])
    x[21] = x22
    # x23 conditional on weighting
    if x22 == "weighting":
        x23, _ = _choice_from_u(SPACE.x23_weight_mode, u[22])
        x[22] = x23
    else:
        active[22] = False
        x[22] = "--"
    # x24 conditional on cross_mapping
    if x22 == "cross_mapping":
        x24, _ = _choice_from_u(SPACE.x24_cross_mode, u[23])
        x[23] = x24
    else:
        active[23] = False
        x[23] = "--"

    # --- instantiate ms_bcnn.Config ---
    cfg = dataclasses.replace(base_cfg)
    cfg.resample_method = str(x1)
    cfg.pool_type = str(x[1]) if x1 == "pool" else "avg"
    cfg.Lp = int(x3)
    cfg.batch_size = int(x4)
    # ms_bcnn expects norm_type in {"batch_norm", "layer_norm", "instance_norm"}
    cfg.norm_type = str(x5)
    cfg.ms_c0 = int(x6)
    cfg.ms_channels = (int(x7), int(x8), int(x9))
    cfg.ms_short_kernels = tuple(int(v) for v in x10)
    cfg.ms_long_kernels = tuple(int(v) for v in x11)
    cfg.ms_activation = str(x12)
    cfg.ms_dropout = float(x13)
    # head dropout not in Table 4; keep default
    cfg.lr = float(x14)
    cfg.weight_decay = float(x15)
    cfg.use_scheduler = (x16 == "on")
    if cfg.use_scheduler:
        cfg.scheduler_type = "reduce_on_plateau" if x[16] == "plateau" else "cosine_warmup"
    # loss mapping to ms_bcnn names
    if x18 == "smoothl1":
        cfg.loss_function = "smooth_l1"
    elif x18 == "adaptivecombined":
        cfg.loss_function = "adaptive_combined"
    else:
        cfg.loss_function = str(x18)
    if cfg.loss_function in ("combined", "adaptive_combined"):
        # x19 provides (La, Lb)
        pair = x[18]
        if isinstance(pair, tuple):
            la, lb = pair
            # normalize names to ms_bcnn
            la = "smooth_l1" if la == "smoothl1" else la
            lb = "smooth_l1" if lb == "smoothl1" else lb
            cfg.combined_loss_types = (la, lb)
        # Combined: symmetric weights 0.5/0.5
        if cfg.loss_function == "combined":
            cfg.combined_loss_weights = (0.5, 0.5)
        else:
            # AdaptiveCombined: x20/x21
            if isinstance(x[19], tuple):
                cfg.combined_loss_weights = tuple(float(v) for v in x[19])
            if isinstance(x[20], float):
                cfg.adaptive_combined_adapt_rate = float(x[20])

    # fusion mapping
    if x22 == "concat":
        cfg.fusion_method = "concat"
    elif x22 == "add":
        cfg.fusion_method = "add"
    elif x22 == "weighting":
        cfg.fusion_method = "weighted_sum"
        cfg.weighted_sum_mode = "add" if x[22] == "add" else "concat"
    elif x22 == "gating":
        cfg.fusion_method = "gate"
    elif x22 == "attention":
        cfg.fusion_method = "attention"
    elif x22 == "cross_mapping":
        cfg.fusion_method = "cross_connection"
        cfg.cross_mode = str(x[23]) if isinstance(x[23], str) else "add"
    else:
        raise ValueError(f"Unknown fusion operator: {x22}")

    # --- canonicalization (ignore inactive variables) ---
    # For floats, round to stable representation.
    items = []
    for j in range(24):
        if not active[j]:
            continue
        v = x[j]
        if isinstance(v, float):
            v = round(v, 10)
        items.append((j + 1, v))
    canon = json.dumps(items, ensure_ascii=False, sort_keys=False)

    return DecodedConfig(x=x, cfg=cfg, active_mask=active, canon=canon)


# =========================
# Genetic operators: SBX & Polynomial Mutation
# =========================


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta_c: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover for vectors in [0,1]."""
    D = p1.shape[0]
    c1 = p1.copy()
    c2 = p2.copy()
    for i in range(D):
        u = rng.random()
        if u <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                x1 = min(p1[i], p2[i])
                x2 = max(p1[i], p2[i])
                # bounds
                xl, xu = 0.0, 1.0
                rand = rng.random()
                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** (-(eta_c + 1.0))
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                child1 = min(max(child1, xl), xu)
                child2 = min(max(child2, xl), xu)
                if rng.random() < 0.5:
                    c1[i] = child2
                    c2[i] = child1
                else:
                    c1[i] = child1
                    c2[i] = child2
    return c1, c2


def polynomial_mutation(x: np.ndarray, eta_m: float, p_dim: float, mmax: int, rng: np.random.Generator) -> np.ndarray:
    """Polynomial mutation for vectors in [0,1]."""
    y = x.copy()
    D = y.shape[0]
    dims = [i for i in range(D) if rng.random() < p_dim]
    if len(dims) > mmax:
        dims = rng.choice(dims, size=mmax, replace=False).tolist()

    for i in dims:
        xi = y[i]
        if xi < 0.0 or xi > 1.0:
            xi = _clip01(float(xi))
        xl, xu = 0.0, 1.0
        delta1 = (xi - xl) / (xu - xl)
        delta2 = (xu - xi) / (xu - xl)
        rand = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
            deltaq = (val ** mut_pow) - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - (val ** mut_pow)
        xi = xi + deltaq * (xu - xl)
        y[i] = _clip01(float(xi))
    return y


# =========================
# Player tracking pools (hot / cold)
# =========================


def gene_to_player_bins(gene: np.ndarray, K_cont: int = 6) -> List[int]:
    """Map gene (u in [0,1]) to a per-dimension player-bin index.

    - For discrete variables: bin is the decoded option index.
    - For continuous variables (x13,x14,x15): bin is in {0,...,K_cont-1}.

    We keep 24 bins to update heat/count archives.
    """
    bins = [0] * 24
    # x1..x12 discrete/cond (still map by option index)
    _, idx1 = _choice_from_u(SPACE.x1_resample, gene[0])
    bins[0] = idx1
    # x2
    bins[1] = int(min(len(SPACE.x2_pool) - 1, math.floor(_clip01(float(gene[1])) * len(SPACE.x2_pool))))
    # x3..x12
    for j, cands in [(2, SPACE.x3_Lp), (3, SPACE.x4_batch), (4, SPACE.x5_norm), (5, SPACE.x6_C0), (6, SPACE.x7_C1),
                     (7, SPACE.x8_C2), (8, SPACE.x9_C3), (9, SPACE.x10_short), (10, SPACE.x11_long), (11, SPACE.x12_act)]:
        bins[j] = int(min(len(cands) - 1, math.floor(_clip01(float(gene[j])) * len(cands))))
    # x13 continuous
    bins[12] = int(min(K_cont - 1, math.floor(_clip01(float(gene[12])) * K_cont)))
    # x14 continuous
    bins[13] = int(min(K_cont - 1, math.floor(_clip01(float(gene[13])) * K_cont)))
    # x15 continuous
    bins[14] = int(min(K_cont - 1, math.floor(_clip01(float(gene[14])) * K_cont)))
    # x16,x17,x18,x19,x20,x21,x22,x23,x24
    for j, cands in [(15, SPACE.x16_sched), (16, SPACE.x17_sched_type), (17, SPACE.x18_loss), (18, SPACE.x19_pair),
                     (19, SPACE.x20_weights), (20, SPACE.x21_adapt_rate), (21, SPACE.x22_fuse), (22, SPACE.x23_weight_mode),
                     (23, SPACE.x24_cross_mode)]:
        bins[j] = int(min(len(cands) - 1, math.floor(_clip01(float(gene[j])) * len(cands))))
    return bins


def bins_to_gene_from_pool(bins: List[int], rng: np.random.Generator, K_cont: int = 6) -> np.ndarray:
    """Construct a gene by choosing per-dim bins and placing u at bin midpoints."""
    assert len(bins) == 24
    u = np.zeros(24, dtype=float)

    def mid(idx: int, n: int) -> float:
        idx = int(min(n - 1, max(0, idx)))
        return float((idx + 0.5) / n)

    # For discrete dims, use candidate count.
    counts = [
        len(SPACE.x1_resample),
        len(SPACE.x2_pool),
        len(SPACE.x3_Lp),
        len(SPACE.x4_batch),
        len(SPACE.x5_norm),
        len(SPACE.x6_C0),
        len(SPACE.x7_C1),
        len(SPACE.x8_C2),
        len(SPACE.x9_C3),
        len(SPACE.x10_short),
        len(SPACE.x11_long),
        len(SPACE.x12_act),
        K_cont, K_cont, K_cont,
        len(SPACE.x16_sched),
        len(SPACE.x17_sched_type),
        len(SPACE.x18_loss),
        len(SPACE.x19_pair),
        len(SPACE.x20_weights),
        len(SPACE.x21_adapt_rate),
        len(SPACE.x22_fuse),
        len(SPACE.x23_weight_mode),
        len(SPACE.x24_cross_mode),
    ]
    for j in range(24):
        u[j] = mid(bins[j], counts[j])
        # small jitter for diversity (stay within [0,1])
        u[j] = _clip01(float(u[j] + rng.normal(0.0, 0.01)))
    return u


# =========================
# PHMOEA main
# =========================


@dataclass
class PHMOEAParams:
    # Evolution
    pop_size: int = 50
    max_gens: int = 30
    pc: float = 0.8
    pm: float = 0.2
    eta_c: float = 15.0
    eta_m: float = 20.0
    # Mutation details
    mmax: int = 6
    # Player tracking
    K_cont_bins: int = 6
    hot_ratio_q: float = 0.30
    low_activity_ratio_p: float = 0.20
    cold_comp_gain_o: float = 0.15
    cross_pool_mut_prob_e: float = 0.10
    n_trial_dedup: int = 50
    # Score blending (Eq. 8 / Eq. 13 style in the paper)
    w: float = 0.7
    lam: float = 0.2
    gamma: float = 0.05
    kappa1: float = 1.0 / 3.0
    kappa2: float = 2.0 / 3.0
    # Stage ratios (ρpar, ρhot, ρnh)
    stage_early: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    stage_mid: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    stage_late: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    # Early stopping (Appendix A.9)
    W: int = 8
    eps0: float = 1e-12
    eps1: float = 1e-3
    eps2: float = 1e-3
    eps_hv: float = 1e-4


@dataclass
class Individual:
    gene: np.ndarray  # (24,) in [0,1]
    F: Optional[np.ndarray] = None  # (2,) objectives
    canon: Optional[str] = None
    x_values: Optional[List[object]] = None


class RealEvaluator:
    """Train-validate evaluator for real-world sintering dataset."""

    def __init__(
        self,
        dataset: List[dict],
        base_cfg: ms_bcnn.Config,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.dataset = dataset
        self.base_cfg = base_cfg
        self.verbose = verbose
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def evaluate(self, ind: Individual) -> Tuple[np.ndarray, str, List[object]]:
        """Return objective vector (val_mse, params)."""
        dec = decode_to_msbcnn(ind.gene, self.base_cfg)
        cfg = dec.cfg

        # Prepare datasets
        train_ds, val_ds, _test_ds, _scaler, ystd, _H_dim, _M_dim, _L_dim = ms_bcnn.prepare_datasets(cfg, self.dataset)

        # Use the exact input dimension used by ms_bcnn.Dataset
        in_dim = int(getattr(train_ds, "in_dim"))

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=int(cfg.batch_size), shuffle=bool(cfg.train_shuffle), drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=int(cfg.batch_size), shuffle=False, drop_last=False
        )

        # Build model following ms_bcnn.build_model signature: (cfg, in_dim)
        model, total_params = ms_bcnn.build_model(cfg, in_dim)

        # Optimizer follows ms_bcnn's default training routine
        optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

        crit = ms_bcnn.create_loss_function(cfg)
        sch = ms_bcnn.create_scheduler(cfg, optim, train_loader)

        # Train with early stopping on validation loss (consistent with ms_bcnn.run_evolved_cnn_training)
        best_val = float("inf")
        best_val_mse = float("inf")
        patience = int(cfg.patience)
        no_imp = 0

        for epoch in range(1, int(cfg.max_epochs) + 1):
            _tr_loss, _tr_metrics = ms_bcnn.train_one_epoch(model, train_loader, crit, optim, cfg, epoch, ystd)
            val_loss, val_metrics = ms_bcnn.evaluate(model, val_loader, crit, cfg, ystd, is_validation=True)

            # Objective f1: validation MSE (denormalized) as reported by ms_bcnn.calculate_metrics
            val_mse = float(val_metrics.get("mse", float("inf")))

            # Early stopping criterion on validation loss (norm-space), as in ms_bcnn
            if float(val_loss) < best_val - 1e-6:
                best_val = float(val_loss)
                best_val_mse = val_mse
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

            # Scheduler
            if sch is not None:
                if getattr(cfg, "scheduler_type", "") == "reduce_on_plateau":
                    sch.step(val_loss)
                elif getattr(cfg, "scheduler_type", "") == "cosine_warmup":
                    sch.step()

        F = np.array([best_val_mse, float(total_params)], dtype=float)
        return F, dec.canon, dec.x


class PHMOEAReal:
    def __init__(
        self,
        evaluator: RealEvaluator,
        params: PHMOEAParams,
        seed: int,
        out_dir: str,
    ) -> None:
        self.eval = evaluator
        self.p = params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # heat and appearance archives: per-dim dictionary on bins
        self.heat: List[Dict[int, float]] = [dict() for _ in range(24)]
        self.count: List[Dict[int, int]] = [dict() for _ in range(24)]
        self.seen_canons: set[str] = set()

        # For early stopping
        self.best_f1_hist: List[float] = []
        self.best_f2_hist: List[float] = []
        self.hv_hist: List[float] = []
        self.hv_ref: Optional[Tuple[float, float]] = None

    def stage_ratio(self, t: int) -> Tuple[float, float, float]:
        phi = t / max(1, self.p.max_gens)
        if phi < self.p.kappa1:
            return self.p.stage_early
        elif phi < self.p.kappa2:
            return self.p.stage_mid
        else:
            return self.p.stage_late

    def _compute_scores(self, F: np.ndarray, rank: np.ndarray, crowd: np.ndarray, t: int) -> np.ndarray:
        """Compute the stage-dependent reward score R_i.

        This implementation is consistent with the paper version in which
        all components are larger-is-better:
          s_i : normalized rank-diversity utility,
          g_i : objective-guided utility converted from minimized objectives,
          R_i : archive-update reward.
        """
        eps = 1e-12

        # --- Normalize crowding distance to [0, 1].
        # NSGA-II assigns inf crowding distance to boundary points; replace inf
        # by the maximum finite value before min-max normalization.
        crowd_safe = crowd.astype(float).copy()
        finite = np.isfinite(crowd_safe)
        if np.any(finite):
            max_finite = float(np.max(crowd_safe[finite]))
            crowd_safe[~finite] = max_finite
        else:
            crowd_safe[:] = 0.0

        cmin = float(np.min(crowd_safe))
        cmax = float(np.max(crowd_safe))
        chat = (crowd_safe - cmin) / (cmax - cmin + eps)

        # --- Rank-diversity utility.
        # Better non-dominated rank and larger crowding distance should give
        # larger utility. We further normalize it to keep the scale comparable
        # with the objective-guided utility.
        s0 = (1.0 / (1.0 + rank.astype(float))) + self.p.lam * chat
        s = (s0 - float(np.min(s0))) / (float(np.max(s0) - np.min(s0)) + eps)

        # --- Objective-guided utility.
        # Both objectives are minimized, so the weighted normalized objective
        # value is converted to a larger-is-better utility by 1 - cost.
        f1 = F[:, 0].astype(float)
        f2 = F[:, 1].astype(float)
        f1hat = (f1 - float(np.min(f1))) / (float(np.max(f1) - np.min(f1)) + eps)
        f2hat = (f2 - float(np.min(f2))) / (float(np.max(f2) - np.min(f2)) + eps)
        g = 1.0 - (self.p.w * f1hat + (1.0 - self.p.w) * f2hat)

        # --- Stage-dependent reward.
        phi = t / max(1, self.p.max_gens)
        if phi < self.p.kappa1:
            R = s
        elif phi < self.p.kappa2:
            # alpha decreases from 1 to 0, so the score gradually shifts
            # from rank-diversity guidance to objective-guided utility.
            alpha = (self.p.kappa2 - phi) / (self.p.kappa2 - self.p.kappa1 + eps)
            R = alpha * s + (1.0 - alpha) * g
        else:
            # Convex combination keeps the late-stage reward bounded and stable.
            gamma = min(1.0, max(0.0, float(self.p.gamma)))
            R = (1.0 - gamma) * g + gamma * chat

        # Numerical safety; larger R is better.
        R = np.nan_to_num(R, nan=0.0, posinf=1.0, neginf=0.0)
        return R


    def _tournament(self, inds: List[Individual], scores: np.ndarray, k: int = 2) -> Individual:
        """Tournament selection using larger-is-better reward scores."""
        cand = self.rng.choice(len(inds), size=k, replace=False)
        best = cand[0]
        for idx in cand[1:]:
            if scores[idx] > scores[best]:
                best = idx
        return inds[int(best)]


    def _update_player_archives(self, pop: List[Individual], fronts: List[List[int]], scores: np.ndarray) -> None:
        """Update player heat/count archives using normalized reward weights.

        The paper normalizes R_i over the population to obtain weights omega_i.
        Here all evaluated individuals in the current population contribute to
        the heat archive, while better individuals contribute larger weights.
        """
        if len(pop) == 0:
            return

        scores_safe = np.asarray(scores, dtype=float)
        scores_safe = np.nan_to_num(scores_safe, nan=0.0, posinf=1.0, neginf=0.0)
        scores_safe = np.maximum(scores_safe, 0.0)
        omega = scores_safe / (float(np.sum(scores_safe)) + 1e-12)

        for idx, w_i in enumerate(omega):
            bins = gene_to_player_bins(pop[idx].gene, K_cont=self.p.K_cont_bins)
            for j, b in enumerate(bins):
                self.heat[j][b] = self.heat[j].get(b, 0.0) + float(w_i)
                self.count[j][b] = self.count[j].get(b, 0) + 1


    def _build_pools(self) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Return (hot_bins, cold_bins, norm_bins) per dimension."""
        hot_bins: List[List[int]] = [[] for _ in range(24)]
        cold_bins: List[List[int]] = [[] for _ in range(24)]
        norm_bins: List[List[int]] = [[] for _ in range(24)]

        # define bin universe sizes (consistent with gene_to_player_bins)
        Kc = self.p.K_cont_bins
        universe_sizes = [
            len(SPACE.x1_resample),
            len(SPACE.x2_pool),
            len(SPACE.x3_Lp),
            len(SPACE.x4_batch),
            len(SPACE.x5_norm),
            len(SPACE.x6_C0),
            len(SPACE.x7_C1),
            len(SPACE.x8_C2),
            len(SPACE.x9_C3),
            len(SPACE.x10_short),
            len(SPACE.x11_long),
            len(SPACE.x12_act),
            Kc, Kc, Kc,
            len(SPACE.x16_sched),
            len(SPACE.x17_sched_type),
            len(SPACE.x18_loss),
            len(SPACE.x19_pair),
            len(SPACE.x20_weights),
            len(SPACE.x21_adapt_rate),
            len(SPACE.x22_fuse),
            len(SPACE.x23_weight_mode),
            len(SPACE.x24_cross_mode),
        ]

        for j in range(24):
            U = list(range(universe_sizes[j]))
            # ensure keys exist
            for b in U:
                self.heat[j].setdefault(b, 0.0)
                self.count[j].setdefault(b, 0)

            # hot by heat
            heat_sorted = sorted(U, key=lambda b: self.heat[j].get(b, 0.0), reverse=True)
            nhot = max(1, int(math.ceil(self.p.hot_ratio_q * len(U))))
            hot = heat_sorted[:nhot]

            # cold by low activity (appearance count)
            count_sorted = sorted(U, key=lambda b: (self.count[j].get(b, 0), self.heat[j].get(b, 0.0)))
            ncold = max(1, int(math.ceil(self.p.low_activity_ratio_p * len(U))))
            cold = count_sorted[:ncold]

            hot_bins[j] = hot
            cold_bins[j] = cold
            norm_bins[j] = [b for b in U if b not in hot]
            if not norm_bins[j]:
                norm_bins[j] = U
        return hot_bins, cold_bins, norm_bins

    def _assemble_from_pool(self, pool_bins: List[List[int]]) -> np.ndarray:
        bins = [int(self.rng.choice(pool_bins[j])) for j in range(24)]
        return bins_to_gene_from_pool(bins, self.rng, K_cont=self.p.K_cont_bins)

    def _cross_pool_mutate(self, gene: np.ndarray, hot_bins, cold_bins, norm_bins) -> np.ndarray:
        if self.rng.random() >= self.p.cross_pool_mut_prob_e:
            return gene
        # mutate one dimension by sampling from the cold pool (exploration)
        j = int(self.rng.integers(0, 24))
        b = int(self.rng.choice(cold_bins[j]))
        bins = gene_to_player_bins(gene, K_cont=self.p.K_cont_bins)
        bins[j] = b
        return bins_to_gene_from_pool(bins, self.rng, K_cont=self.p.K_cont_bins)

    def _make_offspring(self, pop: List[Individual], scores: np.ndarray, t: int) -> List[Individual]:
        rho_par, rho_hot, rho_nh = self.stage_ratio(t)
        N = self.p.pop_size
        n_par = int(round(rho_par * N))
        n_hot = int(round(rho_hot * N))
        n_nh = max(0, N - n_par - n_hot)

        # Elites by highest reward scores.
        order = np.argsort(-scores)
        elites = [pop[int(i)] for i in order[:n_par]]
        off: List[Individual] = [
            Individual(gene=e.gene.copy(), F=e.F.copy(), canon=e.canon, x_values=e.x_values)
            for e in elites
        ]

        hot_bins, cold_bins, norm_bins = self._build_pools()

        def gen_child(use_hot_pool: bool) -> Individual:
            # Parent-based SBX/PM + pool-based assembly as a bias.
            p1 = self._tournament(pop, scores)
            p2 = self._tournament(pop, scores)
            child = p1.gene.copy()
            if self.rng.random() < self.p.pc:
                c1, _c2 = sbx_crossover(p1.gene, p2.gene, self.p.eta_c, self.rng)
                child = c1
            if self.rng.random() < self.p.pm:
                child = polynomial_mutation(
                    child,
                    eta_m=self.p.eta_m,
                    p_dim=1.0 / 24.0,
                    mmax=min(self.p.mmax, 24),
                    rng=self.rng,
                )

            # Pool-biased replacement per dimension.
            if use_hot_pool:
                bias_gene = self._assemble_from_pool(hot_bins)
            else:
                # Non-hot assembly with cold compensation for exploration.
                bins = []
                for j in range(24):
                    if self.rng.random() < self.p.cold_comp_gain_o:
                        bins.append(int(self.rng.choice(cold_bins[j])))
                    else:
                        bins.append(int(self.rng.choice(norm_bins[j])))
                bias_gene = bins_to_gene_from_pool(bins, self.rng, K_cont=self.p.K_cont_bins)

            mix = 0.5
            child = _clip01_vec(mix * child + (1.0 - mix) * bias_gene)
            child = self._cross_pool_mutate(child, hot_bins, cold_bins, norm_bins)
            return Individual(gene=child)

        for _ in range(n_hot):
            off.append(gen_child(use_hot_pool=True))
        for _ in range(n_nh):
            off.append(gen_child(use_hot_pool=False))
        return off


    def _maybe_eval(self, ind: Individual) -> bool:
        """Evaluate if not duplicated; return True if evaluated (or already had F)."""
        if ind.F is not None and ind.canon is not None:
            return True
        # retry-budget deduplication
        for _ in range(self.p.n_trial_dedup):
            F, canon, xvals = self.eval.evaluate(ind)
            if canon in self.seen_canons:
                # resample gene slightly and retry
                ind.gene = _clip01_vec(ind.gene + self.rng.normal(0.0, 0.02, size=ind.gene.shape))
                continue
            ind.F = F
            ind.canon = canon
            ind.x_values = xvals
            self.seen_canons.add(canon)
            return True
        # give up evaluation slot
        return False

    def _update_early_stop_stats(self, pop: List[Individual]) -> None:
        F = np.stack([i.F for i in pop if i.F is not None], axis=0)
        fronts = fast_non_dominated_sort(F)
        ndF = F[fronts[0]] if fronts else F
        best_f1 = float(np.min(ndF[:, 0]))
        best_f2 = float(np.min(ndF[:, 1]))
        self.best_f1_hist.append(best_f1)
        self.best_f2_hist.append(best_f2)

        if self.hv_ref is None:
            # fixed ref from initial population
            r1 = 1.1 * float(np.max(F[:, 0]))
            r2 = 1.1 * float(np.max(F[:, 1]))
            self.hv_ref = (r1, r2)
        hv = hv_2d_min(ndF, self.hv_ref)
        self.hv_hist.append(hv)

    def _check_early_stop(self, t: int) -> bool:
        W = self.p.W
        if t < W:
            return False
        eps0 = self.p.eps0

        def rel_gain(old: float, new: float) -> float:
            # minimization: improvement is old-new
            return max(0.0, (old - new) / max(abs(old), eps0))

        df1 = rel_gain(self.best_f1_hist[-W - 1], self.best_f1_hist[-1])
        df2 = rel_gain(self.best_f2_hist[-W - 1], self.best_f2_hist[-1])
        old_hv = self.hv_hist[-W - 1]
        new_hv = self.hv_hist[-1]
        dhv = max(0.0, (new_hv - old_hv) / max(abs(old_hv), eps0))

        return (df1 < self.p.eps1) and (df2 < self.p.eps2) and (dhv < self.p.eps_hv)

    def run(self) -> Dict[str, object]:
        start = time.time()
        # Base population
        pop: List[Individual] = [Individual(gene=self.rng.random(24)) for _ in range(self.p.pop_size)]
        # Evaluate initial population
        pop2: List[Individual] = []
        for ind in pop:
            ok = self._maybe_eval(ind)
            if ok and ind.F is not None:
                pop2.append(ind)
        # If too many duplicates, refill
        while len(pop2) < self.p.pop_size:
            ind = Individual(gene=self.rng.random(24))
            if self._maybe_eval(ind) and ind.F is not None:
                pop2.append(ind)
        pop = pop2[: self.p.pop_size]

        # Init early stop stats (ref point fixed here)
        self._update_early_stop_stats(pop)

        for t in range(1, self.p.max_gens + 1):
            # current objective matrix
            F = np.stack([i.F for i in pop], axis=0)
            selected, rank, crowd = nsga2_select(F, self.p.pop_size)
            # scores/rewards (larger is better)
            scores = self._compute_scores(F, rank, crowd, t)
            # update player archives using ND set
            fronts = fast_non_dominated_sort(F)
            self._update_player_archives(pop, fronts, scores)

            # offspring
            off = self._make_offspring(pop, scores, t)
            # evaluate offspring (skip unevaluated slots if retry exhausted)
            evaled_off: List[Individual] = []
            for ind in off:
                if ind.F is not None:
                    evaled_off.append(ind)
                    continue
                if self._maybe_eval(ind) and ind.F is not None:
                    evaled_off.append(ind)
            # Combine and select next generation
            comb = pop + evaled_off
            F_comb = np.stack([i.F for i in comb], axis=0)
            sel_idx, _rank2, _crowd2 = nsga2_select(F_comb, self.p.pop_size)
            pop = [comb[i] for i in sel_idx]

            # update early stop trajectories
            self._update_early_stop_stats(pop)
            if self._check_early_stop(t):
                break

            if (t % 1) == 0:
                ndF = np.stack([i.F for i in pop], axis=0)
                nd_fronts = fast_non_dominated_sort(ndF)
                nd = ndF[nd_fronts[0]] if nd_fronts else ndF
                print(
                    f"[Gen {t:02d}] ND={len(nd):3d} best_f1={float(np.min(nd[:,0])):.6g} best_f2={float(np.min(nd[:,1])):.0f} HV={self.hv_hist[-1]:.6g}"
                )

        # Final front
        F_final = np.stack([i.F for i in pop], axis=0)
        fronts = fast_non_dominated_sort(F_final)
        nd_idx = fronts[0] if fronts else list(range(len(pop)))
        nd_pop = [pop[i] for i in nd_idx]
        nd_F = np.stack([i.F for i in nd_pop], axis=0)

        elapsed = time.time() - start
        result = {
            "seed": self.seed,
            "elapsed_sec": elapsed,
            "generations_ran": len(self.best_f1_hist) - 1,
            "evals": len(self.seen_canons),
            "hv_ref": self.hv_ref,
            "pareto_F": nd_F.tolist(),
            "pareto_x": [i.x_values for i in nd_pop],
        }
        with open(os.path.join(self.out_dir, f"phmoea_real_seed{self.seed}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


def _clip01_vec(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def build_base_cfg(seed: int) -> ms_bcnn.Config:
    """Base (fixed) training/data protocol that is *not* searched (Table 4 does not include them)."""
    cfg = ms_bcnn.Config(
        split_mode="sequential",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        split_seed=seed,
        train_shuffle=False,
        # keep max epochs / early stop settings fixed
        max_epochs=200,
        patience=15,
        # output targets fixed (real dataset)
        Lf=5,
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="all_data.pkl", help="Path to all_data.pkl")
    ap.add_argument("--out_dir", type=str, default="./runs_phmoea_real", help="Output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=30)
    args = ap.parse_args()

    set_seed(args.seed)
    with open(args.data, "rb") as f:
        dataset = pickle.load(f)
    assert isinstance(dataset, list), "Dataset must be a list of dict records."
    base_cfg = build_base_cfg(args.seed)

    evaluator = RealEvaluator(dataset=dataset, base_cfg=base_cfg, verbose=False)
    params = PHMOEAParams(pop_size=int(args.pop), max_gens=int(args.gens))
    algo = PHMOEAReal(evaluator=evaluator, params=params, seed=args.seed, out_dir=args.out_dir)
    res = algo.run()
    print("Done. Saved:", os.path.join(args.out_dir, f"phmoea_real_seed{args.seed}.json"))
    print("Pareto size:", len(res["pareto_F"]))


if __name__ == "__main__":
    main()

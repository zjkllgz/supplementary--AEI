
from __future__ import annotations
import os, time, json, argparse, datetime as _dt
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Embedded H-DTLZ2/7 benchmark suite (standalone)
# ============================================================

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import math

import numpy as np


# ------------------------- common projection -------------------------


@dataclass
class HdSuiteConfig:
    neutral: float = 0.5
    gamma_coupling: float = 0.5
    coupling_topology: str = "binary_tree"  # "chain" or "binary_tree"


class _HdProjector:
    """Shared genotype->(z, active_mask) projection for hierarchical benchmarks."""

    def __init__(self, decision_vars: Any, cfg: Optional[HdSuiteConfig] = None):
        self.decision_vars = decision_vars
        self.cfg = cfg or HdSuiteConfig()

    @staticmethod
    def _clip01(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    @staticmethod
    def _x_from_vec(vec: np.ndarray, info: Dict) -> float:
        """Map a variable to a scalar x in [0,1].

        - continuous: use normalized gene directly
        - discrete: map to nearest discrete index (piecewise constant)
        - categorical: argmax one-hot slice -> idx/(L-1)
        """
        t = info.get("type")
        if t == "continuous":
            return float(vec[info["index"]])
        if t == "discrete":
            norm = float(vec[info["index"]])
            values = info.get("values", [])
            if not values:
                return norm
            idx = int(round(norm * (len(values) - 1)))
            idx = max(0, min(len(values) - 1, idx))
            return float(idx / max(1, (len(values) - 1)))
        if t == "categorical":
            idxs = info.get("index")
            if not idxs:
                return 0.5
            # idxs can be a list of indices (one-hot slice)
            k = int(np.argmax(vec[idxs]))
            return float(k / max(1, (len(idxs) - 1)))
        return 0.5

    def _parent_index_tail(self, j: int) -> int:
        """Parent index for coupling *within the tail* (indices >=1).

        We deliberately **exclude z[0]** (the trade-off angle/coordinate) from
        the coupling graph so that the classic DTLZ Pareto fronts remain valid.

        Let tail indices be t = j-1 for j>=1.
        - chain: parent_t = t-1
        - binary_tree: parent_t = floor((t-1)/2)
        Then map back: parent_j = parent_t + 1.
        """
        if j <= 1:
            return 1
        t = j - 1
        if self.cfg.coupling_topology == "chain":
            parent_t = t - 1
        else:
            parent_t = (t - 1) // 2
        return int(parent_t + 1)

    def project(
        self,
        vec: np.ndarray,
        encoder: Any,
        base_cfg: Any,
        repair_fn: Optional[Callable[[Any], Any]] = None,
    ) -> tuple[list[float], list[int], Any]:
        """Return (z_list, active_mask, cfg_obj)."""
        cfg_obj = encoder.decode(vec, base_cfg)
        if repair_fn is not None:
            cfg_obj = repair_fn(cfg_obj)

        var_names: List[str] = list(getattr(encoder, "variable_names"))
        encoding_info: Dict[str, Dict] = getattr(encoder, "encoding_info")

        z: List[float] = []
        active_mask: List[int] = []

        for vn in var_names:
            info = encoding_info[vn]

            # Paper Appendix C.1: Π(·) extracts (and orders) continuous variables; mixed/discrete/categorical
            # variables affect *activation* but do not directly enter z.
            if info.get("type") != "continuous":
                continue

            try:
                active = bool(self.decision_vars.is_variable_active(vn, cfg_obj))
            except Exception:
                active = True

            if not active:
                z.append(self.cfg.neutral)
                active_mask.append(0)
            else:
                xi = self._x_from_vec(vec, info)
                z.append(self._clip01(xi))
                active_mask.append(1)

        # Ensure at least 2 dims for M=2 problems
        if len(z) < 2:
            z = (z + [self.cfg.neutral, self.cfg.neutral])[:2]
            active_mask = (active_mask + [1, 1])[:2]

        return z, active_mask, cfg_obj

    def coupling_term(self, z: list[float], active_mask: list[int]) -> float:
        """Average squared distance between each active var and its parent."""
        g_couple = 0.0
        n_couple = 0
        # Couple only among the tail variables (j>=2), excluding z[0] and leaving z[1] as tail-root.
        for j in range(2, len(z)):
            if active_mask[j] == 1:
                p = self._parent_index_tail(j)
                # p is always in [1, j-1]
                g_couple += (z[j] - z[p]) ** 2
                n_couple += 1
        if n_couple > 0:
            g_couple /= float(n_couple)
        return float(self.cfg.gamma_coupling) * float(g_couple)


# ------------------------- H-DTLZ2 -------------------------


class HdDTLZ2Std:
    """Hierarchical DTLZ2 (M=2) with the true quarter-unit-circle PF."""

    def __init__(self, decision_vars: Any, cfg: Optional[HdSuiteConfig] = None):
        self.proj = _HdProjector(decision_vars, cfg)

    def evaluate(
        self,
        vec: np.ndarray,
        encoder: Any,
        base_cfg: Any,
        repair_fn: Optional[Callable[[Any], Any]] = None,
    ) -> tuple[float, float]:
        z, active_mask, _ = self.proj.project(vec, encoder, base_cfg, repair_fn)
        n = len(z)

        # Standard DTLZ2 g term over j>=2 (index 1..), averaged for stable scale
        g_base = 0.0
        for j in range(1, n):
            g_base += (z[j] - 0.5) ** 2
        g_base /= float(max(1, (n - 1)))

        g = g_base + self.proj.coupling_term(z, active_mask)
        z1 = float(min(1.0, max(0.0, z[0])))
        f1 = (1.0 + g) * math.cos(0.5 * math.pi * z1)
        f2 = (1.0 + g) * math.sin(0.5 * math.pi * z1)
        return float(f1), float(f2)


def true_pf_dtlz2(n: int = 2000) -> np.ndarray:
    """True PF for 2-objective DTLZ2 (g=0): quarter unit circle."""
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.cos(0.5 * math.pi * t), np.sin(0.5 * math.pi * t)], axis=1)


# ------------------------- H-DTLZ7 -------------------------


class HdDTLZ7Std:
    """Hierarchical DTLZ7 (M=2) with a normalized second objective.

For M=2 (classic definition):
  f1 = z1
  g  = 1 + 9/(n-1) * sum_{j=2..n} z_j
  h  = 2 - (f1/g) * (1 + sin(3*pi*f1))
  f2 = g*h

We normalize f2 by 2: \tilde f2 = f2/2, which keeps PF values roughly in [0,1].
"""

    def __init__(self, decision_vars: Any, cfg: Optional[HdSuiteConfig] = None):
        self.proj = _HdProjector(decision_vars, cfg)

    def evaluate(
        self,
        vec: np.ndarray,
        encoder: Any,
        base_cfg: Any,
        repair_fn: Optional[Callable[[Any], Any]] = None,
    ) -> tuple[float, float]:
        z, active_mask, _ = self.proj.project(vec, encoder, base_cfg, repair_fn)
        n = len(z)
        z1 = float(min(1.0, max(0.0, z[0])))

        s = 0.0
        for j in range(1, n):
            s += z[j]
        g = 1.0 + 9.0 * s / float(max(1, (n - 1)))
        # add hierarchical coupling (keeps PF reachable at z_j=0.5 for inactive vars)
        g += self.proj.coupling_term(z, active_mask)

        h = 2.0 - (z1 / g) * (1.0 + math.sin(3.0 * math.pi * z1))
        f1 = z1
        f2 = g * h
        f2 = f2 / 2.0
        return float(f1), float(f2)


def true_pf_dtlz7(n: int = 8000) -> np.ndarray:
    """Approximate PF for (normalized) DTLZ7 with M=2.

We sample t in [0,1] for f1=t, and use g=1 (achieved when z_j=0 for j>=2).
Then f2 = (g*h)/2 = h/2.
We keep only non-dominated points (minimization) to approximate the disconnected PF.
"""
    t = np.linspace(0.0, 1.0, n)
    f1 = t
    g = 1.0
    h = 2.0 - (f1 / g) * (1.0 + np.sin(3.0 * math.pi * f1))
    f2 = (g * h) / 2.0
    pts = np.stack([f1, f2], axis=1)

    # filter non-dominated (minimization)
    nd = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        if not nd[i]:
            continue
        for j in range(pts.shape[0]):
            if i == j or not nd[i]:
                continue
            a = pts[j]
            b = pts[i]
            if (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1]):
                nd[i] = False
                break
    return pts[nd]


# ------------------------- factories -------------------------


def make_hd_problem(
    name: str,
    decision_vars: Any,
    gamma: float = 0.5,
    topology: str = "binary_tree",
) -> Any:
    """Factory for H-DTLZ2/7 (M=2)."""
    # IMPORTANT: the neutral value for INACTIVE tail variables must match the
    # Pareto-set neutral of the base DTLZ problem:
    # - DTLZ2  : z_j=0.5 (j>=2)
    # - DTLZ7  : z_j=0   (j>=2) to achieve g=1
    neutral = 0.5
    if name.strip().lower().endswith("7") or "dtlz7" in name.strip().lower():
        neutral = 0.0
    cfg = HdSuiteConfig(neutral=float(neutral), gamma_coupling=float(gamma), coupling_topology=str(topology))
    key = name.strip().lower()
    if key in {"hd-dtlz2", "h-dtlz2", "hd_dtlz2", "h_dtlz2", "dtlz2"}:
        return HdDTLZ2Std(decision_vars, cfg)
    if key in {"hd-dtlz7", "h-dtlz7", "hd_dtlz7", "h_dtlz7", "dtlz7"}:
        return HdDTLZ7Std(decision_vars, cfg)
    raise ValueError(f"Unknown problem name: {name}")


def true_pf(name: str, n: int = 4000) -> np.ndarray:
    """Reference PF sampler for IGD."""
    key = name.strip().lower()
    if key in {"hd-dtlz2", "h-dtlz2", "hd_dtlz2", "h_dtlz2", "dtlz2"} or key.endswith("2"):
        return true_pf_dtlz2(n)
    if key in {"hd-dtlz7", "h-dtlz7", "hd_dtlz7", "h_dtlz7", "dtlz7"} or key.endswith("7"):
        return true_pf_dtlz7(n)
    raise ValueError(f"Unknown problem name: {name}")

# ============================================================
# Embedded legacy H-DTLZ2 (unused; kept for completeness)
# ============================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import math
import numpy as np


@dataclass
class HdDTLZConfigStdLegacy:
    # Neutral value for inactive variables in the *phenotype* x-vector
    neutral: float = 0.5

    # Coupling strength for active variables (hierarchical structure)
    gamma_coupling: float = 0.5

    # Coupling topology among variables: "chain" or "binary_tree"
    coupling_topology: str = "binary_tree"


class HdDTLZ2StdLegacy:
    """Hierarchical DTLZ2 with 2 objectives, standard DTLZ2 scale and true Pareto front.

    Objectives:
        f1 = (1+g) * cos(pi/2 * x1)
        f2 = (1+g) * sin(pi/2 * x1)

    where g >= 0 and g=0 is achievable:
      - For ACTIVE vars j>=2: x_j = 0.5
      - For INACTIVE vars j>=2: x_j is forced to 0.5 (neutral), so does not affect g
      - Coupling term also becomes 0 at x_j=0.5

    True PF (when g=0): quarter unit circle.
    """

    def __init__(self, decision_vars: Any, cfg: Optional[HdDTLZConfigStdLegacy] = None):
        self.decision_vars = decision_vars
        self.cfg = cfg or HdDTLZConfigStdLegacy()

    @staticmethod
    def _clip01(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    @staticmethod
    def _x_from_vec(vec: np.ndarray, info: Dict) -> float:
        """Map a variable to a scalar x in [0,1].

        - continuous: use normalized gene directly
        - discrete: map to nearest discrete index (piecewise constant)
        - categorical: argmax one-hot slice -> idx/(L-1)
        """
        t = info["type"]
        if t == "continuous":
            return float(vec[info["index"]])
        if t == "discrete":
            norm = float(vec[info["index"]])
            values = info.get("values", [])
            if not values:
                return norm
            idx = int(round(norm * (len(values) - 1)))
            idx = max(0, min(len(values) - 1, idx))
            return float(idx / max(1, (len(values) - 1)))
        if t == "categorical":
            idxs = info["index"]
            if not idxs:
                return 0.5
            k = int(np.argmax(vec[idxs]))
            return float(k / max(1, (len(idxs) - 1)))
        return 0.5

    def _parent_index(self, j: int) -> int:
        """Return parent index for variable j (0-based indexing)."""
        if j <= 0:
            return 0
        topo = self.cfg.coupling_topology
        if topo == "chain":
            return j - 1
        # binary_tree
        return (j - 1) // 2

    def evaluate(
        self,
        vec: np.ndarray,
        encoder: Any,
        base_cfg: Any,
        repair_fn: Optional[Callable[[Any], Any]] = None,
    ) -> tuple[float, float]:
        """
        Evaluate a candidate vector (genotype) to objectives (f1,f2).
        Uses encoder.decode(vec, base_cfg) to obtain cfg and activation masks.
        """
        # decode -> cfg object
        cfg_obj = encoder.decode(vec, base_cfg)
        if repair_fn is not None:
            cfg_obj = repair_fn(cfg_obj)

        var_names: List[str] = list(getattr(encoder, "variable_names"))
        encoding_info: Dict[str, Dict] = getattr(encoder, "encoding_info")

        x: List[float] = []
        active_mask: List[int] = []

        for vn in var_names:
            info = encoding_info[vn]
            try:
                active = bool(self.decision_vars.is_variable_active(vn, cfg_obj))
            except Exception:
                active = True

            if not active:
                x.append(self.cfg.neutral)
                active_mask.append(0)
            else:
                xi = self._x_from_vec(vec, info)
                x.append(self._clip01(xi))
                active_mask.append(1)

        if len(x) < 2:
            # degenerate, but keep safe
            x = (x + [self.cfg.neutral, self.cfg.neutral])[:2]
            active_mask = (active_mask + [1, 1])[:2]

        # Standard DTLZ2 g term over j>=2 (index 1..)
        g_base = 0.0
        for j in range(1, len(x)):
            # inactive vars fixed to neutral => contribute 0
            g_base += (x[j] - 0.5) ** 2
        # Normalize to keep standard DTLZ2-like scale even when the encoding has many variables.
        g_base /= float(max(1, (len(x) - 1)))

        # Hierarchical coupling among ACTIVE vars (j>=2)
        g_couple = 0.0
        n_couple = 0
        for j in range(1, len(x)):
            if active_mask[j] == 1:
                p = self._parent_index(j)
                g_couple += (x[j] - x[p]) ** 2
                n_couple += 1
        if n_couple > 0:
            g_couple /= float(n_couple)
        g_couple *= float(self.cfg.gamma_coupling)

        g = g_base + g_couple

        x1 = self._clip01(x[0])
        f1 = (1.0 + g) * math.cos(0.5 * math.pi * x1)
        f2 = (1.0 + g) * math.sin(0.5 * math.pi * x1)
        return float(f1), float(f2)


def true_pf_dtlz2_legacy(n: int = 2000) -> np.ndarray:
    """True Pareto front for 2-objective DTLZ2 (g=0): quarter unit circle."""
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.cos(0.5 * math.pi * t), np.sin(0.5 * math.pi * t)], axis=1)


def get_default_hd_dtlz2_std_legacy(decision_vars: Any, gamma: float = 0.5, topology: str = "binary_tree") -> HdDTLZ2StdLegacy:
    cfg = HdDTLZConfigStdLegacy(gamma_coupling=float(gamma), coupling_topology=str(topology))
    return HdDTLZ2StdLegacy(decision_vars, cfg)



# ============================================================
# 统一的混合/层级决策空间（与 baselines 统一）
# ============================================================

@dataclass
class _Cfg:
    optimizer: str = "adam"
    use_scheduler: int = 0
    model: str = "A"
    lr: float = 0.5
    adam_beta1: float = 0.5
    adam_beta2: float = 0.5
    sgd_momentum: float = 0.5
    gamma: float = 0.5
    dropout: float = 0.5
    wd: float = 0.5
    batch_size: int = 0
    act: str = "relu"
    depth: int = 0
    aug: int = 0


class SyntheticDecisionVariables:
    def __init__(self):
        self._vars: Dict[str, Dict[str, Any]] = {
            # 与 baselines 保持一致（尤其是层级激活规则依赖 optimizer/use_scheduler）。
            "optimizer": {"type": "categorical", "values": ["adam", "sgd", "rmsprop"]},
            "model": {"type": "categorical", "values": ["A", "B", "C"]},
            "act": {"type": "categorical", "values": ["relu", "tanh", "gelu"]},
            "use_scheduler": {"type": "discrete", "values": [0, 1]},
            "batch_size": {"type": "discrete", "values": [16, 32, 64, 128, 256]},
            "depth": {"type": "discrete", "values": [2, 3, 4, 5, 6]},
            "aug": {"type": "discrete", "values": [0, 1, 2, 3]},
            # continuous 都在 [0,1] 上定义；尺度字段仅用于说明（suite 内不强依赖）。
            "lr": {"type": "continuous", "bounds": (0.0, 1.0), "scale": "log"},
            "adam_beta1": {"type": "continuous", "bounds": (0.0, 1.0), "scale": "lin"},
            "adam_beta2": {"type": "continuous", "bounds": (0.0, 1.0), "scale": "lin"},
            "sgd_momentum": {"type": "continuous", "bounds": (0.0, 1.0), "scale": "lin"},
            "gamma": {"type": "continuous", "bounds": (0.0, 1.0), "scale": "lin"},
            "dropout": {"type": "continuous", "bounds": (0.0, 1.0), "scale": "lin"},
            "wd": {"type": "continuous", "bounds": (0.0, 1.0), "scale": "log"},
        }
        self._order = [
            "lr", "optimizer", "adam_beta1", "adam_beta2", "sgd_momentum",
            "use_scheduler", "gamma", "model", "dropout", "wd",
            "batch_size", "act", "depth", "aug"
        ]

    def get_all_variable_names(self) -> List[str]:
        return list(self._order)

    def get_variable_info(self, var_name: str) -> Dict[str, Any]:
        return dict(self._vars[var_name])

    # 关键：层级激活规则必须与 baselines 完全一致，否则 H-DTLZ 的投影 Π(·) 会改变，
    # 导致目标值分布不同，从而出现“PHLMOEA 开头 IGD 巨大、不与对比算法一致”的现象。
    def is_variable_active(self, var_name: str, cfg_obj) -> bool:
        """Match the in-file _HdProjector: is_variable_active(var_name, cfg_obj).
        cfg_obj may be an object with attributes (cfg.optimizer) or a dict.
        """
        def _get(name, default=None):
            if isinstance(cfg_obj, dict):
                return cfg_obj.get(name, default)
            return getattr(cfg_obj, name, default)

        if var_name in ("adam_beta1", "adam_beta2"):
            return _get("optimizer", "adam") == "adam"
        if var_name == "sgd_momentum":
            return _get("optimizer", "adam") == "sgd"
        if var_name == "gamma":
            return int(_get("use_scheduler", 0)) == 1
        return True


class SyntheticEncoder:
    """Mixed encoding: continuous/discrete scalar in [0,1], categorical one-hot."""
    def __init__(self, dvars: SyntheticDecisionVariables, seed: int = 0):
        self.dvars = dvars
        self.rng = np.random.default_rng(seed)
        self.variable_names = dvars.get_all_variable_names()
        self.encoding_info: Dict[str, Dict[str, Any]] = {}

        dim = 0
        for vn in self.variable_names:
            info = dvars.get_variable_info(vn)
            if info["type"] in ("continuous", "discrete"):
                self.encoding_info[vn] = {
                    "type": info["type"], "index": dim, "values": info.get("values", []),
                }
                dim += 1
            elif info["type"] == "categorical":
                k = len(info["values"])
                idxs = list(range(dim, dim + k))
                self.encoding_info[vn] = {"type": "categorical", "index": idxs, "values": info["values"]}
                dim += k
            else:
                raise ValueError(f"Unknown type: {info['type']}")
        self.dim = dim

        # 组合生成的“模块组”：用于 group-wise 拼装
        self.groups: Dict[str, List[str]] = {
            "opt_core": ["lr", "optimizer", "adam_beta1", "adam_beta2", "sgd_momentum"],
            "sched": ["use_scheduler", "gamma"],
            "modeling": ["model", "act", "depth", "aug"],
            "regular": ["dropout", "wd"],
            "train": ["batch_size"],
        }
        self.group_indices: Dict[str, List[int]] = {}
        for gn, vns in self.groups.items():
            idxs: List[int] = []
            for vn in vns:
                info = self.encoding_info[vn]
                if info["type"] in ("continuous", "discrete"):
                    idxs.append(int(info["index"]))
                else:
                    idxs.extend([int(j) for j in info["index"]])
            self.group_indices[gn] = idxs

    def initialize_population(self, n: int) -> np.ndarray:
        X = np.zeros((n, self.dim), dtype=float)
        for i in range(n):
            for vn in self.variable_names:
                info = self.encoding_info[vn]
                if info["type"] in ("continuous", "discrete"):
                    X[i, info["index"]] = float(self.rng.random())
                else:
                    idxs = info["index"]
                    k = len(idxs)
                    j = int(self.rng.integers(0, k))
                    X[i, idxs] = 0.0
                    X[i, idxs[j]] = 1.0
        return X

    # ✅ REQUIRED by the in-file _HdProjector
    def decode(self, vec, base_cfg=None):
        """
        vec: encoded decision vector (float array)
        base_cfg: a config-like object (can be None)
        return: config object with fields filled
        """
        # 复制 base_cfg，避免就地改写
        if base_cfg is None:
            cfg = _Cfg()
        else:
            cfg = _Cfg(**base_cfg.__dict__)

        for vn in self.variable_names:
            info = self.encoding_info[vn]

            if info["type"] == "continuous":
                j = int(info["index"])
                setattr(cfg, vn, float(np.clip(vec[j], 0.0, 1.0)))

            elif info["type"] == "discrete":
                j = int(info["index"])
                values = info.get("values", [])
                t = float(np.clip(vec[j], 0.0, 1.0))
                if values:
                    idx = int(round(t * (len(values) - 1)))
                    idx = max(0, min(idx, len(values) - 1))
                    setattr(cfg, vn, values[idx])
                else:
                    setattr(cfg, vn, int(round(t)))

            else:  # categorical (one-hot)
                idxs = info["index"]
                vals = info.get("values", [])
                onehot = np.asarray([vec[k] for k in idxs], dtype=float)
                k = int(np.argmax(onehot)) if len(onehot) else 0
                k = max(0, min(k, len(vals) - 1))
                setattr(cfg, vn, vals[k] if vals else None)

        return cfg



def repair_vector(x: np.ndarray, encoder: SyntheticEncoder) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    y = np.clip(y, 0.0, 1.0)
    for vn, info in encoder.encoding_info.items():
        if info["type"] == "categorical":
            idxs = info["index"]
            sl = y[idxs]
            j = int(np.argmax(sl)) if len(sl) else 0
            y[idxs] = 0.0
            y[idxs[j]] = 1.0
    return y


# ============================================================
# Evaluator
# ============================================================

class HdDTLZEvaluator:
    def __init__(self, bench_name: str, decision_vars: SyntheticDecisionVariables):
        self.bench_name = bench_name
        self.problem = make_hd_problem(bench_name, decision_vars)
        self.base_cfg = _Cfg()

    def evaluate(self, vec: np.ndarray, encoder: SyntheticEncoder) -> Tuple[float, float]:
        f1, f2 = self.problem.evaluate(vec, encoder, self.base_cfg, repair_fn=None)
        return float(f1), float(f2)


# ============================================================
# Metrics + ND
# ============================================================

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def nondominated(F: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return F
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if dominates(F[j], F[i]):
                keep[i] = False
                break
    return F[keep]

def igd(F: np.ndarray, P: np.ndarray) -> float:
    if F.size == 0:
        return float("inf")
    d2 = ((P[:, None, :] - F[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min(axis=1)).mean())

def hv_2d_min(F: np.ndarray, ref: Tuple[float, float]) -> float:
    if F.size == 0:
        return 0.0
    pts = F.copy()
    pts = pts[(pts[:, 0] <= ref[0]) & (pts[:, 1] <= ref[1])]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in pts:
        if f2 < prev_f2:
            hv += max(0.0, ref[0] - f1) * max(0.0, prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


# ============================================================
# NSGA-II selection
# ============================================================

def fast_non_dominated_sort(F: np.ndarray):
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    rank = np.zeros(N, dtype=int)
    fronts = [[]]

    def dom(a, b): return np.all(a <= b) and np.any(a < b)

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dom(F[p], F[q]):
                S[p].append(q)
            elif dom(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts, rank

def crowding_distance(F: np.ndarray, front: List[int]):
    if len(front) == 0:
        return {}
    m = F.shape[1]
    dist = {i: 0.0 for i in front}
    for k in range(m):
        vals = [(i, float(F[i, k])) for i in front]
        vals.sort(key=lambda x: x[1])
        dist[vals[0][0]] = float("inf")
        dist[vals[-1][0]] = float("inf")
        fmin = vals[0][1]
        fmax = vals[-1][1]
        if fmax - fmin < 1e-12:
            continue
        for j in range(1, len(vals) - 1):
            dist[vals[j][0]] += (vals[j + 1][1] - vals[j - 1][1]) / (fmax - fmin)
    return dist

def nsga2_select(X: np.ndarray, F: np.ndarray, N: int):
    fronts, _ = fast_non_dominated_sort(F)
    sel = []
    for front in fronts:
        if len(sel) + len(front) <= N:
            sel.extend(front)
        else:
            cd = crowding_distance(F, front)
            front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
            sel.extend(front_sorted[: (N - len(sel))])
            break
    return X[sel].copy(), F[sel].copy()

def tournament_pick(F: np.ndarray, rng: np.random.Generator, k: int = 2):
    idx = rng.integers(0, F.shape[0], size=k)
    fronts, rank = fast_non_dominated_sort(F)
    cd_all = np.zeros(F.shape[0], dtype=float)
    for fr in fronts:
        cd = crowding_distance(F, fr)
        for i, v in cd.items():
            cd_all[i] = v
    best = int(idx[0])
    for j in idx[1:]:
        j = int(j)
        if rank[j] < rank[best]:
            best = j
        elif rank[j] == rank[best] and cd_all[j] > cd_all[best]:
            best = j
    return best, cd_all, fronts


# ============================================================
# Variation operators
# ============================================================

def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float, rng: np.random.Generator, p_c: float = 0.9):
    if rng.random() > p_c:
        return p1.copy(), p2.copy()
    u = rng.random(size=p1.shape[0])
    beta = np.where(u <= 0.5, (2*u) ** (1.0/(eta+1.0)), (1.0/(2*(1-u))) ** (1.0/(eta+1.0)))
    c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
    c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
    return c1, c2

def polynomial_mutation(x: np.ndarray, eta: float, pm: float, rng: np.random.Generator):
    y = x.copy()
    for i in range(y.shape[0]):
        if rng.random() < pm:
            u = rng.random()
            if u < 0.5:
                delta = (2*u) ** (1.0/(eta+1.0)) - 1.0
            else:
                delta = 1.0 - (2*(1-u)) ** (1.0/(eta+1.0))
            y[i] = np.clip(y[i] + delta, 0.0, 1.0)
    return y


# ============================================================
# Duplicate filtering
# ============================================================

def vector_key(x: np.ndarray, encoder: SyntheticEncoder, cont_round: int = 4) -> Tuple:
    key = []
    for vn in encoder.variable_names:
        info = encoder.encoding_info[vn]
        if info["type"] == "categorical":
            idxs = info["index"]
            key.append(int(np.argmax(x[idxs])))
        elif info["type"] == "discrete":
            values = info.get("values", [])
            t = float(np.clip(x[int(info["index"])], 0.0, 1.0))
            if values:
                idx = int(round(t * (len(values) - 1)))
                idx = max(0, min(idx, len(values) - 1))
                key.append(idx)
            else:
                key.append(int(round(t)))
        else:
            key.append(round(float(np.clip(x[int(info["index"])], 0.0, 1.0)), cont_round))
    return tuple(key)


# ============================================================
# Schedules
# ============================================================

def linear_schedule(g: int, G: int, start: float, end: float) -> float:
    if G <= 1:
        return float(end)
    t = (g - 1) / float(G - 1)
    return float(start + (end - start) * t)


# ============================================================
# Offspring generators (two streams)
# ============================================================

def pick_front_parent(X: np.ndarray, fr0: List[int], cd_all: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(fr0) == 0:
        return X[int(rng.integers(0, X.shape[0]))]
    idxs = np.array(fr0, dtype=int)
    cd = cd_all[idxs].copy()
    finite = np.isfinite(cd)
    if np.any(finite):
        m = float(np.max(cd[finite]))
        cd = np.where(finite, cd, m)
    else:
        cd = np.ones_like(cd)
    w = cd + 1e-9
    p = w / np.sum(w)
    pick = int(rng.choice(idxs, p=p))
    return X[pick]

def make_offspring_crossover(
    X: np.ndarray, encoder: SyntheticEncoder, rng: np.random.Generator,
    n: int, sbx_eta: float, mut_eta: float, cx_prob: float,
    prefer_front: bool, fr0: List[int], cd_all: np.ndarray,
    pm: float, dup_set: set, cont_round: int, max_trials: int,
):
    Q = []
    trials = 0
    while len(Q) < n and trials < max_trials:
        trials += 1
        if prefer_front:
            p1 = pick_front_parent(X, fr0, cd_all, rng)
            p2 = pick_front_parent(X, fr0, cd_all, rng)
        else:
            i1 = int(rng.integers(0, X.shape[0]))
            i2 = int(rng.integers(0, X.shape[0]))
            p1, p2 = X[i1], X[i2]
        c1, c2 = sbx_crossover(p1, p2, eta=sbx_eta, rng=rng, p_c=cx_prob)
        c = c1 if rng.random() < 0.5 else c2
        c = polynomial_mutation(c, eta=mut_eta, pm=pm, rng=rng)
        c = repair_vector(c, encoder)
        k = vector_key(c, encoder, cont_round=cont_round)
        if k in dup_set:
            continue
        dup_set.add(k)
        Q.append(c)
    return Q

def make_offspring_composition(
    X: np.ndarray, encoder: SyntheticEncoder, rng: np.random.Generator,
    n: int, fr0: List[int], cd_all: np.ndarray,
    noise_scale: float, pm: float, mut_eta: float,
    dup_set: set, cont_round: int, max_trials: int,
):
    Q = []
    trials = 0
    groups = list(encoder.group_indices.keys())
    while len(Q) < n and trials < max_trials:
        trials += 1
        pA = pick_front_parent(X, fr0, cd_all, rng)
        pB = pick_front_parent(X, fr0, cd_all, rng)
        pC = pick_front_parent(X, fr0, cd_all, rng) if rng.random() < 0.5 else None

        child = np.zeros(encoder.dim, dtype=float)
        for gn in groups:
            idxs = encoder.group_indices[gn]
            r = rng.random()
            src = pA if r < 0.45 else (pB if r < 0.90 else (pC if pC is not None else pB))
            child[idxs] = src[idxs]

        # noise on scalar positions (continuous/discrete)
        for vn in encoder.variable_names:
            info = encoder.encoding_info[vn]
            if info["type"] in ("continuous", "discrete"):
                j = int(info["index"])
                child[j] = np.clip(child[j] + float(rng.normal(0.0, noise_scale)), 0.0, 1.0)

        child = polynomial_mutation(child, eta=mut_eta, pm=pm*0.5, rng=rng)
        child = repair_vector(child, encoder)

        k = vector_key(child, encoder, cont_round=cont_round)
        if k in dup_set:
            continue
        dup_set.add(k)
        Q.append(child)
    return Q


# ============================================================
# Main loop
# ============================================================

def run_phlmoea(
    bench: str, pop: int, gens: int, seed: int,
    ref_pf: np.ndarray, refhv: float,
    sbx_eta: float, mut_eta: float, cx_prob: float,
    cross_ratio_start: float, cross_ratio_end: float,
    prefer_front_in_crossover: bool,
    noise_start: float, noise_end: float,
    enable_dup: bool, cont_round: int, max_trials_mult: int,
    out_dir: str = 'ea_results',
    init_dir: Optional[str] = None,      # shared init population dir (optional)
    union_select: bool = True            # elitism: (parents ∪ offspring) -> select N
):
    dvars = SyntheticDecisionVariables()
    encoder = SyntheticEncoder(dvars, seed=seed)
    evaluator = HdDTLZEvaluator(bench, dvars)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Shared initialization (for fair gen-0 alignment across algos)
    # ------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    _init_dir = init_dir or out_dir
    os.makedirs(_init_dir, exist_ok=True)
    init_path = os.path.join(_init_dir, f"initX_{bench}_seed{seed}_pop{pop}_dim{encoder.dim}.npy")
    if os.path.exists(init_path):
        X = np.load(init_path)
    else:
        X = encoder.initialize_population(pop)
        np.save(init_path, X)

    X = np.array([repair_vector(x, encoder) for x in X], dtype=float)
    F = np.array([evaluator.evaluate(X[i], encoder) for i in range(pop)], dtype=float)

    pm = 1.0 / max(1, encoder.dim)

    curve = []
    # archive 只收集“被评估过”的解（与对比算法一致）：init(pop) + 每代 offspring(pop)
    archive_allF = [F.copy()]
    nd0 = nondominated(F)
    curve.append({
        "gen": 0,
        "fes": int(pop),
        "igd": float(igd(nd0, ref_pf)),
        "hv": float(hv_2d_min(nd0, (refhv, refhv))),
        "nd_size": int(nd0.shape[0]),
    })

    for g in range(1, gens + 1):
        # compute fronts + crowding once
        _, cd_all, fronts = tournament_pick(F, rng, k=2)
        fr0 = fronts[0] if fronts else []

        cross_ratio = float(np.clip(linear_schedule(g, gens, cross_ratio_start, cross_ratio_end), 0.0, 1.0))
        n_cross = int(round(cross_ratio * pop))
        n_combo = pop - n_cross

        noise_scale = float(max(0.0, linear_schedule(g, gens, noise_start, noise_end)))

        dup_set = set()
        if enable_dup:
            for i in range(X.shape[0]):
                dup_set.add(vector_key(X[i], encoder, cont_round=cont_round))

        max_trials = max_trials_mult * pop

        # offspring = crossover + composition
        Q = []
        Q.extend(make_offspring_crossover(
            X, encoder, rng, n_cross, sbx_eta, mut_eta, cx_prob,
            prefer_front=prefer_front_in_crossover, fr0=fr0, cd_all=cd_all,
            pm=pm, dup_set=dup_set, cont_round=cont_round, max_trials=max_trials
        ))
        Q.extend(make_offspring_composition(
            X, encoder, rng, n_combo, fr0=fr0, cd_all=cd_all,
            noise_scale=noise_scale, pm=pm, mut_eta=mut_eta,
            dup_set=dup_set, cont_round=cont_round, max_trials=max_trials
        ))

        # fill if de-dup too strict
        while len(Q) < pop:
            x = repair_vector(rng.random(encoder.dim), encoder)
            if enable_dup:
                k = vector_key(x, encoder, cont_round=cont_round)
                if k in dup_set:
                    continue
                dup_set.add(k)
            Q.append(x)

        Q = np.array(Q[:pop], dtype=float)
        FQ = np.array([evaluator.evaluate(Q[i], encoder) for i in range(pop)], dtype=float)

        # ✅ environmental selection (elitism)
        if union_select:
            # (2N -> N): keep elites from parents ∪ offspring
            Ux = np.vstack([X, Q])
            Uf = np.vstack([F, FQ])
            X, F = nsga2_select(Ux, Uf, pop)
        else:
            # (N -> N): non-elitist replacement (next gen = offspring)
            X, F = Q.copy(), FQ.copy()

        archive_allF.append(FQ.copy())

        nd = nondominated(F)
        curve.append({
            "gen": int(g),
            "fes": int(pop * (g + 1)),
            "igd": float(igd(nd, ref_pf)),
            "hv": float(hv_2d_min(nd, (refhv, refhv))),
            "nd_size": int(nd.shape[0]),
        })

        if g % 10 == 0 or g == gens:
            print(f"[{bench}] gen={g:03d}/{gens} FEs={curve[-1]['fes']} cross={n_cross:3d} combo={n_combo:3d} noise={noise_scale:.4f} "
                  f"ND={nd.shape[0]} IGD={curve[-1]['igd']:.6f} HV={curve[-1]['hv']:.6f}")

    allF = np.vstack(archive_allF)
    arch_nd = nondominated(allF)
    return X, F, curve, arch_nd


# ============================================================
# Save helpers (no pandas)
# ============================================================

def save_curve(curve: List[Dict[str, Any]], csv_path: str, png_path: str, title: str):
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("gen,fes,igd,hv,nd_size\n")
        for c in curve:
            f.write(f"{c['gen']},{c.get('fes', '')},{c['igd']},{c['hv']},{c['nd_size']}\n")
    gens = [c["gen"] for c in curve]
    igds = [c["igd"] for c in curve]
    plt.figure()
    plt.plot(gens, igds)
    plt.xlabel("Generation")
    plt.ylabel("IGD")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

def save_front(popF: np.ndarray, ndF: np.ndarray, ref_pf: np.ndarray, png_path: str, title: str, clip_plot: bool, refhv: float):
    plt.figure()
    if ref_pf is not None and ref_pf.size > 0:
        plt.scatter(ref_pf[:, 0], ref_pf[:, 1], s=5, alpha=0.25, label="Reference PF")
    if popF is not None and popF.size > 0:
        plt.scatter(popF[:, 0], popF[:, 1], s=12, alpha=0.25, label="Population (all)")
    if ndF is not None and ndF.size > 0:
        plt.scatter(ndF[:, 0], ndF[:, 1], s=28, alpha=0.9, label="Obtained ND")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if clip_plot:
        plt.xlim(0, refhv)
        plt.ylim(0, refhv)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo_name", type=str, default="PHLMOEA")
    ap.add_argument("--benches", nargs="+", default=["H-DTLZ2", "H-DTLZ7"])
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="ea_results_phlmoea")

    ap.add_argument("--refhv", type=float, default=1.1)
    ap.add_argument("--pf_points", type=int, default=1500)
    ap.add_argument("--clip_plot", type=int, default=1)

    ap.add_argument("--sbx_eta", type=float, default=15.0)
    ap.add_argument("--mut_eta", type=float, default=20.0)
    ap.add_argument("--cx_prob", type=float, default=0.9)

    # offspring 两路比例调度（交叉->组合）
    ap.add_argument("--cross_ratio_start", type=float, default=0.70)
    ap.add_argument("--cross_ratio_end", type=float, default=0.30)
    ap.add_argument("--prefer_front_in_crossover", type=int, default=1)

    # 组合生成扰动调度（前大后小）
    ap.add_argument("--noise_start", type=float, default=0.08)
    ap.add_argument("--noise_end", type=float, default=0.02)

    # 去重
    ap.add_argument("--enable_dup", type=int, default=1)
    ap.add_argument("--cont_round", type=int, default=4)
    ap.add_argument("--max_trials_mult", type=int, default=50)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    clip_plot = bool(args.clip_plot)

    tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summary = []

    for bench in args.benches:
        print("\n" + "=" * 90)
        print(f"[RUN] {args.algo_name} bench={bench} pop={args.pop} gens={args.gens} seed={args.seed} "
              f"cross_ratio {args.cross_ratio_start}->{args.cross_ratio_end} "
              f"combo_noise {args.noise_start}->{args.noise_end} dup={bool(args.enable_dup)}")

        ref_pf = true_pf(bench, n=args.pf_points)

        t0 = time.time()
        X, F, curve, arch_nd = run_phlmoea(
            bench, args.pop, args.gens, args.seed,
            ref_pf=ref_pf, refhv=args.refhv,
            sbx_eta=args.sbx_eta, mut_eta=args.mut_eta, cx_prob=args.cx_prob,
            cross_ratio_start=args.cross_ratio_start, cross_ratio_end=args.cross_ratio_end,
            prefer_front_in_crossover=bool(args.prefer_front_in_crossover),
            noise_start=args.noise_start, noise_end=args.noise_end,
            enable_dup=bool(args.enable_dup), cont_round=args.cont_round,
            max_trials_mult=args.max_trials_mult,
            out_dir=args.out_dir,
        )
        t1 = time.time()

        ndF = nondominated(F)
        igd_final = igd(ndF, ref_pf)
        hv_final = hv_2d_min(ndF, (args.refhv, args.refhv))

        curve_csv = os.path.join(args.out_dir, f"{args.algo_name}_{bench}_{tag}_igd_curve.csv")
        curve_png = os.path.join(args.out_dir, f"{args.algo_name}_{bench}_{tag}_igd_curve.png")
        nd_csv = os.path.join(args.out_dir, f"{args.algo_name}_{bench}_{tag}_archive_nd.csv")
        pop_all_csv = os.path.join(args.out_dir, f"{args.algo_name}_{bench}_{tag}_pop_all.csv")
        ref_csv = os.path.join(args.out_dir, f"{args.algo_name}_{bench}_{tag}_ref_pf.csv")
        pareto_png = os.path.join(args.out_dir, f"{args.algo_name}_{bench}_{tag}_pareto.png")

        np.savetxt(pop_all_csv, F, delimiter=",", header="f1,f2", comments="")
        np.savetxt(nd_csv, arch_nd, delimiter=",", header="f1,f2", comments="")
        np.savetxt(ref_csv, ref_pf, delimiter=",", header="f1,f2", comments="")

        save_curve(curve, curve_csv, curve_png, title=f"IGD Curve ({bench})")
        save_front(F, ndF, ref_pf, pareto_png, title=f"Pareto Front ({bench})",
                   clip_plot=clip_plot, refhv=args.refhv)

        summ = {
            "algo": args.algo_name,
            "bench": bench,
            "pop": args.pop,
            "gens": args.gens,
            "seed": args.seed,
            "IGD_final": float(igd_final),
            "HV_final": float(hv_final),
            "ND_size": int(ndF.shape[0]),
            "Archive_ND_size": int(arch_nd.shape[0]),
            "time_sec": float(t1 - t0),
            "files": {
                "igd_curve_csv": os.path.basename(curve_csv),
                "igd_curve_png": os.path.basename(curve_png),
                "pop_all_csv": os.path.basename(pop_all_csv),
                "archive_nd_csv": os.path.basename(nd_csv),
                "ref_pf_csv": os.path.basename(ref_csv),
                "pareto_png": os.path.basename(pareto_png),
            }
        }
        all_summary.append(summ)
        print(f"[RESULT] {bench}: IGD={igd_final:.6f} HV={hv_final:.6f} ND={ndF.shape[0]} time={t1-t0:.2f}s")

    master = os.path.join(args.out_dir, f"{args.algo_name}_MASTER_{tag}.json")
    with open(master, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "-" * 90)
    print(f"[MASTER] saved: {master}")
    print("-" * 90)


if __name__ == "__main__":
    main()

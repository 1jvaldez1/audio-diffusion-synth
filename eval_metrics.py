"""
eval_metrics.py
===============
Distributional evaluation of generated audio against real audio.

Everything here works on one row per clip in a shared comparison space, so it
is independent of which diffusion path produced the sample. In the current
notebook that space is the flattened, normalized log-mel spectrogram that the
existing Fréchet cell already uses. A pretrained embedding can be swapped in
without changing anything below.

    1. Put real, generated and round-tripped clips in the same space.
    2. Call evaluate(...). It fits PCA on the real set only, then returns
         - a table of every metric against a real-vs-real floor, and
         - a stretch spectrum per candidate (how the geometry is distorted).

What is in the module
---------------------
    fit_reducer            centering (+ optional standardization) and PCA, real data only
    frechet_split          Gaussian W2^2 split into mean and covariance terms
    stretch_spectrum       exact per-direction decomposition of the covariance term
    mmd_permutation_test   unbiased RBF MMD^2 with a group-aware permutation test
    calibrated_report      every metric against a matched-size real-vs-real floor
    evaluate               the three above in one call
    roundtrip_mulaw        mu-law round trip for the waveform path's floor

Notes on comparability
----------------------
* Fréchet values computed after PCA are NOT comparable to published FAD
  numbers. They are comparable across your own paths, which is the point.
* The MMD here follows the same estimator family as KAD (unbiased, RBF kernel,
  median-heuristic bandwidth). For numbers you intend to compare with other
  papers, compute official KAD with the kadtk toolkit.

Dependencies: numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "Reducer", "fit_reducer",
    "gaussian_stats", "frechet_split",
    "StretchSpectrum", "stretch_spectrum",
    "median_bandwidth", "mmd2_unbiased", "mmd_permutation_test",
    "default_metrics", "calibrated_report", "format_report",
    "evaluate", "roundtrip_mulaw",
]


# --------------------------------------------------------------------------
# Linear algebra helpers
# --------------------------------------------------------------------------

def _sym(S: np.ndarray) -> np.ndarray:
    return (S + S.T) / 2.0


def _sym_power(S: np.ndarray, power: float, rel_floor: float = 1e-10) -> np.ndarray:
    """Matrix power of a symmetric PSD matrix through its eigendecomposition.

    Eigenvalues are floored at rel_floor * largest eigenvalue so that negative
    powers stay finite when the matrix is close to singular.
    """
    w, V = np.linalg.eigh(_sym(S))
    w = np.clip(w, rel_floor * max(w.max(), 1e-300), None)
    return (V * w ** power) @ V.T


# --------------------------------------------------------------------------
# Reducer: standardize + PCA, fit on real data only
# --------------------------------------------------------------------------

@dataclass
class Reducer:
    """Centering, optional standardization and PCA learned from the real set."""
    mean: np.ndarray          # (d,)   feature means of the real set
    scale: np.ndarray         # (d,)   feature scales (all ones unless standardized)
    components: np.ndarray    # (k, d) PCA directions in (scaled) feature space
    explained: np.ndarray     # (k,)   fraction of variance per kept component

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean) / self.scale) @ self.components.T

    def residual(self, X: np.ndarray) -> np.ndarray:
        """Squared distance of each clip from the PCA subspace, shape (n,)."""
        Z = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        return np.sum(Z ** 2, axis=1) - np.sum((Z @ self.components.T) ** 2, axis=1)

    def to_feature_space(self, directions: np.ndarray) -> np.ndarray:
        """Map directions from PCA space back to feature space.

        directions has shape (..., k); the result has shape (..., d). For
        spectrogram inputs, reshape the result to (n_bins, n_frames) to see
        which time-frequency region a stretch direction corresponds to.
        """
        return np.asarray(directions) @ self.components


def fit_reducer(
    real: np.ndarray,
    max_dim: Optional[int] = None,
    var_keep: float = 0.99,
    standardize: bool = False,
) -> Reducer:
    """Fit centering (+ optional standardization) and PCA on the real set.

    Why reduce at all: the covariance terms need Sigma_real to be invertible
    and well estimated. With N clips in d dimensions that fails unless N is
    comfortably larger than d, which is never true for raw spectrograms and
    rarely true for per-class audio embeddings.

    The kept dimension k is the smallest number of components explaining
    var_keep of the variance, capped at max_dim. If max_dim is not given it
    defaults to N // 8: calibrated_report compares half-size sets, and this
    leaves each of those about four samples per dimension.

    standardize=False (default) only centres the data, which matches
    sklearn's PCA and suits inputs already in common units, such as
    normalized spectrogram values. Use standardize=True for embeddings whose
    dimensions have unrelated scales.
    """
    X = np.asarray(real, dtype=np.float64)
    n, d = X.shape
    mean = X.mean(axis=0)
    if standardize:
        scale = X.std(axis=0, ddof=1)
        scale = np.where(scale < 1e-12, 1.0, scale)
    else:
        scale = np.ones(d)
    Z = (X - mean) / scale

    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    frac = sv ** 2 / np.sum(sv ** 2)
    k_var = int(np.searchsorted(np.cumsum(frac), var_keep) + 1)
    if max_dim is None:
        max_dim = max(2, n // 8)
    k = int(max(1, min(k_var, max_dim, len(sv) - 1)))
    return Reducer(mean=mean, scale=scale, components=Vt[:k], explained=frac[:k])


# --------------------------------------------------------------------------
# Gaussian W2 (Fréchet) split into mean and covariance terms
# --------------------------------------------------------------------------

def gaussian_stats(X: np.ndarray, shrinkage: float = 0.0):
    """Mean and covariance, with optional shrinkage toward a scaled identity.

    shrinkage = s replaces Sigma by (1 - s) * Sigma + s * (trace(Sigma)/d) * I.
    A small value (0.01 to 0.1) stabilizes the inverse when samples are scarce.
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    S = np.atleast_2d(np.cov(X, rowvar=False))
    if shrinkage > 0:
        d = S.shape[0]
        S = (1 - shrinkage) * S + shrinkage * (np.trace(S) / d) * np.eye(d)
    return mu, _sym(S)


def frechet_split(real: np.ndarray, gen: np.ndarray, shrinkage: float = 0.0) -> Dict[str, float]:
    """Squared Gaussian W2 distance, reported as its two additive parts.

        w2_sq   = mean_sq + cov_sq
        mean_sq = ||mu_r - mu_g||^2                      (location shift)
        cov_sq  = Tr(S_r + S_g - 2 (S_r^1/2 S_g S_r^1/2)^1/2)   (shape difference)

    This is the Fréchet distance; the split is what makes it interpretable.
    """
    mu_r, S_r = gaussian_stats(real, shrinkage)
    mu_g, S_g = gaussian_stats(gen, shrinkage)
    mean_sq = float(np.sum((mu_r - mu_g) ** 2))
    R = _sym_power(S_r, 0.5)
    M = _sym(R @ S_g @ R)
    w = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    cov_sq = float(np.trace(S_r) + np.trace(S_g) - 2.0 * np.sqrt(w).sum())
    cov_sq = max(cov_sq, 0.0)
    return {"mean_sq": mean_sq, "cov_sq": cov_sq, "w2_sq": mean_sq + cov_sq}


# --------------------------------------------------------------------------
# Stretch spectrum: where the covariance term comes from
# --------------------------------------------------------------------------

@dataclass
class StretchSpectrum:
    """Per-direction breakdown of the covariance term of Gaussian W2.

    Rows are sorted by contribution, largest first.

    stretch[i]       factor by which the generator scales direction i relative
                     to real data (as a standard-deviation ratio along the
                     transport direction). < 1 under-dispersed, > 1 over-dispersed.
    directions[i]    unit vector for direction i, in the reduced (PCA) space.
    contribution[i]  that direction's share of cov_sq, in the same units.
                     contribution.sum() equals cov_sq up to numerical error.
    """
    stretch: np.ndarray
    directions: np.ndarray
    contribution: np.ndarray
    cov_sq: float
    mean_sq: float
    contraction_share: float     # fraction of cov_sq from directions with stretch < 1
    expansion_share: float       # fraction of cov_sq from directions with stretch > 1
    log_volume_ratio: float      # sum(log stretch): < 0 means the generated cloud is smaller
    mean_shift_along: np.ndarray  # mean shift projected on each direction, in real-data std units

    def top(self, k: int = 5) -> List[Dict[str, float]]:
        total = self.cov_sq if self.cov_sq > 0 else 1.0
        return [
            {
                "rank": i + 1,
                "stretch": float(self.stretch[i]),
                "share_of_cov_sq": float(self.contribution[i] / total),
                "mean_shift_in_real_std": float(self.mean_shift_along[i]),
            }
            for i in range(min(k, len(self.stretch)))
        ]


def stretch_spectrum(real: np.ndarray, gen: np.ndarray, shrinkage: float = 0.0) -> StretchSpectrum:
    """Decompose the covariance term of Gaussian W2 direction by direction.

    For Gaussians, the optimal transport map from real to generated is affine:

        T(z) = mu_g + A (z - mu_r),
        A    = S_r^-1/2 (S_r^1/2 S_g S_r^1/2)^1/2 S_r^-1/2.

    A is symmetric positive definite, so it is a pure stretch: it has
    orthogonal eigenvectors v_i with eigenvalues lambda_i > 0 and no rotation
    part. The covariance term of W2^2 is then exactly

        cov_sq = Tr((A - I) S_r (A - I)) = sum_i (lambda_i - 1)^2 * v_i' S_r v_i.

    Each summand is the cost of moving real data to generated data along v_i.
    lambda_i < 1 means the generator produces too little variation along v_i
    (collapse); lambda_i > 1 means too much (noise, artifacts).

    Inputs should already be reduced with a Reducer so that S_r is invertible.
    """
    mu_r, S_r = gaussian_stats(real, shrinkage)
    mu_g, S_g = gaussian_stats(gen, shrinkage)

    R = _sym_power(S_r, 0.5)
    R_inv = _sym_power(S_r, -0.5)
    M_half = _sym_power(R @ S_g @ R, 0.5, rel_floor=0.0)
    A = _sym(R_inv @ M_half @ R_inv)

    lam, V = np.linalg.eigh(A)
    lam = np.clip(lam, 1e-12, None)
    var_along = np.einsum("ji,jk,ki->i", V, S_r, V)        # v_i' S_r v_i
    contrib = (lam - 1.0) ** 2 * var_along

    order = np.argsort(-contrib)
    lam, V, contrib, var_along = lam[order], V[:, order], contrib[order], var_along[order]

    cov_sq = float(contrib.sum())
    total = cov_sq if cov_sq > 0 else 1.0
    shift = (V.T @ (mu_g - mu_r)) / np.sqrt(np.clip(var_along, 1e-300, None))

    return StretchSpectrum(
        stretch=lam,
        directions=V.T,
        contribution=contrib,
        cov_sq=cov_sq,
        mean_sq=float(np.sum((mu_r - mu_g) ** 2)),
        contraction_share=float(contrib[lam < 1].sum() / total),
        expansion_share=float(contrib[lam > 1].sum() / total),
        log_volume_ratio=float(np.log(lam).sum()),
        mean_shift_along=shift,
    )


# --------------------------------------------------------------------------
# Kernel MMD with a group-aware permutation test
# --------------------------------------------------------------------------

def _sq_dists(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    Y = X if Y is None else Y
    d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.clip(d2, 0.0, None)


def median_bandwidth(X: np.ndarray, max_points: int = 2000, seed: int = 0) -> float:
    """Median-heuristic RBF bandwidth (sigma) from pairwise distances in X.

    Compute this once on the real data and reuse it for every comparison, so
    that all MMD values share one kernel and can be compared with each other.
    """
    X = np.asarray(X, dtype=np.float64)
    if len(X) > max_points:
        X = X[np.random.default_rng(seed).choice(len(X), max_points, replace=False)]
    d2 = _sq_dists(X)
    iu = np.triu_indices(len(X), k=1)
    med = np.median(d2[iu])
    return float(np.sqrt(med)) if med > 0 else 1.0


def _mmd2_from_K(K: np.ndarray, s: np.ndarray) -> float:
    """Unbiased MMD^2 from a pooled kernel matrix and a 0/1 membership vector."""
    t = 1.0 - s
    m, n = s.sum(), t.sum()
    Ks, Kt = K @ s, K @ t
    diag = np.diag(K)
    aa = (s @ Ks - s @ diag) / (m * (m - 1))
    bb = (t @ Kt - t @ diag) / (n * (n - 1))
    ab = (s @ Kt) / (m * n)
    return float(aa + bb - 2.0 * ab)


def mmd2_unbiased(real: np.ndarray, gen: np.ndarray, bandwidth: Optional[float] = None) -> float:
    """Unbiased estimate of squared MMD with a Gaussian RBF kernel.

    The estimate is centred on zero when the two distributions are equal, so
    small negative values are normal and mean "no detectable difference".
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    if bandwidth is None:
        bandwidth = median_bandwidth(real)
    Z = np.vstack([real, gen])
    K = np.exp(-_sq_dists(Z) / (2.0 * bandwidth ** 2))
    s = np.zeros(len(Z))
    s[: len(real)] = 1.0
    return _mmd2_from_K(K, s)


def mmd_permutation_test(
    real: np.ndarray,
    gen: np.ndarray,
    groups_real: Optional[Sequence] = None,
    groups_gen: Optional[Sequence] = None,
    n_perm: int = 1000,
    bandwidth: Optional[float] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Permutation test of H0: real and generated embeddings share a distribution.

    groups_real / groups_gen give one group id per clip (recording, individual,
    deployment). Labels are permuted at the group level, so clips cut from the
    same recording always move together. Treating them as independent makes
    the test overconfident. If a groups argument is None, each clip in that set
    is its own group, which is the right default for generated samples.

    The p-value is evidence about whether the sets differ. It is not a quality
    score: with enough samples any generator is detectably different.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    n_r, n_g = len(real), len(gen)
    if bandwidth is None:
        bandwidth = median_bandwidth(real)

    Z = np.vstack([real, gen])
    K = np.exp(-_sq_dists(Z) / (2.0 * bandwidth ** 2))

    g_r = np.arange(n_r) if groups_real is None else np.unique(np.asarray(groups_real), return_inverse=True)[1]
    n_gr = int(g_r.max()) + 1
    g_g = np.arange(n_g) if groups_gen is None else np.unique(np.asarray(groups_gen), return_inverse=True)[1]
    groups = np.concatenate([g_r, g_g + n_gr])
    n_groups = int(groups.max()) + 1

    s_obs = np.zeros(len(Z))
    s_obs[:n_r] = 1.0
    observed = _mmd2_from_K(K, s_obs)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        pick = np.zeros(n_groups, dtype=bool)
        pick[rng.choice(n_groups, n_gr, replace=False)] = True
        s = pick[groups].astype(np.float64)
        # A permutation needs at least two clips on each side.
        if s.sum() < 2 or (len(s) - s.sum()) < 2:
            null[b] = np.nan
            continue
        null[b] = _mmd2_from_K(K, s)
    null = null[~np.isnan(null)]
    p = (1.0 + np.sum(null >= observed)) / (1.0 + len(null))
    return {
        "mmd2": observed,
        "p_value": float(p),
        "null_95th": float(np.percentile(null, 95)),
        "bandwidth": float(bandwidth),
        "n_perm": int(len(null)),
    }


# --------------------------------------------------------------------------
# Calibrated report: every metric against a matched real-vs-real floor
# --------------------------------------------------------------------------

def default_metrics(a: np.ndarray, b: np.ndarray, bandwidth: Optional[float] = None,
                    shrinkage: float = 0.0) -> Dict[str, float]:
    out = frechet_split(a, b, shrinkage=shrinkage)
    out["mmd2"] = mmd2_unbiased(a, b, bandwidth=bandwidth)
    return out


def calibrated_report(
    real: np.ndarray,
    candidates: Dict[str, np.ndarray],
    groups_real: Optional[Sequence] = None,
    aligned: Iterable[str] = (),
    n_splits: int = 30,
    shrinkage: float = 0.0,
    metrics_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compare each candidate set to real data, relative to a real-vs-real floor.

    On each of n_splits rounds the real data is split in half by group, into
    A (reference) and B (held out). Then:

        floor      = metric(A, B)                  real vs. real
        candidate  = metric(A, C)                  real vs. candidate

    where C has the same number of clips as B. Matching the sizes matters:
    Fréchet-type distances are biased upward at small sample sizes, so a
    full-size comparison set against a half-size floor would not be fair.

    candidates maps a name to an embedding array, for example
        {"path_A": ..., "path_B": ..., "roundtrip_mel": ..., "roundtrip_mulaw": ...}

    aligned lists the candidates whose rows correspond one-to-one with the rows
    of real, which is the case for round trips of the real clips. For those, C
    is taken from the B rows, so the round-tripped clips are never copies of
    clips in the reference half. For all other candidates C is a random
    subsample.

    For each candidate and metric the result holds
        value            median over splits
        floor            median real-vs-real value over splits
        ratio            value / floor. nan when the floor is not reliably
                         positive across splits, which is expected for unbiased
                         MMD^2 (it is centred on zero); read frac_above instead
        frac_above       fraction of splits where the candidate exceeded the
                         floor from the same split
        lo, hi           5th and 95th percentile of the value over splits

    The lo/hi range describes split-to-split variability. It is not a formal
    confidence interval.
    """
    real = np.asarray(real, dtype=np.float64)
    n = len(real)
    aligned = set(aligned)
    for name in aligned:
        if len(candidates[name]) != n:
            raise ValueError(f"aligned candidate '{name}' must have one row per real clip")

    bandwidth = median_bandwidth(real)
    if metrics_fn is None:
        metrics_fn = lambda a, b: default_metrics(a, b, bandwidth=bandwidth, shrinkage=shrinkage)

    g = np.arange(n) if groups_real is None else np.unique(np.asarray(groups_real), return_inverse=True)[1]
    n_groups = int(g.max()) + 1
    if n_groups < 4:
        raise ValueError("need at least 4 groups of real clips to form a real-vs-real split")

    rng = np.random.default_rng(seed)
    floor_runs: List[Dict[str, float]] = []
    cand_runs: Dict[str, List[Dict[str, float]]] = {k: [] for k in candidates}

    for _ in range(n_splits):
        in_a = np.zeros(n_groups, dtype=bool)
        in_a[rng.permutation(n_groups)[: n_groups // 2]] = True
        idx_a = np.flatnonzero(in_a[g])
        idx_b = np.flatnonzero(~in_a[g])
        A, B = real[idx_a], real[idx_b]
        floor_runs.append(metrics_fn(A, B))
        for name, C in candidates.items():
            C = np.asarray(C, dtype=np.float64)
            if name in aligned:
                sub = C[idx_b]
            else:
                m = min(len(idx_b), len(C))
                sub = C[rng.choice(len(C), m, replace=False)]
            cand_runs[name].append(metrics_fn(A, sub))

    keys = list(floor_runs[0].keys())
    floor = {k: np.array([r[k] for r in floor_runs]) for k in keys}
    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, runs in cand_runs.items():
        report[name] = {}
        for k in keys:
            v = np.array([r[k] for r in runs])
            f_med = float(np.median(floor[k]))
            report[name][k] = {
                "value": float(np.median(v)),
                "floor": f_med,
                "ratio": float(np.median(v) / f_med) if np.percentile(floor[k], 5) > 0 else float("nan"),
                "frac_above": float(np.mean(v > floor[k])),
                "lo": float(np.percentile(v, 5)),
                "hi": float(np.percentile(v, 95)),
            }
    return report


def format_report(report: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Plain-text table of a calibrated_report result."""
    lines = [f"{'candidate':<18}{'metric':<10}{'value':>11}{'floor':>11}{'ratio':>9}{'above':>8}{'5%':>11}{'95%':>11}"]
    for name, metrics in report.items():
        for k, r in metrics.items():
            ratio = f"{r['ratio']:.2f}" if np.isfinite(r["ratio"]) else "n/a"
            lines.append(
                f"{name:<18}{k:<10}{r['value']:>11.4g}{r['floor']:>11.4g}{ratio:>9}"
                f"{r['frac_above']:>8.2f}{r['lo']:>11.4g}{r['hi']:>11.4g}"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# One-call evaluation
# --------------------------------------------------------------------------

def evaluate(
    real: np.ndarray,
    candidates: Dict[str, np.ndarray],
    groups_real: Optional[Sequence] = None,
    aligned: Iterable[str] = (),
    max_dim: Optional[int] = 20,
    standardize: bool = False,
    n_splits: int = 30,
    n_perm: int = 1000,
    shrinkage: float = 0.0,
    seed: int = 0,
) -> Dict[str, object]:
    """Reduce, then run the calibrated report, stretch spectra and MMD tests.

    real and every candidate are (n_clips, d) arrays in the same space.
    See calibrated_report for the meaning of `aligned` and `groups_real`.

    Returns a dict with
        reducer       the fitted Reducer
        report        calibrated_report output (print with format_report)
        spectra       {candidate: StretchSpectrum}, each against the full real set
        mmd           {candidate: mmd_permutation_test output}
        off_subspace  {candidate: ratio}. Every metric above only sees the PCA
                      subspace of the real data. This is the candidate's mean
                      squared distance from that subspace divided by the same
                      quantity for held-out real clips (PCA fit on one half of
                      the real data, measured on the other). About 1 is
                      normal; well above 1 means the candidate contains
                      structure the other metrics cannot see. Trivially 1 when
                      the input has no more dimensions than max_dim.
    """
    real = np.asarray(real, dtype=np.float64)
    reducer = fit_reducer(real, max_dim=max_dim, standardize=standardize)

    g = np.arange(len(real)) if groups_real is None else np.unique(np.asarray(groups_real), return_inverse=True)[1]
    in_a = np.zeros(int(g.max()) + 1, dtype=bool)
    in_a[np.random.default_rng(seed).permutation(len(in_a))[: len(in_a) // 2]] = True
    half = fit_reducer(real[in_a[g]], max_dim=reducer.components.shape[0], var_keep=1.0, standardize=standardize)
    base = float(np.mean(half.residual(real[~in_a[g]])))
    off = {k: (float(np.mean(half.residual(v)) / base) if base > 1e-12 else 1.0) for k, v in candidates.items()}

    Zr = reducer.transform(real)
    Zc = {k: reducer.transform(v) for k, v in candidates.items()}
    bw = median_bandwidth(Zr)
    return {
        "reducer": reducer,
        "off_subspace": off,
        "report": calibrated_report(Zr, Zc, groups_real=groups_real, aligned=aligned,
                                    n_splits=n_splits, shrinkage=shrinkage, seed=seed),
        "spectra": {k: stretch_spectrum(Zr, v, shrinkage=shrinkage) for k, v in Zc.items()},
        "mmd": {k: mmd_permutation_test(Zr, v, groups_real=groups_real, n_perm=n_perm,
                                        bandwidth=bw, seed=seed) for k, v in Zc.items()},
    }


# --------------------------------------------------------------------------
# Representation round trip for the waveform path
# --------------------------------------------------------------------------
# A round trip pushes a REAL clip through a path's representation and back,
# with no diffusion model involved. Comparing round-tripped real clips with
# untouched real clips measures what the representation alone costs; the rest
# of a path's distance is what its generator adds.
#
# The spectrogram paths need no helper here: use the pipeline's own transforms
# (for the linear-STFT path, the notebook's real_linear_as_mel is already this
# round trip). Only mu-law needs code.

def roundtrip_mulaw(y: np.ndarray, mu: int = 255, bits: Optional[int] = 8) -> np.ndarray:
    """Waveform -> mu-law companding (+ quantization) -> waveform.

    bits=8 quantizes to 256 levels. bits=None applies companding only, which
    is exactly invertible: if the model trains on continuous companded values,
    the round trip is the identity and this path's floor is zero by design.
    Expects y in [-1, 1]; values outside are clipped. Works on any array shape.
    """
    y = np.clip(np.asarray(y, dtype=np.float64), -1.0, 1.0)
    comp = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    if bits is not None:
        levels = 2 ** bits
        comp = np.round((comp + 1.0) / 2.0 * (levels - 1)) / (levels - 1) * 2.0 - 1.0
    return np.sign(comp) * np.expm1(np.abs(comp) * np.log1p(mu)) / mu


# --------------------------------------------------------------------------
# Self-test on data with a known answer:  python eval_metrics.py
# --------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.default_rng(0)
    n, d = 1200, 12
    L = rng.normal(size=(d, d))
    real = rng.normal(size=(n, d)) @ L.T

    # A generator that halves the variation along one known direction and
    # adds a small mean shift, built by applying a symmetric stretch.
    S_r = np.cov(real, rowvar=False)
    u = np.linalg.eigh(S_r)[1][:, -1]                  # dominant real direction
    A_true = np.eye(d) - 0.5 * np.outer(u, u)          # stretch 0.5 along u
    fresh = rng.normal(size=(n, d)) @ L.T
    gen = (fresh - fresh.mean(0)) @ A_true.T + fresh.mean(0) + 0.3 * u

    fs = frechet_split(real, gen)
    ss = stretch_spectrum(real, gen)
    assert abs(ss.contribution.sum() - fs["cov_sq"]) < 1e-8 * max(1.0, fs["cov_sq"]), "decomposition is not exact"
    assert abs(ss.stretch[0] - 0.5) < 0.05, f"expected stretch 0.5, got {ss.stretch[0]:.3f}"
    assert abs(abs(ss.directions[0] @ u) - 1.0) < 0.02, "did not recover the collapsed direction"
    assert ss.contraction_share > 0.9

    same = mmd_permutation_test(real[:300], fresh[:300], n_perm=300)
    diff = mmd_permutation_test(real[:300], gen[:300], n_perm=300)
    assert same["p_value"] > 0.05 and diff["p_value"] < 0.01

    rep = calibrated_report(real, {"generated": gen, "fresh_real": fresh}, n_splits=10)
    assert rep["generated"]["w2_sq"]["ratio"] > 3.0
    assert 0.5 < rep["fresh_real"]["w2_sq"]["ratio"] < 2.0

    x = np.linspace(-1, 1, 1001)
    assert np.max(np.abs(roundtrip_mulaw(x) - x)) < 0.03
    assert np.max(np.abs(roundtrip_mulaw(x, bits=None) - x)) < 1e-12

    # Spectrogram-shaped input (more features than clips) through evaluate().
    hi = rng.normal(size=(300, 12)) @ rng.normal(size=(12, 4096)) + 0.05 * rng.normal(size=(300, 4096))
    out = evaluate(hi, {"copy_plus_noise": hi + 0.01 * rng.normal(size=hi.shape)},
                   aligned=["copy_plus_noise"], max_dim=10, n_splits=5, n_perm=100)
    assert out["reducer"].to_feature_space(out["spectra"]["copy_plus_noise"].directions[0]).shape == (4096,)
    assert 0.5 < out["report"]["copy_plus_noise"]["w2_sq"]["ratio"] < 2.0
    # With all 12 real directions kept, what is left of real clips is noise,
    # so a constant offset outside that subspace must stand out.
    off = evaluate(hi, {"offset": hi + 0.5, "same": hi + 0.0}, max_dim=12, n_splits=5, n_perm=50)["off_subspace"]
    assert off["offset"] > 5.0, "a constant offset outside the real subspace should be flagged"
    assert 0.5 < off["same"] < 1.5

    print("frechet split :", {k: round(v, 4) for k, v in fs.items()})
    print("top directions:", ss.top(3))
    print("mmd same/diff p:", same["p_value"], diff["p_value"])
    print(format_report(rep))
    print("self-test passed")


if __name__ == "__main__":
    _self_test()

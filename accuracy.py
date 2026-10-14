"""Thematic-map accuracy assessment (Congalton 1991 conventions). NumPy only.

Confusion matrix layout: ROWS = reference (truth), COLUMNS = map (predicted).
    Producer's accuracy (PA) = diagonal / row total     (omission error = 1 - PA)
    User's accuracy     (UA) = diagonal / column total  (commission error = 1 - UA)
"""
from __future__ import annotations

import csv
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def confusion_matrix(y_ref, y_pred, labels: Optional[Sequence] = None) -> Tuple[np.ndarray, List]:
    y_ref = np.asarray(y_ref).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_ref.shape != y_pred.shape:
        raise ValueError("Reference and predicted arrays must have the same length.")
    if labels is None:
        labels = sorted(set(np.unique(y_ref)) | set(np.unique(y_pred)))
    labels = list(labels)
    index = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for r, p in zip(y_ref, y_pred):
        if r in index and p in index:
            cm[index[r], index[p]] += 1
    return cm, labels


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b != 0)
    return out


def kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe != 1 else float("nan")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95 % Wilson score interval for the overall accuracy (treats samples as independent)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def summarize(cm: np.ndarray) -> Dict:
    cm = np.asarray(cm)
    n = int(cm.sum())
    diag = np.diag(cm)
    pa = _safe_div(diag, cm.sum(axis=1))
    ua = _safe_div(diag, cm.sum(axis=0))
    f1 = _safe_div(2 * pa * ua, pa + ua)
    lo, hi = wilson_interval(int(diag.sum()), n)
    return {
        "n": n,
        "overall_accuracy": float(diag.sum() / n) if n else float("nan"),
        "oa_ci95": (lo, hi),
        "kappa": kappa(cm),
        "producers_accuracy": pa,
        "users_accuracy": ua,
        "f1": f1,
        "reference_totals": cm.sum(axis=1),
        "map_totals": cm.sum(axis=0),
    }


def format_report(cm: np.ndarray, labels: Sequence, names: Optional[Dict] = None) -> str:
    s = summarize(cm)
    names = names or {}
    nm = [str(names.get(l, l)) for l in labels]
    w = max(8, max(len(x) for x in nm) + 1)
    corner = "reference \\ map"          # defined outside the f-string: backslashes are not allowed inside {...} before Python 3.12
    lines = [f"{corner:<{w}}" + "".join(f"{x:>{w}}" for x in nm) + f"{'Total':>{w}}{'PA %':>{w}}"]
    for i, x in enumerate(nm):
        pa = s["producers_accuracy"][i] * 100
        lines.append(f"{x:<{w}}" + "".join(f"{int(v):>{w}}" for v in cm[i]) +
                     f"{int(cm[i].sum()):>{w}}{pa:>{w}.1f}")
    lines.append(f"{'Total':<{w}}" + "".join(f"{int(v):>{w}}" for v in cm.sum(axis=0)) + f"{s['n']:>{w}}")
    lines.append(f"{'UA %':<{w}}" + "".join(f"{v * 100:>{w}.1f}" for v in s["users_accuracy"]))
    lo, hi = s["oa_ci95"]
    lines.append("")
    lines.append(f"Overall accuracy: {s['overall_accuracy'] * 100:.1f} %  (95 % CI {lo * 100:.1f}-{hi * 100:.1f} %, n={s['n']})")
    lines.append(f"Cohen's kappa:    {s['kappa']:.3f}")
    return "\n".join(lines)


def save_csv(cm: np.ndarray, labels: Sequence, path: str, names: Optional[Dict] = None) -> None:
    s = summarize(cm)
    names = names or {}
    nm = [str(names.get(l, l)) for l in labels]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reference \\ map"] + nm + ["total", "producers_accuracy"])
        for i, x in enumerate(nm):
            w.writerow([x] + [int(v) for v in cm[i]] + [int(cm[i].sum()), f"{s['producers_accuracy'][i]:.4f}"])
        w.writerow(["total"] + [int(v) for v in cm.sum(axis=0)] + [s["n"], ""])
        w.writerow(["users_accuracy"] + [f"{v:.4f}" for v in s["users_accuracy"]] + ["", ""])
        w.writerow([])
        w.writerow(["overall_accuracy", f"{s['overall_accuracy']:.4f}"])
        w.writerow(["kappa", f"{s['kappa']:.4f}"])


# ------------------------------------------------------------------ spatially blocked validation
def block_ids(rows: np.ndarray, cols: np.ndarray, block_px: int) -> np.ndarray:
    """Integer id of the block_px x block_px tile that each (row, col) pixel belongs to."""
    if block_px < 1:
        raise ValueError("block_px must be >= 1")
    br = np.asarray(rows) // block_px
    bc = np.asarray(cols) // block_px
    return (br.astype(np.int64) << 32) | bc.astype(np.int64)


def blocked_split(rows, cols, y, block_px: int = 100, test_fraction: float = 0.3, seed: int = 42):
    """Hold out whole blocks so that train and test pixels are never neighbours.

    A random per-pixel split lets neighbouring, near-identical pixels land on both sides and
    inflates accuracy; splitting by block avoids that. Returns (train_idx, test_idx).
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ids = block_ids(rows, cols, block_px)
    y = np.asarray(y)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(len(uniq) * test_fraction)))
    if len(uniq) < 2:
        raise ValueError("Samples fall in a single block; use a smaller block size or an independent validation set.")
    test_blocks = set(rng.choice(uniq, size=min(n_test, len(uniq) - 1), replace=False).tolist())
    is_test = np.array([i in test_blocks for i in ids])
    return np.where(~is_test)[0], np.where(is_test)[0]


# ------------------------------------------------------------------ binary masks / segmentation
def binary_metrics(pred: np.ndarray, ref: np.ndarray, valid: Optional[np.ndarray] = None) -> Dict:
    """Pixel-wise metrics of a binary mask against a binary reference (e.g. water, buildings).

    Reports IoU (Jaccard), Dice/F1, precision (= user's accuracy), recall (= producer's accuracy),
    overall agreement and kappa, all restricted to `valid` pixels when given.
    """
    p = np.asarray(pred).astype(bool); r = np.asarray(ref).astype(bool)
    if p.shape != r.shape:
        raise ValueError("Prediction and reference must have the same shape.")
    if valid is not None:
        v = np.asarray(valid).astype(bool); p, r = p[v], r[v]
    tp = int((p & r).sum()); fp = int((p & ~r).sum()); fn = int((~p & r).sum()); tn = int((~p & ~r).sum())
    cm = np.array([[tn, fp], [fn, tp]])          # rows = reference (0,1), cols = map (0,1)
    def div(a, b): return float(a / b) if b else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn,
            "iou": div(tp, tp + fp + fn), "dice": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "overall_agreement": div(tp + tn, tp + fp + fn + tn), "kappa": kappa(cm)}


def instance_metrics(pred_labels: np.ndarray, ref_labels: np.ndarray, iou_threshold: float = 0.5) -> Dict:
    """Instance-level precision/recall: a predicted object is a true positive if it overlaps a
    reference object with IoU >= threshold (one-to-one greedy matching by descending IoU)."""
    pl, rl = np.asarray(pred_labels), np.asarray(ref_labels)
    if pl.shape != rl.shape:
        raise ValueError("Label images must have the same shape.")
    pids = [i for i in np.unique(pl) if i != 0]; rids = [i for i in np.unique(rl) if i != 0]
    pairs = []
    for i in pids:
        pm = pl == i
        for j in np.unique(rl[pm]):
            if j == 0:
                continue
            rm = rl == j
            inter = int((pm & rm).sum()); union = int((pm | rm).sum())
            pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)
    used_p, used_r, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < iou_threshold:
            break
        if i not in used_p and j not in used_r:
            used_p.add(i); used_r.add(j); tp += 1
    fp, fn = len(pids) - tp, len(rids) - tp
    return {"predicted": len(pids), "reference": len(rids), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / len(pids) if pids else float("nan"),
            "recall": tp / len(rids) if rids else float("nan"), "iou_threshold": iou_threshold}


# ------------------------------------------------------------------ stratified, area-adjusted estimators (Olofsson et al. 2013, 2014)
def olofsson(cm_map_ref: np.ndarray, map_counts: Sequence[float], pixel_area: float = 1.0, z: float = 1.96) -> Dict:
    """Accuracy and area estimates from a stratified random sample (strata = MAP classes).

    cm_map_ref : sample counts, ROWS = map class (stratum), COLUMNS = reference class (same class order)
    map_counts : number of map pixels in each class (stratum sizes); pixel_area converts pixels to area units

    Returns overall/user's/producer's accuracy and the reference-based area of each class, each with a standard
    error and a z-based confidence interval. Unlike a plain confusion matrix, small classes are not under-weighted
    by the sampling design. Requires at least 2 samples per stratum for the variance terms.
    """
    n_ij = np.asarray(cm_map_ref, dtype=float)
    k = n_ij.shape[0]
    if n_ij.shape != (k, k):
        raise ValueError("The confusion matrix must be square (same classes for map and reference).")
    counts = np.asarray(map_counts, dtype=float)
    if counts.shape != (k,) or (counts <= 0).any():
        raise ValueError("map_counts needs one positive value per class.")
    n_i = n_ij.sum(axis=1)
    if (n_i < 2).any():
        raise ValueError("Every map class needs at least 2 reference samples.")
    W = counts / counts.sum()
    frac = n_ij / n_i[:, None]                       # n_ij / n_i
    p = W[:, None] * frac                            # estimated area proportions p_ij
    p_col = p.sum(axis=0)                            # reference-based class proportions p_.j
    diag = np.diag(frac)                             # user's accuracy U_i = n_ii / n_i
    oa = float(np.sum(W * diag))
    var_oa = float(np.sum(W ** 2 * diag * (1 - diag) / (n_i - 1)))
    ua = diag.copy()
    var_ua = ua * (1 - ua) / (n_i - 1)
    pa = np.where(p_col > 0, np.diag(p) / np.where(p_col > 0, p_col, 1), np.nan)
    var_pa = np.full(k, np.nan)
    v_cell = (frac * (1 - frac)) / (n_i[:, None] - 1)            # per-cell variance of n_ij / n_i
    for j in range(k):
        if p_col[j] <= 0:
            continue
        others = sum(W[i] ** 2 * v_cell[i, j] for i in range(k) if i != j)
        var_pa[j] = (W[j] ** 2 * (1 - pa[j]) ** 2 * ua[j] * (1 - ua[j]) / (n_i[j] - 1) + pa[j] ** 2 * others) / p_col[j] ** 2
    total_area = counts.sum() * pixel_area
    area = p_col * total_area
    se_area = np.sqrt(np.array([np.sum(W ** 2 * v_cell[:, j]) for j in range(k)])) * total_area
    se = lambda v: np.sqrt(v)
    return {
        "overall_accuracy": oa, "oa_se": float(se(var_oa)), "oa_ci95": (oa - z * se(var_oa), oa + z * se(var_oa)),
        "users_accuracy": ua, "users_se": se(var_ua),
        "producers_accuracy": pa, "producers_se": se(var_pa),
        "area_map": counts * pixel_area, "area_estimated": area, "area_se": se_area,
        "area_ci95": np.stack([area - z * se_area, area + z * se_area], axis=1),
        "samples_per_class": n_i, "weights": W, "z": z,
    }


def olofsson_grouped(n_hj: np.ndarray, strata_counts: Sequence[float], strata_group: Sequence[int], pixel_area: float = 1.0,
                     z: float = 1.96) -> Dict:
    """Olofsson-type estimators when several sampling strata are MERGED into the map categories that are reported.

    The sample was drawn per original map class (stratum h); the categories reported (e.g. 'change' = classes 2 and 3) are unions of
    strata. Each stratum keeps its own weight W_h and sample size n_h, so unequal allocation inside a merged category does not bias
    the result (pooling the samples of merged classes as if they were one simple random sample does).

    n_hj          : (H, K) sample counts, rows = strata (map classes), columns = reference categories
    strata_counts : H map pixel counts (stratum sizes)
    strata_group  : H indices 0..K-1, the map category of every stratum (category k uses the same name/order as reference k)
    """
    n = np.asarray(n_hj, dtype=float)
    H, K = n.shape
    counts = np.asarray(strata_counts, dtype=float)
    grp = np.asarray(strata_group, dtype=int)
    if counts.shape != (H,) or (counts <= 0).any() or grp.shape != (H,) or grp.min() < 0 or grp.max() >= K:
        raise ValueError("Invalid strata definition.")
    n_h = n.sum(axis=1)
    if (n_h < 2).any():
        raise ValueError("Every map class needs at least 2 reference samples (after excluding 'uncertain').")
    W = counts / counts.sum()
    frac = n / n_h[:, None]
    var_cell = n * (n_h[:, None] - n) / (n_h[:, None] * (n_h[:, None] - 1))     # s^2 of the indicator ref == j in stratum h
    inside = (grp[:, None] == np.arange(K)[None, :])                             # stratum h belongs to map category j
    # overall accuracy: indicator ref == map category of the stratum
    p_correct = frac[np.arange(H), grp]
    oa = float(np.sum(W * p_correct))
    var_oa = float(np.sum(W ** 2 * var_cell[np.arange(H), grp] / n_h))
    # reference-based category proportions and their variance
    p_col = (W[:, None] * frac).sum(axis=0)
    var_col = (W[:, None] ** 2 * var_cell / n_h[:, None]).sum(axis=0)
    # user's accuracy of map category k = share of its area whose reference is k
    Wk = np.array([W[inside[:, k]].sum() for k in range(K)])
    ua = np.full(K, np.nan); var_ua = np.full(K, np.nan)
    for k in range(K):
        if Wk[k] <= 0:
            continue
        m = inside[:, k]
        ua[k] = float(np.sum(W[m] * frac[m, k]) / Wk[k])
        var_ua[k] = float(np.sum(W[m] ** 2 * var_cell[m, k] / n_h[m]) / Wk[k] ** 2)
    # producer's accuracy of category j = share of the reference-j area that the map labels j (ratio estimator)
    pa = np.full(K, np.nan); var_pa = np.full(K, np.nan)
    for j in range(K):
        if p_col[j] <= 0:
            continue
        m = inside[:, j]
        Y = float(np.sum(W[m] * frac[m, j]))
        R = Y / p_col[j]
        terms = W ** 2 / n_h * (np.where(m, var_cell[:, j], 0.0) + R ** 2 * var_cell[:, j] - 2 * R * np.where(m, var_cell[:, j], 0.0))
        pa[j] = R
        var_pa[j] = float(terms.sum() / p_col[j] ** 2)
    total_area = counts.sum() * pixel_area
    area = p_col * total_area
    se_area = np.sqrt(var_col) * total_area
    area_map = np.array([counts[inside[:, k]].sum() for k in range(K)]) * pixel_area
    se = np.sqrt
    return {
        "overall_accuracy": oa, "oa_se": float(se(var_oa)), "oa_ci95": (oa - z * se(var_oa), oa + z * se(var_oa)),
        "users_accuracy": ua, "users_se": se(var_ua),
        "producers_accuracy": pa, "producers_se": se(var_pa),
        "area_map": area_map, "area_estimated": area, "area_se": se_area,
        "area_ci95": np.stack([area - z * se_area, area + z * se_area], axis=1),
        "samples_per_class": n_h, "weights": W, "z": z,
    }


def grouped_metrics(n_hj: np.ndarray, strata_counts: Sequence[float], strata_group: Sequence[int],
                    n_boot: int = 1000, seed: int = 1, z: float = 1.96) -> Dict:
    """Area-adjusted F1, IoU, kappa and quantity/allocation disagreement, with bootstrap 95 % intervals.

    Same inputs as olofsson_grouped. The population proportion matrix P (rows = map category, columns = reference category)
    is  P[k, j] = sum over strata h of map category k of  W_h * n_hj / n_h ; every measure is computed from P, so unequal
    sampling intensity never biases it. Per category j:  precision = UA = P_jj / P_j.   recall = PA = P_jj / P_.j
    F1 = 2 P_jj / (P_j. + P_.j) (Dice)   IoU = P_jj / (P_j. + P_.j - P_jj) (Jaccard).  For a two-category map (e.g. water / land)
    read the row of the category of interest. Intervals: points are resampled with replacement inside every stratum."""
    n = np.asarray(n_hj, dtype=float)
    H, K = n.shape
    counts = np.asarray(strata_counts, dtype=float)
    grp = np.asarray(strata_group, dtype=int)
    W = counts / counts.sum()

    def proportions(nn):
        nh = nn.sum(axis=1, keepdims=True)
        frac = np.divide(nn, nh, out=np.zeros_like(nn), where=nh > 0)
        P = np.zeros((K, K))
        for h in range(H):
            P[grp[h]] += W[h] * frac[h]
        return P

    def measures(P):
        tp = np.diag(P); row = P.sum(axis=1); col = P.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(row + col > 0, 2 * tp / (row + col), np.nan)
            iou = np.where(row + col - tp > 0, tp / (row + col - tp), np.nan)
        oa = float(tp.sum()); pe = float((row * col).sum())
        kappa = (oa - pe) / (1 - pe) if pe < 1 else float("nan")
        q = float(0.5 * np.abs(row - col).sum())
        return f1, iou, kappa, q, (1 - oa) - q

    P = proportions(n)
    f1, iou, kappa_, q, a = measures(P)
    rng = np.random.default_rng(seed)
    nh = n.sum(axis=1).astype(int)
    boots = []
    for _ in range(int(n_boot)):
        nn = np.stack([rng.multinomial(nh[h], n[h] / nh[h]) for h in range(H)]).astype(float)
        boots.append(measures(proportions(nn)))

    def ci(i):
        arr = np.array([b[i] for b in boots], dtype=float)
        if arr.size == 0:
            return None
        lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
        return np.stack([lo, hi], axis=-1) if np.ndim(lo) else (float(lo), float(hi))

    return {"P": P, "f1": f1, "iou": iou, "kappa": kappa_, "quantity_disagreement": q, "allocation_disagreement": a,
            "f1_ci95": ci(0), "iou_ci95": ci(1), "kappa_ci95": ci(2),
            "macro_f1": float(np.nanmean(f1)), "mean_iou": float(np.nanmean(iou)), "n_boot": int(n_boot)}

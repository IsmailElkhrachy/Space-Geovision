"""Supervised land-cover classification with real, reproducible accuracy assessment.

Inputs : a multi-band raster + a vector layer of labelled samples (points or polygons).
Outputs: classified GeoTIFF, confusion matrix (CSV), text report, JSON provenance.

Validation is either (a) an INDEPENDENT vector layer of validation samples, or (b) a spatially
blocked hold-out of the training layer. A random per-pixel split is deliberately not offered.
NumPy/scikit-learn only at import time; rasterio/geopandas are imported lazily by the I/O helpers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import accuracy as acc

MODELS = ("rf", "svm_linear", "svm_rbf")


@dataclass
class ClassifierConfig:
    model: str = "rf"
    n_trees: int = 100
    tune_svm: bool = True            # grid-search C (and gamma for RBF) with 3-fold CV on the training set
    block_px: int = 100              # size of validation blocks when no independent set is given
    test_fraction: float = 0.3
    seed: int = 42


def make_model(cfg: ClassifierConfig):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if cfg.model == "rf":
        return RandomForestClassifier(n_estimators=cfg.n_trees, bootstrap=True, oob_score=True,
                                      class_weight="balanced", n_jobs=-1, random_state=cfg.seed)
    if cfg.model in ("svm_linear", "svm_rbf"):
        kernel = "linear" if cfg.model == "svm_linear" else "rbf"
        pipe = make_pipeline(StandardScaler(), SVC(kernel=kernel, class_weight="balanced", random_state=cfg.seed))
        if not cfg.tune_svm:
            return pipe
        grid = {"svc__C": [0.1, 1, 10, 100]}
        if kernel == "rbf":
            grid["svc__gamma"] = ["scale", 0.01, 0.1, 1]
        return GridSearchCV(pipe, grid, cv=3, n_jobs=-1)
    raise ValueError(f"Unknown model '{cfg.model}' (choose from {MODELS}).")


@dataclass
class Evaluation:
    labels: List[int]
    names: Dict[int, str]
    cm: np.ndarray
    summary: Dict
    validation: str
    n_train: int
    n_val: int
    oob_score: Optional[float] = None
    best_params: Optional[Dict] = None
    feature_importance: Optional[List[float]] = None


def train_and_evaluate(X_train, y_train, X_val, y_val, cfg: ClassifierConfig, names: Optional[Dict[int, str]] = None,
                       validation: str = "independent"):
    """Fit on the training samples and evaluate ONLY on the held-out validation samples."""
    X_train, y_train = np.asarray(X_train, dtype=np.float32), np.asarray(y_train)
    X_val, y_val = np.asarray(X_val, dtype=np.float32), np.asarray(y_val)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Need at least two classes in the training samples.")
    model = make_model(cfg)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    labels = sorted(set(np.unique(y_train)) | set(np.unique(y_val)))
    cm, labels = acc.confusion_matrix(y_val, pred, labels)
    est = getattr(model, "best_estimator_", model)
    ev = Evaluation(
        labels=[int(l) for l in labels], names=names or {int(l): str(l) for l in labels}, cm=cm,
        summary=acc.summarize(cm), validation=validation, n_train=len(y_train), n_val=len(y_val),
        oob_score=float(model.oob_score_) if hasattr(model, "oob_score_") else None,
        best_params=getattr(model, "best_params_", None),
        feature_importance=[float(v) for v in model.feature_importances_] if hasattr(model, "feature_importances_") else None)
    return model, ev


def predict_raster(model, stack: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    """Classify a (bands, rows, cols) stack. Pixels with NaN in any band get 0 (= no data)."""
    if stack.ndim != 3:
        raise ValueError("stack must have shape (bands, rows, cols)")
    b, h, w = stack.shape
    flat = stack.reshape(b, -1).T
    valid = np.isfinite(flat).all(axis=1)
    out = np.zeros(flat.shape[0], dtype=np.uint8)
    idx = np.where(valid)[0]
    for s in range(0, idx.size, chunk):
        sel = idx[s:s + chunk]
        out[sel] = model.predict(flat[sel].astype(np.float32))
    return out.reshape(h, w)


def sample_stack(stack: np.ndarray, label_raster: np.ndarray):
    """Feature vectors, labels and pixel coordinates for every labelled, valid pixel."""
    rows, cols = np.nonzero(label_raster > 0)
    X = stack[:, rows, cols].T
    ok = np.isfinite(X).all(axis=1)
    return X[ok], label_raster[rows[ok], cols[ok]], rows[ok], cols[ok]


# ------------------------------------------------------------------ raster / vector I/O (lazy imports)
def read_stack(path: str):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        return arr, src.profile.copy(), src.transform, src.crs


def rasterize_samples(vector_path: str, class_field: str, shape: Tuple[int, int], transform, raster_crs,
                      class_map: Optional[Dict[str, int]] = None):
    """Burn class ids (1..n) of a point/polygon layer onto the raster grid. Returns (label_raster, class_map)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = gpd.read_file(vector_path)
    if class_field not in gdf.columns:
        raise ValueError(f"Field '{class_field}' not found. Available: {', '.join(map(str, gdf.columns))}")
    if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf[class_field].notna()]
    if class_map is None:
        class_map = {str(v): i + 1 for i, v in enumerate(sorted(gdf[class_field].astype(str).unique()))}
    else:
        unknown = set(gdf[class_field].astype(str)) - set(class_map)
        if unknown:
            raise ValueError(f"Validation layer has classes not present in training: {sorted(unknown)}")
    shapes = [(g, class_map[str(v)]) for g, v in zip(gdf.geometry, gdf[class_field])]
    lab = rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8")
    return lab, class_map


def run_supervised(raster_path: str, train_vector: str, class_field: str, out_dir: str,
                   cfg: Optional[ClassifierConfig] = None, val_vector: Optional[str] = None,
                   progress=None) -> Dict:
    """Complete workflow used by the GUI. Returns paths + summary; never reports training accuracy."""
    import rasterio
    cfg = cfg or ClassifierConfig()
    os.makedirs(out_dir, exist_ok=True)
    say = progress or (lambda pct, msg: None)

    say(5, "Reading raster")
    stack, profile, transform, crs = read_stack(raster_path)
    shape = stack.shape[1:]

    say(15, "Rasterising training samples")
    train_lab, class_map = rasterize_samples(train_vector, class_field, shape, transform, crs)
    names = {v: k for k, v in class_map.items()}

    if val_vector:
        val_lab, _ = rasterize_samples(val_vector, class_field, shape, transform, crs, class_map)
        overlap = (train_lab > 0) & (val_lab > 0)
        n_overlap = int(overlap.sum())
        val_lab[overlap] = 0     # a pixel can never be both a training and a validation sample
        Xt, yt, _, _ = sample_stack(stack, train_lab)
        Xv, yv, _, _ = sample_stack(stack, val_lab)
        mode = f"independent validation layer ({n_overlap} overlapping pixels removed)"
    else:
        X, y, rows, cols = sample_stack(stack, train_lab)
        tr, te = acc.blocked_split(rows, cols, y, cfg.block_px, cfg.test_fraction, cfg.seed)
        Xt, yt, Xv, yv = X[tr], y[tr], X[te], y[te]
        mode = f"spatially blocked hold-out ({cfg.block_px}px blocks, {int(cfg.test_fraction * 100)}% of blocks)"
    if len(yv) == 0:
        raise ValueError("No validation samples fall on valid raster pixels.")

    say(35, f"Training {cfg.model}")
    model, ev = train_and_evaluate(Xt, yt, Xv, yv, cfg, names, mode)

    say(70, "Classifying the full raster")
    classified = predict_raster(model, stack)

    say(90, "Writing outputs")
    stem = os.path.splitext(os.path.basename(raster_path))[0]
    tif = os.path.join(out_dir, f"{stem}_{cfg.model}_classified.tif")
    prof = profile.copy()
    prof.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(tif, "w", **prof) as dst:
        dst.write(classified, 1)
    csv_path = os.path.join(out_dir, f"{stem}_{cfg.model}_confusion_matrix.csv")
    acc.save_csv(ev.cm, ev.labels, csv_path, names)
    report = acc.format_report(ev.cm, ev.labels, names)
    header = (f"Model: {cfg.model}\nValidation: {mode}\nTraining pixels: {ev.n_train}   Validation pixels: {ev.n_val}\n"
              + (f"OOB score (training data, NOT a validation result): {ev.oob_score:.3f}\n" if ev.oob_score is not None else "")
              + (f"Selected SVM parameters: {ev.best_params}\n" if ev.best_params else "") + "\n")
    txt_path = os.path.join(out_dir, f"{stem}_{cfg.model}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + report + "\n\nNote: pixels are not independent samples; confidence intervals are optimistic.\n")
    prov = os.path.join(out_dir, f"{stem}_{cfg.model}_provenance.json")
    with open(prov, "w", encoding="utf-8") as f:
        json.dump({"raster": os.path.basename(raster_path), "train_vector": os.path.basename(train_vector),
                   "val_vector": os.path.basename(val_vector) if val_vector else None, "class_field": class_field,
                   "classes": names, "config": asdict(cfg), "validation": mode,
                   "overall_accuracy": ev.summary["overall_accuracy"], "kappa": ev.summary["kappa"],
                   "feature_importance": ev.feature_importance}, f, indent=2)
    say(100, "Done")
    return {"classified_tif": tif, "confusion_csv": csv_path, "report_txt": txt_path, "provenance": prov,
            "report": header + report, "evaluation": ev, "classified": classified, "profile": prof}

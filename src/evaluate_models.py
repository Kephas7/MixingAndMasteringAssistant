"""
evaluate_models.py — MMMA Model Evaluation for Thesis Findings
================================================================

Loads the already-trained EQ and Compression RandomForest models and
reproduces the EXACT held-out test set used at training time (same
track-level split, same random_state=42, same segmentation), then
computes a full evaluation report:

  1. Overall R2 / MAE / RMSE on the held-out test set (confirms the
     previously reported R2 of 0.72 for EQ and 0.91 for compression).
  2. Per-target R2 / MAE / RMSE (per EQ band, per compression parameter).
  3. RandomForest feature importances, ranked.
  4. 5-fold group (track-level) cross-validation R2 mean +/- std, to
     check whether the headline R2 is stable or a lucky split.
  5. Predicted-vs-actual and residual diagnostic plots per model.

Nothing here is hand-picked: the track-level split logic, segmentation,
and feature/target definitions are imported directly from
train_eq_model.py and train_compression_model.py so the held-out set is
byte-for-byte the same population of segments used originally.

Usage
-----
    python src/evaluate_models.py [/path/to/musdb18]

    Defaults to data/raw/musdb18 if no path is given.

Outputs
-------
    results/model_metrics.txt              Full text report
    results/eq_pred_vs_actual.png
    results/eq_residuals.png
    results/compression_pred_vs_actual.png
    results/compression_residuals.png
    data/processed/eval_dataset_cache.joblib   Cached extracted features
                                                (delete to force re-extraction)
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_eq_model as eqm
import train_compression_model as cm

# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_MUSDB_ROOT = os.path.join("data", "raw", "musdb18")
EQ_MODEL_PATH = os.path.join("models", "eq_model.joblib")
COMP_MODEL_PATH = os.path.join("models", "compression_model.joblib")
RESULTS_DIR = "results"
CACHE_PATH = os.path.join("data", "processed", "eval_dataset_cache.joblib")

SEGMENT_DUR = 30.0
HOP_DUR = 15.0
VAL_FRAC = 0.20
RANDOM_STATE = 42
N_CV_FOLDS = 5

REPORTED_R2 = {"eq": 0.72, "compression": 0.91}


# ── Dataset reconstruction (must mirror training exactly) ──────────────────

def _process_track(name: str, y: np.ndarray, sr: int):
    """Extract EQ and compression (X, Y) rows for every segment of one track."""
    segments = eqm._segment_track(y, sr, segment_dur=SEGMENT_DUR, hop_dur=HOP_DUR)
    rows = []
    for seg in segments:
        x_eq = eqm.extract_audio_features(seg, sr)
        energies = eqm.extract_band_energies(seg, sr)
        y_eq = np.array([energies[b] for b in eqm.BAND_NAMES], dtype=np.float32)

        x_comp = cm.extract_compression_features(seg, sr)
        y_comp = cm.derive_compression_targets(seg, sr)

        rows.append((x_eq, y_eq, x_comp, y_comp, name))
    return rows


def build_full_dataset(musdb_root: str, use_cache: bool = True) -> dict:
    """
    Reproduce the full 150-track MUSDB18 dataset (train+test combined),
    the same track-level 80/20 split used in training, and extract
    features/targets for every segment of every track.

    Returns a dict with X_eq, Y_eq, X_comp, Y_comp, groups (track name per
    segment), train_track_names, val_track_names.
    """
    cfg_key = dict(segment_dur=SEGMENT_DUR, hop_dur=HOP_DUR, val_frac=VAL_FRAC,
                    random_state=RANDOM_STATE, musdb_root=os.path.abspath(musdb_root))

    if use_cache and os.path.exists(CACHE_PATH):
        cached = joblib.load(CACHE_PATH)
        if cached.get("_config") == cfg_key:
            print(f"[cache] Loaded extracted features from {CACHE_PATH}")
            return cached
        print("[cache] Cache config mismatch — re-extracting.")

    print("=" * 70)
    print("Reconstructing dataset (this mirrors train_eq_model.py exactly)")
    print("=" * 70)

    t0 = time.time()
    all_tracks = eqm._load_all_tracks(musdb_root)
    if not all_tracks:
        raise RuntimeError(f"No tracks loaded from {musdb_root}")
    print(f"\nLoaded {len(all_tracks)} tracks in {time.time()-t0:.1f}s")

    train_tracks, val_tracks = eqm._split_tracks(
        all_tracks, val_frac=VAL_FRAC, random_state=RANDOM_STATE
    )
    train_names = [t[0] for t in train_tracks]
    val_names = [t[0] for t in val_tracks]
    print(f"Track-level split (random_state={RANDOM_STATE}, val_frac={VAL_FRAC}):")
    print(f"  Train tracks : {len(train_names)}")
    print(f"  Val tracks   : {len(val_names)}")

    print(f"\nExtracting EQ + compression features for all {len(all_tracks)} tracks "
          f"(parallel, {os.cpu_count()} CPUs)...")
    t0 = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_process_track)(name, y, sr) for name, y, sr in all_tracks
    )
    print(f"Feature extraction done in {time.time()-t0:.1f}s")

    X_eq_rows, Y_eq_rows, X_comp_rows, Y_comp_rows, groups = [], [], [], [], []
    for track_rows in results:
        for x_eq, y_eq, x_comp, y_comp, name in track_rows:
            X_eq_rows.append(x_eq)
            Y_eq_rows.append(y_eq)
            X_comp_rows.append(x_comp)
            Y_comp_rows.append(y_comp)
            groups.append(name)

    dataset = {
        "X_eq": np.stack(X_eq_rows).astype(np.float32),
        "Y_eq": np.stack(Y_eq_rows).astype(np.float32),
        "X_comp": np.stack(X_comp_rows).astype(np.float32),
        "Y_comp": np.stack(Y_comp_rows).astype(np.float32),
        "groups": np.array(groups),
        "train_track_names": train_names,
        "val_track_names": val_names,
        "_config": cfg_key,
    }

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    joblib.dump(dataset, CACHE_PATH, compress=3)
    print(f"\nCached extracted dataset -> {CACHE_PATH}")
    return dataset


# ── Metric computation ───────────────────────────────────────────────────────

def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, target_names: List[str]) -> dict:
    """Overall + per-target R2 / MAE / RMSE using sklearn's own functions."""
    overall = {
        "r2": float(r2_score(Y_true, Y_pred, multioutput="uniform_average")),
        "mae": float(mean_absolute_error(Y_true, Y_pred, multioutput="uniform_average")),
        "rmse": float(root_mean_squared_error(Y_true, Y_pred, multioutput="uniform_average")),
    }
    per_target = {}
    for i, name in enumerate(target_names):
        per_target[name] = {
            "r2": float(r2_score(Y_true[:, i], Y_pred[:, i])),
            "mae": float(mean_absolute_error(Y_true[:, i], Y_pred[:, i])),
            "rmse": float(root_mean_squared_error(Y_true[:, i], Y_pred[:, i])),
        }
    return {"overall": overall, "per_target": per_target}


def feature_importance_ranking(rf: RandomForestRegressor, feature_names: List[str]) -> List[Tuple[str, float]]:
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    return [(feature_names[i], float(importances[i])) for i in order]


def group_kfold_cv(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, rf_params: dict,
                    n_splits: int = N_CV_FOLDS) -> np.ndarray:
    """
    5-fold cross-validation with track-level grouping (GroupKFold), so
    segments from the same track never span both the train and test side
    of a fold. A fresh StandardScaler + RandomForestRegressor (same
    hyperparameters as the saved model) is fit per fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=groups), 1):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(**rf_params)),
        ])
        pipeline.fit(X[train_idx], Y[train_idx])
        Y_pred = pipeline.predict(X[test_idx])
        r2 = r2_score(Y[test_idx], Y_pred, multioutput="uniform_average")
        fold_r2.append(r2)
        n_train_tracks = len(set(groups[train_idx]))
        n_test_tracks = len(set(groups[test_idx]))
        print(f"    Fold {fold_idx}/{n_splits}: train={len(train_idx)} segs "
              f"({n_train_tracks} tracks), test={len(test_idx)} segs "
              f"({n_test_tracks} tracks)  ->  R2 = {r2:.4f}")
    return np.array(fold_r2)


# ── Plotting ─────────────────────────────────────────────────────────────────

def _grid_shape(n: int) -> Tuple[int, int]:
    ncols = 4 if n > 4 else n
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def plot_pred_vs_actual(Y_true: np.ndarray, Y_pred: np.ndarray, target_names: List[str],
                         title: str, out_path: str) -> None:
    n = len(target_names)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for i, name in enumerate(target_names):
        ax = axes[i // ncols][i % ncols]
        yt, yp = Y_true[:, i], Y_pred[:, i]
        ax.scatter(yt, yp, s=8, alpha=0.35, edgecolors="none")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        r2 = r2_score(yt, yp)
        ax.set_title(f"{name}  (R2={r2:.3f})")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved plot -> {out_path}")


def plot_residuals(Y_true: np.ndarray, Y_pred: np.ndarray, target_names: List[str],
                    title: str, out_path: str) -> None:
    n = len(target_names)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for i, name in enumerate(target_names):
        ax = axes[i // ncols][i % ncols]
        yt, yp = Y_true[:, i], Y_pred[:, i]
        residuals = yt - yp
        ax.scatter(yp, residuals, s=8, alpha=0.35, edgecolors="none")
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (actual - predicted)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved plot -> {out_path}")


# ── Report formatting ────────────────────────────────────────────────────────

def format_metrics_table(metrics: dict, target_names: List[str], units: str) -> str:
    lines = []
    o = metrics["overall"]
    lines.append(f"  Overall  R2={o['r2']:.4f}   MAE={o['mae']:.4f} {units}   RMSE={o['rmse']:.4f} {units}")
    lines.append("")
    name_w = max(len(n) for n in target_names) + 2
    lines.append(f"  {'Target':<{name_w}}{'R2':>10}{'MAE':>14}{'RMSE':>14}")
    for name in target_names:
        m = metrics["per_target"][name]
        lines.append(f"  {name:<{name_w}}{m['r2']:>10.4f}{m['mae']:>14.4f}{m['rmse']:>14.4f}")
    return "\n".join(lines)


def format_importances(ranked: List[Tuple[str, float]]) -> str:
    lines = []
    for rank, (name, val) in enumerate(ranked, 1):
        lines.append(f"    {rank:2d}. {name:<28s} {val:.4f}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained EQ and compression models")
    parser.add_argument("musdb_path", nargs="?", default=DEFAULT_MUSDB_ROOT,
                         help="Path to the MUSDB18 root directory")
    parser.add_argument("--no-cache", action="store_true",
                         help="Force re-extraction of features, ignoring any cache")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(EQ_MODEL_PATH) or not os.path.exists(COMP_MODEL_PATH):
        print(f"Error: expected trained models at {EQ_MODEL_PATH} and {COMP_MODEL_PATH}")
        sys.exit(1)

    eq_bundle = joblib.load(EQ_MODEL_PATH)
    comp_bundle = joblib.load(COMP_MODEL_PATH)
    eq_pipeline = eq_bundle["pipeline"]
    comp_pipeline = comp_bundle["pipeline"]
    eq_targets = eq_bundle["band_names"]
    comp_targets = comp_bundle["target_names"]
    eq_feature_names = eq_bundle["feature_names"]
    comp_feature_names = comp_bundle["feature_names"]

    dataset = build_full_dataset(args.musdb_path, use_cache=not args.no_cache)

    val_names = set(dataset["val_track_names"])
    groups = dataset["groups"]
    val_mask = np.array([g in val_names for g in groups])

    X_eq_val = dataset["X_eq"][val_mask]
    Y_eq_val = dataset["Y_eq"][val_mask]
    X_comp_val = dataset["X_comp"][val_mask]
    Y_comp_val = dataset["Y_comp"][val_mask]

    report_lines = []

    def emit(line: str = ""):
        print(line)
        report_lines.append(line)

    emit("=" * 70)
    emit("MMMA MODEL EVALUATION REPORT")
    emit("=" * 70)
    emit(f"MUSDB18 root       : {os.path.abspath(args.musdb_path)}")
    emit(f"Total tracks       : {len(dataset['train_track_names']) + len(dataset['val_track_names'])}")
    emit(f"Train tracks       : {len(dataset['train_track_names'])}")
    emit(f"Held-out val tracks: {len(dataset['val_track_names'])}")
    emit(f"Held-out val segs  : EQ={X_eq_val.shape[0]}, Compression={X_comp_val.shape[0]}")
    emit(f"Segment duration   : {SEGMENT_DUR}s, hop {HOP_DUR}s (50% overlap)")
    emit(f"Split strategy     : track-level, random_state={RANDOM_STATE}, val_frac={VAL_FRAC}")
    emit("(Assumes default --segment-dur 30 was used at training time, matching")
    emit(" train_eq_model.py / train_compression_model.py defaults; verified below")
    emit(" by reproducing the previously reported overall R2 values.)")
    emit("")

    # ── 1 & 2: Overall + per-target metrics on held-out test set ───────────
    Y_eq_pred_val = eq_pipeline.predict(X_eq_val)
    Y_comp_pred_val = comp_pipeline.predict(X_comp_val)

    eq_metrics = compute_metrics(Y_eq_val, Y_eq_pred_val, eq_targets)
    comp_metrics = compute_metrics(Y_comp_val, Y_comp_pred_val, comp_targets)

    emit("-" * 70)
    emit("1-2. HELD-OUT TEST SET METRICS")
    emit("-" * 70)

    emit("\n[EQ MODEL]  targets = normalized per-band energy (0-1, unitless)")
    reproduced = eq_metrics["overall"]["r2"]
    diff = reproduced - REPORTED_R2["eq"]
    emit(f"  Reported R2 (thesis draft) : {REPORTED_R2['eq']:.4f}")
    emit(f"  Reproduced R2 (this run)   : {reproduced:.4f}   (diff = {diff:+.4f})")
    if abs(diff) < 0.02:
        emit("  -> MATCH: reproduced R2 confirms the held-out set was reconstructed correctly.")
    else:
        emit("  -> MISMATCH: reproduced R2 differs from the reported value by more than 0.02.")
        emit("     Do not trust downstream numbers until this is investigated (segment_dur,")
        emit("     max_tracks, or dataset version may differ from the original training run).")
    emit("")
    emit(format_metrics_table(eq_metrics, eq_targets, units="(0-1 scale)"))

    emit("\n[COMPRESSION MODEL]  targets = real units (ratio unitless, others in dB/ms)")
    reproduced_c = comp_metrics["overall"]["r2"]
    diff_c = reproduced_c - REPORTED_R2["compression"]
    emit(f"  Reported R2 (thesis draft) : {REPORTED_R2['compression']:.4f}")
    emit(f"  Reproduced R2 (this run)   : {reproduced_c:.4f}   (diff = {diff_c:+.4f})")
    if abs(diff_c) < 0.02:
        emit("  -> MATCH: reproduced R2 confirms the held-out set was reconstructed correctly.")
    else:
        emit("  -> MISMATCH: reproduced R2 differs from the reported value by more than 0.02.")
        emit("     Do not trust downstream numbers until this is investigated.")
    emit("")
    emit("  NOTE: 'Overall' MAE/RMSE below averages across ratio (unitless), threshold_db (dB),")
    emit("  attack_ms and release_ms (ms) simultaneously -- mixed units, included only because")
    emit("  it was requested. The per-target rows are the physically meaningful numbers.")
    emit(format_metrics_table(comp_metrics, comp_targets, units="(mixed units)"))

    # ── 3: Feature importances ──────────────────────────────────────────────
    emit("\n" + "-" * 70)
    emit("3. FEATURE IMPORTANCES (RandomForestRegressor.feature_importances_)")
    emit("-" * 70)

    eq_rf = eq_pipeline.named_steps["rf"]
    comp_rf = comp_pipeline.named_steps["rf"]
    eq_importances = feature_importance_ranking(eq_rf, eq_feature_names)
    comp_importances = feature_importance_ranking(comp_rf, comp_feature_names)

    emit(f"\n[EQ MODEL] all {len(eq_importances)} features, most to least important:")
    emit(format_importances(eq_importances))

    emit(f"\n[COMPRESSION MODEL] all {len(comp_importances)} features, most to least important:")
    emit(format_importances(comp_importances))

    # ── 4: Group k-fold cross-validation ────────────────────────────────────
    emit("\n" + "-" * 70)
    emit(f"4. {N_CV_FOLDS}-FOLD GROUP (TRACK-LEVEL) CROSS-VALIDATION")
    emit("-" * 70)
    emit(f"Refit on the FULL {len(dataset['train_track_names']) + len(dataset['val_track_names'])}-track dataset "
         f"(train+held-out combined), {N_CV_FOLDS} folds split by track so no")
    emit("track's segments appear in both the train and test side of any fold.")
    emit("This is a stability check on the headline R2, independent of the single")
    emit("held-out split evaluated above; it retrains fresh models per fold.")

    eq_rf_params = eq_rf.get_params()
    comp_rf_params = comp_rf.get_params()

    emit(f"\n[EQ MODEL] RandomForest params: n_estimators={eq_rf_params['n_estimators']}, "
         f"max_depth={eq_rf_params['max_depth']}")
    eq_cv_r2 = group_kfold_cv(dataset["X_eq"], dataset["Y_eq"], groups, eq_rf_params)
    emit(f"  Fold R2 scores : {np.array2string(eq_cv_r2, precision=4)}")
    emit(f"  Mean R2 +/- SD : {eq_cv_r2.mean():.4f} +/- {eq_cv_r2.std():.4f}")

    emit(f"\n[COMPRESSION MODEL] RandomForest params: n_estimators={comp_rf_params['n_estimators']}, "
         f"max_depth={comp_rf_params['max_depth']}")
    comp_cv_r2 = group_kfold_cv(dataset["X_comp"], dataset["Y_comp"], groups, comp_rf_params)
    emit(f"  Fold R2 scores : {np.array2string(comp_cv_r2, precision=4)}")
    emit(f"  Mean R2 +/- SD : {comp_cv_r2.mean():.4f} +/- {comp_cv_r2.std():.4f}")

    # ── 5: Diagnostic plots ──────────────────────────────────────────────────
    emit("\n" + "-" * 70)
    emit("5. DIAGNOSTIC PLOTS (held-out test set)")
    emit("-" * 70)

    plot_pred_vs_actual(Y_eq_val, Y_eq_pred_val, eq_targets,
                         "EQ Model: Predicted vs Actual (held-out test set)",
                         os.path.join(RESULTS_DIR, "eq_pred_vs_actual.png"))
    plot_residuals(Y_eq_val, Y_eq_pred_val, eq_targets,
                    "EQ Model: Residuals vs Predicted (held-out test set)",
                    os.path.join(RESULTS_DIR, "eq_residuals.png"))
    plot_pred_vs_actual(Y_comp_val, Y_comp_pred_val, comp_targets,
                         "Compression Model: Predicted vs Actual (held-out test set)",
                         os.path.join(RESULTS_DIR, "compression_pred_vs_actual.png"))
    plot_residuals(Y_comp_val, Y_comp_pred_val, comp_targets,
                    "Compression Model: Residuals vs Predicted (held-out test set)",
                    os.path.join(RESULTS_DIR, "compression_residuals.png"))

    emit("\n" + "=" * 70)
    emit("Evaluation complete.")
    emit("=" * 70)

    out_path = os.path.join(RESULTS_DIR, "model_metrics.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    main()

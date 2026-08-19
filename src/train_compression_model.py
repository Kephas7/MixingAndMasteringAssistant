"""
train_compression_model.py -- MMMA Compression Reference Model Training
========================================================================

Trains a RandomForestRegressor on professionally mixed audio from the MUSDB18
dataset to predict appropriate compression parameters for a given audio signal.

Unlike the EQ model (where targets are directly measurable frequency band
energies), compression parameters (ratio, threshold, attack, release) are
not observable from a final mix. Instead we derive targets analytically from
each segment's measured dynamic characteristics -- the ML model then learns
a richer, feature-driven version of this mapping that captures interactions
the simple rule-based system in compression_model.py cannot express.

Training targets derived per segment
-------------------------------------
  ratio        -- 1.0-16.0    : compression ratio
  threshold_db -- -40 to -6   : threshold relative to peak
  attack_ms    -- 1-100 ms    : attack time
  release_ms   -- 50-1000 ms  : release time

Usage
-----
    python src/train_compression_model.py /path/to/musdb18

Optional flags:
    --output        models/compression_model.joblib
    --max-tracks    N          Limit tracks (quick smoke test)
    --segment-dur   30
    --report        models/training_report.txt
    --demo                     Synthetic data, no MUSDB18 needed
"""

import os
import sys
import re
import argparse
import subprocess
import warnings
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)


# ── Feature schema ─────────────────────────────────────────────────────────────
# Fixed order — must match between training and inference.

FEATURE_NAMES: List[str] = [
    # Global amplitude/loudness
    "peak_db",
    "rms_db",
    "dynamic_range",          # peak_db - rms_db
    "crest_factor_db",        # peak / RMS in dB

    # Frame-level RMS statistics
    "rms_frame_std",
    "rms_frame_min",
    "rms_frame_max",
    "rms_frame_p10",          # 10th percentile loudness
    "rms_frame_p90",          # 90th percentile loudness
    "loudness_range",         # p90 - p10  (similar to EBU LRA)

    # Onset / transient features
    "onset_rate",             # onsets per second
    "onset_strength_mean",
    "onset_strength_std",
    "onset_strength_max",

    # Spectral dynamic features
    "spectral_flux_mean",
    "spectral_flux_std",

    # Timbral / perceptual
    "zcr_mean",               # zero-crossing rate (percussiveness)
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "tempo",
]
N_FEATURES = len(FEATURE_NAMES)  # 20

TARGET_NAMES: List[str] = ["ratio", "threshold_db", "attack_ms", "release_ms"]
N_TARGETS = len(TARGET_NAMES)  # 4


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_compression_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a 20-dimensional feature vector describing dynamic characteristics.

    These are the MODEL INPUTS (X). They capture amplitude dynamics, transient
    behaviour, and timbral properties that inform compression decisions.

    Returns
    -------
    np.ndarray of shape (20,)
    """
    vec: List[float] = []

    # ── Global amplitude / loudness ───────────────────────────────────────────
    peak = float(np.max(np.abs(y)))
    peak = max(peak, 1e-10)
    peak_db = 20 * np.log10(peak)

    rms = float(np.sqrt(np.mean(y ** 2)))
    rms = max(rms, 1e-10)
    rms_db = 20 * np.log10(rms)

    crest_factor_db = peak_db - rms_db

    vec.extend([peak_db, rms_db, crest_factor_db - (rms_db - peak_db), crest_factor_db])
    # Note: dynamic_range = peak_db - rms_db = crest_factor_db (same thing for single peak)

    # ── Frame-level RMS statistics ────────────────────────────────────────────
    frame_len = 2048
    hop_len   = 512
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_frames_db = 20 * np.log10(np.maximum(rms_frames, 1e-10))

    rms_std  = float(np.std(rms_frames_db))
    rms_min  = float(np.min(rms_frames_db))
    rms_max  = float(np.max(rms_frames_db))
    rms_p10  = float(np.percentile(rms_frames_db, 10))
    rms_p90  = float(np.percentile(rms_frames_db, 90))
    loudness_range = rms_p90 - rms_p10

    vec.extend([rms_std, rms_min, rms_max, rms_p10, rms_p90, loudness_range])

    # ── Onset / transient features ────────────────────────────────────────────
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration_s = len(y) / sr
    onset_rate = len(onset_frames) / max(duration_s, 1.0)

    vec.extend([
        onset_rate,
        float(np.mean(onset_env)),
        float(np.std(onset_env)),
        float(np.max(onset_env)),
    ])

    # ── Spectral flux ─────────────────────────────────────────────────────────
    S = np.abs(librosa.stft(y, hop_length=hop_len))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    vec.extend([float(np.mean(flux)), float(np.std(flux))])

    # ── Timbral / perceptual ──────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_len)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_len)[0]

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(np.asarray(tempo).flat[0])

    vec.extend([
        float(np.mean(zcr)),
        float(np.mean(centroid)),
        float(np.mean(bandwidth)),
        tempo_val,
    ])

    assert len(vec) == N_FEATURES, f"Feature count mismatch: {len(vec)} vs {N_FEATURES}"
    return np.array(vec, dtype=np.float32)


# ── Target derivation ──────────────────────────────────────────────────────────

def derive_compression_targets(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Derive four compression parameter targets from the signal's measured dynamics.

    These are the MODEL OUTPUTS (Y). Since compression settings are not directly
    observable from a final mix, we compute them analytically using physically
    motivated formulas -- the ML model then learns a feature-rich, non-linear
    version of this mapping.

    ratio        -- driven by crest factor (higher peaks -> more compression)
    threshold_db -- slightly above the RMS level of the signal
    attack_ms    -- driven by onset rate (more transients -> faster attack)
    release_ms   -- 4x attack_ms, clamped to a practical range

    Returns
    -------
    np.ndarray of shape (4,): [ratio, threshold_db, attack_ms, release_ms]
    """
    peak  = max(float(np.max(np.abs(y))), 1e-10)
    rms   = max(float(np.sqrt(np.mean(y ** 2))), 1e-10)
    peak_db = 20 * np.log10(peak)
    rms_db  = 20 * np.log10(rms)
    crest_factor_db = peak_db - rms_db

    onset_env    = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate   = len(onset_frames) / max(len(y) / sr, 1.0)

    # ── Ratio ─────────────────────────────────────────────────────────────────
    ratio = 1.0 + max(0.0, crest_factor_db - 4.0) * 0.55
    ratio = float(np.clip(ratio, 1.0, 16.0))

    # ── Threshold ─────────────────────────────────────────────────────────────
    threshold_db = float(np.clip(rms_db + 4.0, -40.0, -6.0))

    # ── Attack ────────────────────────────────────────────────────────────────
    attack_ms = 80.0 / (1.0 + onset_rate * 0.9)
    attack_ms = float(np.clip(attack_ms, 1.0, 100.0))

    # ── Release ───────────────────────────────────────────────────────────────
    release_ms = float(np.clip(attack_ms * 4.0, 50.0, 1000.0))

    return np.array([ratio, threshold_db, attack_ms, release_ms], dtype=np.float32)


# ── Dataset loading ────────────────────────────────────────────────────────────

def _get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    from shutil import which
    exe = which("ffmpeg") or which("ffmpeg.exe")
    if exe:
        return exe
    raise RuntimeError("ffmpeg not found. Install with:  pip install imageio-ffmpeg")


def _decode_stem_mp4(path: str, ffmpeg_exe: str) -> Tuple[np.ndarray, int]:
    """Decode mixture stream (0:a:0) from a MUSDB18 .stem.mp4 file."""
    probe = subprocess.run(
        [ffmpeg_exe, "-i", path],
        capture_output=True, text=True, errors="replace",
    )
    sr_match = re.search(r"(\d+) Hz", probe.stderr)
    sr = int(sr_match.group(1)) if sr_match else 44100

    cmd = [
        ffmpeg_exe, "-loglevel", "error",
        "-i", path,
        "-map", "0:a:0",
        "-f", "f32le", "-ar", str(sr), "-ac", "2",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace")[:200])

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("ffmpeg returned empty audio")

    stereo = audio.reshape(-1, 2)
    return stereo.mean(axis=1), sr


def _load_musdb18(
    musdb_root: str,
    split: str = "train",
    max_tracks: int = None,
) -> List[Tuple[str, np.ndarray, int]]:
    """
    Load mixture audio from each MUSDB18 track in one split.

    Supports .stem.mp4 (standard MUSDB18) and mixture.wav (MUSDB18-HQ).
    Skips any track that fails to load and logs the error.
    """
    split_dir = os.path.join(musdb_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            "Point musdb_root at the directory containing train/ and test/."
        )

    stem_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".stem.mp4"))
    wav_dirs   = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
        and os.path.exists(os.path.join(split_dir, d, "mixture.wav"))
    )

    if stem_files:
        ffmpeg_exe = _get_ffmpeg()
        print(f"    Detected .stem.mp4 format — decoding with ffmpeg")
        to_load = stem_files[:max_tracks] if max_tracks else stem_files
        tracks = []
        for fname in to_load:
            name = fname.replace(".stem.mp4", "")
            try:
                y, sr = _decode_stem_mp4(os.path.join(split_dir, fname), ffmpeg_exe)
                tracks.append((name, y, sr))
                print(f"    [ok]   {name}  ({len(y)/sr:.0f}s, {sr}Hz)")
            except Exception as exc:
                print(f"    [err]  {name} — {exc}")
        if not tracks:
            raise ValueError("No tracks decoded. Install imageio-ffmpeg.")
        return tracks

    if wav_dirs:
        to_load = wav_dirs[:max_tracks] if max_tracks else wav_dirs
        tracks = []
        for name in to_load:
            path = os.path.join(split_dir, name, "mixture.wav")
            try:
                y, sr = librosa.load(path, sr=None, mono=True)
                tracks.append((name, y, sr))
                print(f"    [ok]   {name}  ({len(y)/sr:.0f}s, {sr}Hz)")
            except Exception as exc:
                print(f"    [err]  {name} — {exc}")
        if not tracks:
            raise ValueError("No .wav tracks loaded.")
        return tracks

    raise FileNotFoundError(
        f"No .stem.mp4 files or mixture.wav subdirectories found in {split_dir}."
    )


def _load_all_tracks(
    musdb_root: str,
    max_tracks: int = None,
) -> List[Tuple[str, np.ndarray, int]]:
    """
    Load mixture audio from ALL MUSDB18 tracks (train/ + test/ combined).

    This gives 150 tracks for the full MUSDB18 dataset (100 train + 50 test).
    Loading from both splits before any train/val split avoids the bias that
    would arise from training only on the 'train' split labels.
    """
    print(f"\nLoading all MUSDB18 tracks (train + test)...")
    all_tracks: List[Tuple[str, np.ndarray, int]] = []

    for split in ("train", "test"):
        split_dir = os.path.join(musdb_root, split)
        if not os.path.isdir(split_dir):
            print(f"  [warn] {split}/ not found in {musdb_root} — skipping")
            continue
        print(f"\n  Loading '{split}' split:")
        try:
            split_tracks = _load_musdb18(musdb_root, split=split)
            all_tracks.extend(split_tracks)
            print(f"  → {len(split_tracks)} tracks from '{split}'")
        except Exception as exc:
            print(f"  [warn] Could not load '{split}' split: {exc}")

    if max_tracks:
        all_tracks = all_tracks[:max_tracks]

    print(f"\n  Total tracks loaded: {len(all_tracks)}")
    return all_tracks


def _split_tracks(
    tracks: List[Tuple[str, np.ndarray, int]],
    val_frac: float = 0.2,
    random_state: int = 42,
) -> Tuple[List, List]:
    """
    Split a list of tracks into train and validation subsets.

    The split is performed at the TRACK level before any segmentation, so
    no track appears in both partitions — this eliminates the data leakage
    that would occur from splitting overlapping segments.
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(tracks))
    n_val = max(1, round(len(tracks) * val_frac))
    val_set = set(indices[:n_val].tolist())
    train_tracks = [t for i, t in enumerate(tracks) if i not in val_set]
    val_tracks   = [t for i, t in enumerate(tracks) if i in val_set]
    return train_tracks, val_tracks


def _segment_track(
    y: np.ndarray, sr: int,
    segment_dur: float = 30.0,
    hop_dur: float = 15.0,
) -> List[np.ndarray]:
    seg_len = int(segment_dur * sr)
    hop_len = int(hop_dur * sr)
    segments = [
        y[start: start + seg_len]
        for start in range(0, len(y) - seg_len + 1, hop_len)
    ]
    return segments if segments else [y]


def _tracks_to_xy(
    tracks: List[Tuple[str, np.ndarray, int]],
    segment_dur: float = 30.0,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Segment all tracks and extract (X, Y) arrays for model training/evaluation.

    Returns
    -------
    X          : (n_segments, N_FEATURES)
    Y          : (n_segments, N_TARGETS)
    n_segments : total number of segments extracted
    n_failed   : segments that failed feature extraction
    """
    if label:
        print(f"\n  Extracting features from {label} tracks...")

    X_rows: List[np.ndarray] = []
    Y_rows: List[np.ndarray] = []
    n_failed = 0

    for track_name, y, sr in tracks:
        segments = _segment_track(y, sr, segment_dur=segment_dur)
        print(f"    {track_name}: {len(segments)} segments")

        for seg_idx, seg in enumerate(segments):
            try:
                x_vec = extract_compression_features(seg, sr)
                y_vec = derive_compression_targets(seg, sr)
                X_rows.append(x_vec)
                Y_rows.append(y_vec)
            except Exception as exc:
                print(f"      [err] {track_name} seg{seg_idx}: {exc}")
                n_failed += 1

    n_segments = len(X_rows)
    X = np.stack(X_rows).astype(np.float32) if X_rows else np.empty((0, N_FEATURES), dtype=np.float32)
    Y = np.stack(Y_rows).astype(np.float32) if Y_rows else np.empty((0, N_TARGETS), dtype=np.float32)

    if label:
        print(f"  {label.capitalize()} segments extracted: {n_segments}"
              + (f"  ({n_failed} failed)" if n_failed else ""))
    return X, Y, n_segments, n_failed


def build_dataset(
    musdb_root: str,
    split: str = "train",
    segment_dur: float = 30.0,
    max_tracks: int = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build (X, Y) matrices from a single MUSDB18 split.

    Used only by demo mode. The full training pipeline uses _load_all_tracks()
    + _split_tracks() + _tracks_to_xy() to avoid data leakage.
    """
    print(f"\nBuilding dataset from MUSDB18 '{split}' split...")
    tracks = _load_musdb18(musdb_root, split=split, max_tracks=max_tracks)

    X_rows, Y_rows, sample_ids = [], [], []
    failed = 0

    for track_name, y, sr in tracks:
        segments = _segment_track(y, sr, segment_dur=segment_dur)
        print(f"    {track_name}: {len(segments)} segments")

        for seg_idx, seg in enumerate(segments):
            try:
                x_vec = extract_compression_features(seg, sr)
                y_vec = derive_compression_targets(seg, sr)
                X_rows.append(x_vec)
                Y_rows.append(y_vec)
                sample_ids.append(f"{track_name}_seg{seg_idx:02d}")
            except Exception as exc:
                print(f"      [err] segment {seg_idx}: {exc}")
                failed += 1

    if not X_rows:
        raise ValueError("No samples extracted.")

    X = np.stack(X_rows).astype(np.float32)
    Y = np.stack(Y_rows).astype(np.float32)
    print(f"\nDataset ready: {X.shape[0]} samples | {X.shape[1]} features | {Y.shape[1]} targets")
    if failed:
        print(f"  ({failed} segments skipped due to errors)")
    return X, Y, sample_ids


# ── Model training ─────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> Tuple[Pipeline, dict]:
    """
    Train a StandardScaler → RandomForestRegressor pipeline on pre-split data.

    The train/val split must be performed at the TRACK level before calling
    this function (see _split_tracks), so segments from the same track cannot
    appear in both partitions.

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    metrics  : dict with overall_r2, per_target_r2, mse
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    print(f"\nTraining RandomForestRegressor...")
    print(f"  Train samples : {X_train.shape[0]}")
    print(f"  Val samples   : {X_val.shape[0]}")
    print(f"  Features      : {X_train.shape[1]}")
    print(f"  Targets       : {', '.join(TARGET_NAMES)}")

    pipeline.fit(X_train, Y_train)

    Y_pred = pipeline.predict(X_val)
    mse    = mean_squared_error(Y_val, Y_pred)
    r2     = r2_score(Y_val, Y_pred, multioutput="uniform_average")

    per_target_r2 = {
        name: float(r2_score(Y_val[:, i], Y_pred[:, i]))
        for i, name in enumerate(TARGET_NAMES)
    }

    print(f"\nValidation results:")
    print(f"  Overall MSE : {mse:.4f}")
    print(f"  Overall R²  : {r2:.4f}   (1.0 = perfect, 0.0 = predicts mean)")
    print(f"\n  Per-target R²:")
    for name, target_r2 in per_target_r2.items():
        bar      = "#" * max(0, int((target_r2 + 0.1) * 20))
        mean_val = float(Y_val[:, TARGET_NAMES.index(name)].mean())
        print(f"    {name:15s}  {target_r2:+.3f}  {bar}  (mean val: {mean_val:.2f})")

    rf = pipeline.named_steps["rf"]
    importances = rf.feature_importances_
    top10 = np.argsort(importances)[::-1][:10]
    print(f"\n  Top-10 most informative features:")
    for rank, idx in enumerate(top10, 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:28s}  {importances[idx]:.4f}")

    metrics = {
        "overall_r2":    float(r2),
        "mse":           float(mse),
        "per_target_r2": per_target_r2,
    }
    return pipeline, metrics


# ── Model persistence ──────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, output_path: str) -> None:
    bundle = {
        "pipeline":      pipeline,
        "target_names":  TARGET_NAMES,
        "feature_names": FEATURE_NAMES,
        "n_features":    N_FEATURES,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nModel saved -> {output_path}  ({size_kb:.0f} KB)")


# ── Training report ────────────────────────────────────────────────────────────

def _write_compression_report(
    report_path: str,
    n_total_tracks: int,
    n_train_tracks: int,
    n_val_tracks: int,
    n_train_segs: int,
    n_val_segs: int,
    metrics: dict,
    mode: str = "a",
) -> None:
    """
    Write (or append) the compression model training report.

    Parameters
    ----------
    mode : 'a' to append after EQ report, 'w' to overwrite
    """
    lines = [
        "=" * 60,
        "MMMA Training Report — Compression Model",
        "=" * 60,
        "",
        f"Dataset",
        f"  MUSDB18 tracks (train + test)  : {n_total_tracks}",
        f"  Train tracks (80%)             : {n_train_tracks}",
        f"  Validation tracks (20%)        : {n_val_tracks}",
        f"  Train segments                 : {n_train_segs}",
        f"  Validation segments            : {n_val_segs}",
        f"  Segment duration               : 30 s, 50% overlap",
        f"  Split strategy                 : track-level (no leakage)",
        f"  random_state                   : 42",
        "",
        f"Results",
        f"  Overall R²  : {metrics['overall_r2']:.4f}",
        f"  Overall MSE : {metrics['mse']:.4f}",
        "",
        f"  Per-target R²:",
    ]
    for name, r2 in metrics["per_target_r2"].items():
        lines.append(f"    {name:15s}  {r2:+.4f}")
    lines += ["", ""]

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nTraining report saved -> {report_path}")


# ── Inference API ──────────────────────────────────────────────────────────────

def predict_compression_params(
    y: np.ndarray,
    sr: int,
    model_path: str = "models/compression_model.joblib",
) -> Dict[str, float]:
    """
    Predict optimal compression parameters for an audio signal.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Compression model not found at '{model_path}'. "
            "Run train_compression_model.py first."
        )

    bundle   = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    if len(y) / sr > 45:
        segments = _segment_track(y, sr, segment_dur=30.0, hop_dur=15.0)
        X_segs   = np.stack([extract_compression_features(seg, sr) for seg in segments])
        x_vec    = X_segs.mean(axis=0, keepdims=True)
    else:
        x_vec = extract_compression_features(y, sr).reshape(1, -1)

    Y_pred = pipeline.predict(x_vec)[0]

    return {
        "ratio":        float(np.clip(Y_pred[0], 1.0, 16.0)),
        "threshold_db": float(np.clip(Y_pred[1], -40.0, -6.0)),
        "attack_ms":    float(np.clip(Y_pred[2], 1.0, 100.0)),
        "release_ms":   float(np.clip(Y_pred[3], 50.0, 1000.0)),
    }


# ── Demo mode ──────────────────────────────────────────────────────────────────

def _run_demo() -> None:
    print("\n[DEMO MODE] Generating 40 synthetic clips of varying dynamics...")
    sr  = 22050
    dur = 10
    rng = np.random.default_rng(1)
    t   = np.linspace(0, dur, sr * dur)

    X_rows, Y_rows = [], []
    for i in range(40):
        n_tones = rng.integers(1, 6)
        freqs   = rng.uniform(80, 8000, n_tones)
        y = sum(rng.uniform(0.1, 1.0) * np.sin(2 * np.pi * f * t) for f in freqs)
        env = 1.0 + rng.uniform(0, 3) * (rng.random(len(t)) > 0.95).astype(float)
        y   = (y * env).astype(np.float32)
        y  /= max(float(np.max(np.abs(y))), 1e-8)

        X_rows.append(extract_compression_features(y, sr))
        Y_rows.append(derive_compression_targets(y, sr))

    X = np.stack(X_rows).astype(np.float32)
    Y = np.stack(Y_rows).astype(np.float32)
    print(f"Synthetic dataset: {X.shape}")

    # Segment-level split is fine for synthetic demo data
    n_val = max(1, int(len(X) * 0.2))
    pipeline, _ = train_model(X[n_val:], Y[n_val:], X[:n_val], Y[:n_val])
    save_model(pipeline, "models/compression_model_demo.joblib")

    test_y = rng.uniform(-1, 1, sr * 5).astype(np.float32)
    result = predict_compression_params(test_y, sr, model_path="models/compression_model_demo.joblib")
    print("\npredict_compression_params() output:")
    for k, v in result.items():
        print(f"  {k:15s}  {v:.3f}")
    print("\nDemo complete - pipeline is working correctly.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MMMA Compression reference model on MUSDB18",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("musdb_path", nargs="?", default=None,
                        help="Path to the MUSDB18 root directory")
    parser.add_argument("--output", "-o", default="models/compression_model.joblib")
    parser.add_argument("--max-tracks", type=int, default=None, metavar="N",
                        help="Limit to first N tracks total (useful for quick smoke tests)")
    parser.add_argument("--segment-dur", type=float, default=30.0, metavar="SECS")
    parser.add_argument("--report", default="models/training_report.txt",
                        help="Path to save (append) the training report")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.demo:
        _run_demo()
        sys.exit(0)

    if args.musdb_path is None:
        print("Error: provide the MUSDB18 path or pass --demo.")
        print("  python src/train_compression_model.py /path/to/musdb18")
        print("  python src/train_compression_model.py --demo")
        sys.exit(1)

    print("=" * 60)
    print("MMMA Compression Reference Model — Training")
    print("=" * 60)

    # Step 1: load ALL 150 tracks (train + test splits combined)
    all_tracks = _load_all_tracks(args.musdb_path, max_tracks=args.max_tracks)
    if not all_tracks:
        print("Error: no tracks loaded. Check the MUSDB18 path.")
        sys.exit(1)

    # Step 2: track-level 80/20 split (fixes data leakage from overlapping segments)
    train_tracks, val_tracks = _split_tracks(all_tracks, val_frac=0.2, random_state=42)
    print(f"\nTrack-level split  (random_state=42, val_frac=0.20):")
    print(f"  Train tracks : {len(train_tracks)}")
    print(f"  Val tracks   : {len(val_tracks)}")

    # Step 3: segment each partition independently
    X_train, Y_train, n_train_segs, _ = _tracks_to_xy(
        train_tracks, segment_dur=args.segment_dur, label="train"
    )
    X_val, Y_val, n_val_segs, _ = _tracks_to_xy(
        val_tracks, segment_dur=args.segment_dur, label="val"
    )

    print(f"\nSegment counts:")
    print(f"  Train : {n_train_segs}")
    print(f"  Val   : {n_val_segs}")

    if X_train.shape[0] == 0:
        print("Error: no training samples extracted.")
        sys.exit(1)

    # Step 4: train on train segments, evaluate on val segments
    pipeline, metrics = train_model(X_train, Y_train, X_val, Y_val)

    # Step 5: save model and report
    save_model(pipeline, args.output)

    # Append compression section to the report (EQ script writes it first with mode='w')
    report_mode = "a" if os.path.exists(args.report) else "w"
    _write_compression_report(
        report_path=args.report,
        n_total_tracks=len(all_tracks),
        n_train_tracks=len(train_tracks),
        n_val_tracks=len(val_tracks),
        n_train_segs=n_train_segs,
        n_val_segs=n_val_segs,
        metrics=metrics,
        mode=report_mode,
    )

    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"  1. In compression_model.py, call predict_compression_params(y, sr)")
    print(f"     instead of the rule-based get_compression_recommendations().")
    print(f"  2. Import: from train_compression_model import predict_compression_params")
    print(f"  3. Model bundle: {args.output}")
    print(f"  4. Training report: {args.report}")
    print("=" * 60)

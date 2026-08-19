"""
train_eq_model.py — MMMA EQ Reference Model Training
======================================================

Trains a RandomForestRegressor on professionally mixed audio from the MUSDB18
dataset to learn what a balanced mix looks like in terms of frequency band
energy distribution across 7 bands (Sub-bass → Air).

The trained model replaces the hardcoded IDEAL_ENERGY dict in eq_model.py.
At inference time, predict_ideal_eq(y, sr) returns per-band energy targets
that are data-driven rather than hand-picked.

Usage
-----
    python src/train_eq_model.py /path/to/musdb18-hq

Optional flags:
    --output        models/eq_model.joblib   Save path
    --max-tracks    N                        Limit tracks (useful for quick tests)
    --segment-dur   30                       Seconds per training segment
    --demo                                   Run on synthetic data (no MUSDB18 needed)

MUSDB18-HQ directory layout expected:
    musdb18-hq/
        train/
            Artist - Title/
                mixture.wav
        test/
            ...

Download: https://zenodo.org/record/3338373
"""

import os
import sys
import argparse
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

import re
import subprocess

# ── Band definitions (must stay identical to eq_model.py) ─────────────────────

FREQUENCY_BANDS: Dict[str, Tuple[int, int]] = {
    "Sub-bass":  (20,    60),
    "Bass":      (60,    250),
    "Low-mid":   (250,   500),
    "Mid":       (500,   2000),
    "Upper-mid": (2000,  6000),
    "Presence":  (6000,  12000),
    "Air":       (12000, 20000),
}
BAND_NAMES = list(FREQUENCY_BANDS.keys())

# ── Feature schema (fixed order — must match between train and predict) ────────
# 20 MFCC means + 20 MFCC stds + 8 spectral/temporal + 12 chroma = 60 features

FEATURE_NAMES: List[str] = (
    [f"mfcc_{i}_mean" for i in range(20)] +
    [f"mfcc_{i}_std"  for i in range(20)] +
    ["spectral_centroid_mean", "spectral_centroid_std",
     "spectral_bandwidth_mean", "spectral_rolloff_mean",
     "zcr_mean", "rms_mean", "rms_std", "tempo"] +
    [f"chroma_{i}_mean" for i in range(12)]
)
N_FEATURES = len(FEATURE_NAMES)  # 60


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_band_energies(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute normalised frequency band energies (0–1).

    Identical to the implementation in eq_model.py so that the training
    targets are in exactly the same space as the inference-time inputs.
    """
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0] * 2 - 2)

    band_energies: Dict[str, float] = {}
    for band, (lo, hi) in FREQUENCY_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_energies[band] = float(np.mean(S[mask, :])) if np.any(mask) else 0.0

    max_e = max(band_energies.values()) if band_energies else 1.0
    if max_e > 0:
        band_energies = {k: v / max_e for k, v in band_energies.items()}
    else:
        band_energies = {k: 0.0 for k in band_energies}

    return band_energies


def extract_audio_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract a 60-dimensional feature vector describing the audio.

    These features are the MODEL INPUTS (X). They characterise the musical
    content — timbre, harmony, rhythm — without including the band energies
    themselves (which are the targets Y).

    Returns
    -------
    np.ndarray of shape (60,)
    """
    vec: List[float] = []

    # ── MFCCs (40 features: 20 means + 20 stds) ──────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    vec.extend(float(np.mean(mfccs[i])) for i in range(20))
    vec.extend(float(np.std(mfccs[i]))  for i in range(20))

    # ── Spectral features (4 features) ───────────────────────────────────────
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    vec.append(float(np.mean(centroid)))
    vec.append(float(np.std(centroid)))
    vec.append(float(np.mean(bandwidth)))
    vec.append(float(np.mean(rolloff)))

    # ── Temporal / energy features (4 features) ───────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    vec.append(float(np.mean(zcr)))
    vec.append(float(np.mean(rms)))
    vec.append(float(np.std(rms)))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    vec.append(float(np.asarray(tempo).flat[0]))

    # ── Chroma (12 features: mean per pitch class) ────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    vec.extend(float(np.mean(chroma[i])) for i in range(12))

    assert len(vec) == N_FEATURES, f"Feature count mismatch: got {len(vec)}, expected {N_FEATURES}"
    return np.array(vec, dtype=np.float32)


# ── Dataset loading ────────────────────────────────────────────────────────────

def _get_ffmpeg() -> str:
    """Return path to ffmpeg executable (prefers imageio-ffmpeg bundle)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    from shutil import which
    exe = which("ffmpeg") or which("ffmpeg.exe")
    if exe:
        return exe
    raise RuntimeError(
        "ffmpeg not found. Install imageio-ffmpeg with:  pip install imageio-ffmpeg"
    )


def _decode_stem_mp4(path: str, ffmpeg_exe: str) -> Tuple[np.ndarray, int]:
    """
    Decode the mixture stream (0:a:0) from a MUSDB18 .stem.mp4 file.

    MUSDB18 stem MP4 layout:
      stream 0:a:0 = mixture (stereo, 44100 Hz)
      stream 0:a:1 = drums
      stream 0:a:2 = bass
      stream 0:a:3 = other
      stream 0:a:4 = vocals

    Uses raw PCM output piped to stdout — no temp files needed.
    """
    probe = subprocess.run(
        [ffmpeg_exe, "-i", path],
        capture_output=True, text=True, errors="replace",
    )
    sr_match = re.search(r"(\d+) Hz", probe.stderr)
    sr = int(sr_match.group(1)) if sr_match else 44100

    cmd = [
        ffmpeg_exe,
        "-loglevel", "error",
        "-i", path,
        "-map", "0:a:0",
        "-f", "f32le",
        "-ar", str(sr),
        "-ac", "2",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode(errors='replace')[:200]}")

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("ffmpeg returned empty audio data")

    stereo = audio.reshape(-1, 2)
    y = stereo.mean(axis=1)
    return y, sr


def _load_musdb18_wav(
    musdb_root: str,
    split: str = "train",
    max_tracks: int = None,
) -> List[Tuple[str, np.ndarray, int]]:
    """
    Load the mixture audio from each track in a MUSDB18 split.

    Supports both formats:
    - MUSDB18    : .stem.mp4 files (decoded directly via ffmpeg subprocess)
    - MUSDB18-HQ : Artist - Title/ folders containing mixture.wav

    Returns list of (track_name, y_mono, sr) tuples.
    Skips any track that fails to load and logs the error.
    """
    split_dir = os.path.join(musdb_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            "Make sure musdb_root points to the MUSDB18 root directory "
            "(the one containing 'train/' and 'test/' subdirectories)."
        )

    stem_mp4_files = sorted(
        f for f in os.listdir(split_dir) if f.endswith(".stem.mp4")
    )
    wav_dirs = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
        and os.path.exists(os.path.join(split_dir, d, "mixture.wav"))
    )

    if stem_mp4_files:
        ffmpeg_exe = _get_ffmpeg()
        print(f"    Detected .stem.mp4 format — decoding with ffmpeg")
        files_to_load = stem_mp4_files[:max_tracks] if max_tracks else stem_mp4_files
        tracks = []
        for fname in files_to_load:
            track_name = fname.replace(".stem.mp4", "")
            path = os.path.join(split_dir, fname)
            try:
                y, sr = _decode_stem_mp4(path, ffmpeg_exe)
                tracks.append((track_name, y, sr))
                print(f"    [ok]   {track_name}  ({len(y)/sr:.0f}s, {sr}Hz)")
            except Exception as exc:
                print(f"    [err]  {track_name} — {exc}")
        if not tracks:
            raise ValueError(
                f"No tracks successfully decoded from {split_dir}.\n"
                "Install imageio-ffmpeg:  pip install imageio-ffmpeg"
            )
        return tracks

    if wav_dirs:
        dirs_to_load = wav_dirs[:max_tracks] if max_tracks else wav_dirs
        tracks = []
        for track_name in dirs_to_load:
            mixture_path = os.path.join(split_dir, track_name, "mixture.wav")
            try:
                y, sr = librosa.load(mixture_path, sr=None, mono=True)
                tracks.append((track_name, y, sr))
                print(f"    [ok]   {track_name}  ({len(y)/sr:.0f}s, {sr}Hz)")
            except Exception as exc:
                print(f"    [err]  {track_name} — {exc}")
        if not tracks:
            raise ValueError(f"No tracks successfully loaded from {split_dir}.")
        return tracks

    raise FileNotFoundError(
        f"No .stem.mp4 files or mixture.wav subdirectories found in {split_dir}.\n"
        "Verify the MUSDB18 dataset was extracted correctly."
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

    Returns list of (track_name, y_mono, sr) tuples.
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
            split_tracks = _load_musdb18_wav(musdb_root, split=split)
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

    Parameters
    ----------
    tracks       : list of (name, y, sr) tuples
    val_frac     : fraction of tracks to hold out for validation (default 0.2)
    random_state : seed for reproducibility

    Returns
    -------
    (train_tracks, val_tracks)
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(tracks))
    n_val = max(1, round(len(tracks) * val_frac))
    val_set = set(indices[:n_val].tolist())
    train_tracks = [t for i, t in enumerate(tracks) if i not in val_set]
    val_tracks   = [t for i, t in enumerate(tracks) if i in val_set]
    return train_tracks, val_tracks


def _tracks_to_xy(
    tracks: List[Tuple[str, np.ndarray, int]],
    segment_dur: float = 30.0,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Segment all tracks and extract (X, Y) arrays for model training/evaluation.

    Parameters
    ----------
    tracks      : list of (name, y, sr) tuples
    segment_dur : length of each segment in seconds
    label       : print prefix for progress reporting ("train" or "val")

    Returns
    -------
    X          : (n_segments, N_FEATURES)
    Y          : (n_segments, n_bands)
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

        for seg_idx, segment in enumerate(segments):
            try:
                x_vec    = extract_audio_features(segment, sr)
                energies = extract_band_energies(segment, sr)
                y_vec    = np.array([energies[b] for b in BAND_NAMES], dtype=np.float32)
                X_rows.append(x_vec)
                Y_rows.append(y_vec)
            except Exception as exc:
                print(f"      [err] {track_name} seg{seg_idx}: {exc}")
                n_failed += 1

    n_segments = len(X_rows)
    X = np.stack(X_rows).astype(np.float32) if X_rows else np.empty((0, N_FEATURES), dtype=np.float32)
    Y = np.stack(Y_rows).astype(np.float32) if Y_rows else np.empty((0, len(BAND_NAMES)), dtype=np.float32)

    if label:
        print(f"  {label.capitalize()} segments extracted: {n_segments}"
              + (f"  ({n_failed} failed)" if n_failed else ""))
    return X, Y, n_segments, n_failed


def _segment_track(
    y: np.ndarray,
    sr: int,
    segment_dur: float = 30.0,
    hop_dur: float = 15.0,
) -> List[np.ndarray]:
    """
    Split a long track into overlapping fixed-length segments.

    Overlapping segments (default 50% overlap) multiply the number of
    training samples without loading extra tracks.
    """
    seg_len = int(segment_dur * sr)
    hop_len = int(hop_dur * sr)

    segments = [
        y[start: start + seg_len]
        for start in range(0, len(y) - seg_len + 1, hop_len)
    ]
    if not segments:
        segments = [y]
    return segments


def build_dataset(
    musdb_root: str,
    split: str = "train",
    segment_dur: float = 30.0,
    max_tracks: int = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build the (X, Y) matrices from a single MUSDB18 split.

    Used only by demo mode. The full training pipeline uses _load_all_tracks()
    + _split_tracks() + _tracks_to_xy() to avoid data leakage.
    """
    print(f"\nBuilding dataset from MUSDB18 '{split}' split...")
    tracks = _load_musdb18_wav(musdb_root, split=split, max_tracks=max_tracks)

    X_rows, Y_rows, sample_ids = [], [], []
    failed = 0

    for track_name, y, sr in tracks:
        segments = _segment_track(y, sr, segment_dur=segment_dur)
        print(f"    {track_name}: {len(segments)} segments")

        for seg_idx, segment in enumerate(segments):
            try:
                x_vec    = extract_audio_features(segment, sr)
                energies = extract_band_energies(segment, sr)
                y_vec    = np.array([energies[b] for b in BAND_NAMES], dtype=np.float32)
                X_rows.append(x_vec)
                Y_rows.append(y_vec)
                sample_ids.append(f"{track_name}_seg{seg_idx:02d}")
            except Exception as exc:
                print(f"      [err] segment {seg_idx}: {exc}")
                failed += 1

    if not X_rows:
        raise ValueError("No samples extracted. Check librosa installation and audio files.")

    X = np.stack(X_rows).astype(np.float32)
    Y = np.stack(Y_rows).astype(np.float32)
    print(f"\nDataset ready: {X.shape[0]} samples  |  {X.shape[1]} features  |  {Y.shape[1]} targets")
    if failed:
        print(f"  ({failed} segments failed and were skipped)")
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

    Parameters
    ----------
    X_train, Y_train : training features and targets
    X_val,   Y_val   : validation features and targets (held-out tracks only)

    Returns
    -------
    pipeline : fitted sklearn Pipeline
    metrics  : dict with overall_r2, per_band_r2, mse
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
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
    print(f"  Targets       : {Y_train.shape[1]} (one per EQ band)")

    pipeline.fit(X_train, Y_train)

    # ── Validation metrics ────────────────────────────────────────────────────
    Y_pred = pipeline.predict(X_val)
    mse    = mean_squared_error(Y_val, Y_pred)
    r2     = r2_score(Y_val, Y_pred, multioutput="uniform_average")

    per_band_r2 = {
        band: float(r2_score(Y_val[:, i], Y_pred[:, i]))
        for i, band in enumerate(BAND_NAMES)
    }

    print(f"\nValidation results:")
    print(f"  Overall MSE : {mse:.4f}")
    print(f"  Overall R²  : {r2:.4f}   (1.0 = perfect, 0.0 = predicts mean)")
    print(f"\n  Per-band R²:")
    for band, band_r2 in per_band_r2.items():
        bar = "#" * max(0, int((band_r2 + 0.1) * 20))
        print(f"    {band:12s}  {band_r2:+.3f}  {bar}")

    # ── Feature importances (top 10) ──────────────────────────────────────────
    rf = pipeline.named_steps["rf"]
    importances = rf.feature_importances_
    top10_idx = np.argsort(importances)[::-1][:10]
    print(f"\n  Top-10 most informative features:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:30s}  {importances[idx]:.4f}")

    metrics = {
        "overall_r2":   float(r2),
        "mse":          float(mse),
        "per_band_r2":  per_band_r2,
    }
    return pipeline, metrics


# ── Model persistence ──────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, output_path: str) -> None:
    """
    Save the fitted pipeline + metadata in a single joblib bundle.

    The bundle stores band_names and feature_names alongside the pipeline
    so that inference code can verify schema compatibility.
    """
    bundle = {
        "pipeline":      pipeline,
        "band_names":    BAND_NAMES,
        "feature_names": FEATURE_NAMES,
        "n_features":    N_FEATURES,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(bundle, output_path, compress=3)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nModel saved -> {output_path}  ({size_kb:.0f} KB)")


# ── Training report ────────────────────────────────────────────────────────────

def _write_eq_report(
    report_path: str,
    n_total_tracks: int,
    n_train_tracks: int,
    n_val_tracks: int,
    n_train_segs: int,
    n_val_segs: int,
    metrics: dict,
    mode: str = "w",
) -> None:
    """
    Write (or append) the EQ model training report to a text file.

    Parameters
    ----------
    report_path    : path to training_report.txt
    mode           : 'w' to overwrite, 'a' to append (use 'a' if compression
                     model report has already been written)
    """
    lines = [
        "=" * 60,
        "MMMA Training Report — EQ Model",
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
        f"  Per-band R²:",
    ]
    for band, r2 in metrics["per_band_r2"].items():
        lines.append(f"    {band:12s}  {r2:+.4f}")
    lines += ["", ""]

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nTraining report saved -> {report_path}")


# ── Inference API ──────────────────────────────────────────────────────────────

def predict_ideal_eq(
    y: np.ndarray,
    sr: int,
    model_path: str = "models/eq_model.joblib",
) -> Dict[str, float]:
    """
    Predict ideal frequency band energies for an audio signal using the
    trained model, replacing the hardcoded IDEAL_ENERGY dict in eq_model.py.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run train_eq_model.py first to generate it."
        )

    bundle     = joblib.load(model_path)
    pipeline   = bundle["pipeline"]
    band_names = bundle["band_names"]

    if len(y) / sr > 45:
        segments = _segment_track(y, sr, segment_dur=30.0, hop_dur=15.0)
        X_segs   = np.stack([extract_audio_features(seg, sr) for seg in segments])
        x_vec    = X_segs.mean(axis=0, keepdims=True)
    else:
        x_vec = extract_audio_features(y, sr).reshape(1, -1)

    Y_pred = pipeline.predict(x_vec)[0]
    Y_pred = np.clip(Y_pred, 0.0, 1.0)

    return {band: float(val) for band, val in zip(band_names, Y_pred)}


# ── Demo mode (no MUSDB18 needed — validates the pipeline end-to-end) ─────────

def _run_demo() -> None:
    """
    Generate synthetic training data (sine waves + noise) to verify the full
    pipeline runs without the MUSDB18 dataset.
    """
    print("\n[DEMO MODE] Generating 40 synthetic audio clips (no MUSDB18 needed)...")
    sr = 22050
    duration = 10
    rng = np.random.default_rng(0)

    X_rows, Y_rows = [], []
    for i in range(40):
        t = np.linspace(0, duration, sr * duration)
        weights = rng.uniform(0.1, 1.0, 7)
        y = np.zeros(sr * duration, dtype=np.float32)
        nyq = sr / 2 - 10
        for w, (lo, hi) in zip(weights, FREQUENCY_BANDS.values()):
            if lo >= nyq:
                continue
            freq = rng.uniform(lo, min(hi, nyq))
            y   += w * np.sin(2 * np.pi * freq * t).astype(np.float32)
        y /= np.max(np.abs(y) + 1e-8)

        X_rows.append(extract_audio_features(y, sr))
        energies = extract_band_energies(y, sr)
        Y_rows.append([energies[b] for b in BAND_NAMES])

    X = np.stack(X_rows).astype(np.float32)
    Y = np.stack(Y_rows).astype(np.float32)
    print(f"Synthetic dataset: {X.shape}")

    # Segment-level split is fine for synthetic demo data
    n_val = max(1, int(len(X) * 0.2))
    pipeline, _ = train_model(X[n_val:], Y[n_val:], X[:n_val], Y[:n_val])
    save_model(pipeline, "models/eq_model_demo.joblib")

    test_y = rng.uniform(-1, 1, sr * 5).astype(np.float32)
    result = predict_ideal_eq(test_y, sr, model_path="models/eq_model_demo.joblib")
    print("\npredict_ideal_eq() output:")
    for band, val in result.items():
        print(f"  {band:12s}  {val:.3f}")
    print("\nDemo complete - pipeline is working correctly.")


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MMMA EQ reference model on MUSDB18",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "musdb_path",
        nargs="?",
        default=None,
        help="Path to the MUSDB18-HQ root directory (contains train/ and test/)",
    )
    parser.add_argument(
        "--output", "-o",
        default="models/eq_model.joblib",
        help="Output path for the saved model bundle",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N tracks total (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--segment-dur",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Length of each training segment in seconds",
    )
    parser.add_argument(
        "--report",
        default="models/training_report.txt",
        help="Path to save the training report",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on synthetic data to validate the pipeline (no MUSDB18 needed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.demo:
        _run_demo()
        sys.exit(0)

    if args.musdb_path is None:
        print("Error: provide the MUSDB18 path or pass --demo to use synthetic data.")
        print("  python src/train_eq_model.py /path/to/musdb18-hq")
        print("  python src/train_eq_model.py --demo")
        sys.exit(1)

    # ── Full training run ──────────────────────────────────────────────────────
    print("=" * 60)
    print("MMMA EQ Reference Model — Training")
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

    _write_eq_report(
        report_path=args.report,
        n_total_tracks=len(all_tracks),
        n_train_tracks=len(train_tracks),
        n_val_tracks=len(val_tracks),
        n_train_segs=n_train_segs,
        n_val_segs=n_val_segs,
        metrics=metrics,
        mode="w",   # overwrite — run compression script second to append its section
    )

    print("\n" + "=" * 60)
    print("Next steps:")
    print(f"  1. In eq_model.py, call predict_ideal_eq(y, sr) instead of")
    print(f"     the hardcoded IDEAL_ENERGY dict.")
    print(f"  2. Import: from train_eq_model import predict_ideal_eq")
    print(f"  3. Model bundle: {args.output}")
    print(f"  4. Training report: {args.report}")
    print("=" * 60)

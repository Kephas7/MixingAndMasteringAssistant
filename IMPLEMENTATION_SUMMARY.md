# MMMA — Implementation Summary

Factual account of what the codebase actually does, compiled by reading every
source file in the repository (`app.py`, all of `src/`, `requirements.txt`,
`.gitignore`, and the contents of `data/`, `models/`, `tests/`, `notebooks/`).
Nothing below is inferred from naming or intent — where the code doesn't show
something, it's marked "not found in code."

---

## 1. Project structure

Top two levels (excluding `venv/`, `__pycache__/`, `.git/`):

```
MixingAndMasteringAssistant/
├── app.py                     Streamlit UI — the only application entry point
├── requirements.txt           Pinned dependencies (UTF-16 encoded, see §9)
├── data/
│   ├── raw/musdb18/           MUSDB18 dataset (train/ + test/, gitignored)
│   └── processed/             Empty except for a cache file this evaluation run produced
├── models/
│   ├── eq_model.joblib        Trained EQ RandomForest bundle
│   ├── compression_model.joblib   Trained compression RandomForest bundle
│   └── eq_model_demo.joblib   Output of --demo synthetic-data run (see §5)
├── notebooks/                 Empty
├── tests/                     Empty — no automated tests exist
├── results/                   Output of src/evaluate_models.py (metrics + plots)
└── src/
    ├── audio_processor.py     Real-time-safe DSP: EQ/compressor/limiter + WAV export + preview plots
    ├── feature_extraction.py  Audio loading + basic info/spectral helpers
    ├── eq_model.py            EQ analysis: band energies, ML-or-fallback ideal targets, recommendations
    ├── compression_model.py   Dynamics analysis: metrics, ML-or-fallback params, recommendations
    ├── mastering_model.py     Loudness measurement (LUFS/dBTP/LRA) + platform comparison
    ├── recommender.py         Combines EQ/compression/mastering results into a score + priority list
    ├── stem_separator.py      Demucs-based stem separation, per-stem analysis, stem mixer
    ├── train_eq_model.py      CLI training script for the EQ model; also hosts predict_ideal_eq()
    ├── train_compression_model.py  CLI training script for the compression model; also hosts predict_compression_params()
    └── evaluate_models.py     Added this session — reproduces the held-out split and computes metrics; not used by the app at runtime
```

**Entry point.** `app.py` is a Streamlit script (`st.set_page_config(...)` at
module level, `main()` called unconditionally at the bottom). There is no
`README`, `Procfile`, `pyproject.toml`, or launch script anywhere in the repo,
so the run command is not documented in the codebase — it is inferred to be
`streamlit run app.py`, the only way a script built this way is normally
launched.

Two secondary entry points exist, both CLI scripts, neither wired to the app:
`python src/train_eq_model.py <musdb_path>` and
`python src/train_compression_model.py <musdb_path>` (each also accepts
`--demo` for a synthetic-data smoke test, and `--output`/`--max-tracks`/
`--segment-dur`/`--report` flags). `src/evaluate_models.py` is a third,
added in this session for thesis evaluation only.

---

## 2. Architecture and data flow

### Trace: uploaded file → final output

1. `app.py: main()` — `st.file_uploader(..., type=["wav","mp3","flac"])` → `uploaded_file.read()` → `file_bytes`.
2. `run_full_analysis(file_bytes)` (cached, `@st.cache_data`) runs once per unique upload:
   - `feature_extraction.load_audio(io.BytesIO(file_bytes))` → `librosa.load(sr=None)` → `(y, sr)`, mono.
   - `feature_extraction.get_basic_info(y, sr)`, `get_spectral_features(y, sr)` — duration/tempo/RMS, centroid/bandwidth/rolloff.
   - `eq_model.analyze_frequency_bands(y, sr)` — 7-band STFT energy.
   - `eq_model.get_ideal_energy(y, sr)` → tries `train_eq_model.predict_ideal_eq(y, sr)` (loads `models/eq_model.joblib`, extracts 60-dim feature vector, predicts, clips 0–1); falls back to the hardcoded `IDEAL_ENERGY` dict on any exception.
   - `eq_model.get_eq_recommendations(band_energies, ideal_energy)` — dB delta → Boost/Cut/Neutral per band.
   - `compression_model.analyze_dynamics(y, sr)` — rule-based peak/RMS/crest-factor metrics.
   - `compression_model.get_compression_params(y, sr)` → tries `train_compression_model.predict_compression_params(y, sr)` (loads `models/compression_model.joblib`); falls back to an analytic formula.
   - `compression_model.get_compression_recommendations(dynamics, ml_params)` — assessment label is always rule-based; ratio/attack/release come from `ml_params` when present.
   - `mastering_model.analyze_loudness(y, sr)` — resamples to 48 kHz, K-weights, gated LUFS/LRA, oversampled true peak.
   - `mastering_model.get_mastering_recommendations(loudness)`.
   - `recommender.generate_mix_report(eq_recs, comp_recs, master_recs)` — 0–100 score, grade, top-3 priorities.
   - `audio_processor.export_audio(y, sr)` — int16 PCM WAV bytes, stored as `orig_bytes`.
   - Everything is packed into one dict `R` and returned/cached.
3. `render_progress_indicator()` draws the 6-step navigation; `st.session_state["current_step"]` (1–6) selects which `render_stepN(R)` runs.
4. Step 2 (Stems) independently calls `run_stem_separation(file_bytes)` (also cached): re-decodes audio, calls `stem_separator.separate_stems` (Demucs), `analyze_stems` (re-runs EQ/dynamics analysis per stem), `get_stem_insights`.
5. Steps 3–5 (EQ, Dynamics, Loudness) each: pre-load slider defaults from the ML/rule-based recommendations on first visit, apply the corresponding `audio_processor.apply_eq` / `apply_compression` / `apply_limiter` live for a before/after `st.audio` preview, and on "Apply & Continue" bake the current slider values into new WAV bytes stored in `st.session_state` (`eq_audio` → `comp_audio` → `final_audio`), passed forward as the source for the next step.
6. Step 6 (Export) shows before/after audio, a processing-chain summary read back from `st.session_state`, the mix report's strengths/priorities, and a `st.download_button` for `final_audio`.

### Stages/layers as they exist in code

| Stage | File(s) |
|---|---|
| Ingestion | `feature_extraction.py` (`load_audio`) |
| Basic analysis / rule-based features | `feature_extraction.py`, `eq_model.py` (`analyze_frequency_bands`), `compression_model.py` (`analyze_dynamics`) |
| ML feature extraction + inference | `train_eq_model.py`, `train_compression_model.py` (imported lazily from `eq_model.py`/`compression_model.py`) |
| Source separation | `stem_separator.py` (Demucs) |
| Interactive DSP / preview | `audio_processor.py` |
| Loudness / mastering measurement | `mastering_model.py` |
| Recommendation synthesis | `recommender.py` |
| UI / session orchestration | `app.py` |

---

## 3. Audio ingestion and preprocessing

- **Formats accepted (UI level):** WAV, MP3, FLAC — `st.file_uploader(..., type=["wav","mp3","flac"])` in `app.py`. `librosa.load` underneath may support more, but the uploader restricts to these three.
- **Sample rate:** **not standardized.** `feature_extraction.load_audio()` calls `librosa.load(audio_buffer, sr=None)`, which preserves the file's native sample rate. Nothing downstream resamples the working `y` to a fixed rate. Two places resample *internally, for their own purposes only*, without changing the app's working audio:
  - `mastering_model.analyze_loudness()` resamples a local copy to 48 kHz because the K-weighting filter coefficients are only valid at 48 kHz.
  - `stem_separator.separate_stems()` resamples a local copy to `model.samplerate` (44100 for `htdemucs`) because Demucs requires it.
- **Mono/stereo:** `librosa.load`'s default `mono=True` is used (not overridden), so any stereo upload is downmixed to mono at the point of loading. The entire pipeline — EQ, compression, feature extraction, loudness measurement — operates on mono only. Demucs separation duplicates the mono signal into a fake stereo pair (`np.stack([y, y])`) because `htdemucs` expects stereo input, then the four output stems are averaged back to mono (`stem_np.mean(axis=0)`) before use. **No stereo/spatial analysis exists anywhere in the app.**
- **Where:** `src/feature_extraction.py: load_audio()`, called from `app.py`'s `run_full_analysis()`, `run_stem_separation()`, and `decode_audio_bytes()` (re-hydrates intermediate WAV bytes stored in `st.session_state` between wizard steps).

---

## 4. Feature extraction

### EQ model — `train_eq_model.py: extract_audio_features()` — **60 features, confirmed**

```python
FEATURE_NAMES = (
    [f"mfcc_{i}_mean" for i in range(20)] +
    [f"mfcc_{i}_std"  for i in range(20)] +
    ["spectral_centroid_mean", "spectral_centroid_std",
     "spectral_bandwidth_mean", "spectral_rolloff_mean",
     "zcr_mean", "rms_mean", "rms_std", "tempo"] +
    [f"chroma_{i}_mean" for i in range(12)]
)   # 20 + 20 + 8 + 12 = 60
```

Computed with **librosa**: `librosa.feature.mfcc(y, sr, n_mfcc=20)`,
`spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`,
`zero_crossing_rate`, `rms`, `librosa.beat.beat_track` (tempo),
`chroma_stft` — all at librosa's default `n_fft=2048`, `hop_length=512`;
only `n_mfcc=20` is explicitly set. `sr` is whatever sample rate the input
segment carries (44100 Hz for MUSDB18 during training; native upload rate
at inference — see §3).

### Compression model — `train_compression_model.py: extract_compression_features()` — **20 features, confirmed**

```python
FEATURE_NAMES = [
    "peak_db", "rms_db", "dynamic_range", "crest_factor_db",
    "rms_frame_std", "rms_frame_min", "rms_frame_max",
    "rms_frame_p10", "rms_frame_p90", "loudness_range",
    "onset_rate", "onset_strength_mean", "onset_strength_std", "onset_strength_max",
    "spectral_flux_mean", "spectral_flux_std",
    "zcr_mean", "spectral_centroid_mean", "spectral_bandwidth_mean", "tempo",
]   # 20
```

Computed with **librosa**: `feature.rms(frame_length=2048, hop_length=512)`
for the frame-level statistics, `onset.onset_strength` +
`onset.onset_detect` for onset features, manual spectral flux
(`np.diff` of `librosa.stft(y, hop_length=512)` magnitude), `beat_track`
for tempo, plus centroid/bandwidth/ZCR at `hop_length=512`. Peak/RMS/dB
values are computed manually with `np.log10`.

**Verified inconsistency in this feature vector:** the `dynamic_range`
element is not `peak_db - rms_db` despite the inline comment claiming so.
The actual code is:

```python
vec.extend([peak_db, rms_db, crest_factor_db - (rms_db - peak_db), crest_factor_db])
# Note: dynamic_range = peak_db - rms_db = crest_factor_db (same thing for single peak)
```

Since `crest_factor_db = peak_db - rms_db`, the third term algebraically
reduces to `2 * crest_factor_db`, not `crest_factor_db`. The model was
trained on this doubled value under the name `dynamic_range` — it isn't
what the comment says it is, and it's redundant with the `crest_factor_db`
feature already in the vector (see §10).

Note: `compression_model.py`'s own `analyze_dynamics()` (used for the
app's rule-based assessment, not for the ML feature vector) computes
`dynamic_range` correctly as `peak_db - rms_db`, and uses non-overlapping
2048-sample frames (no explicit hop) rather than the ML feature
extractor's `hop_length=512` — the two "dynamic range"-flavoured
computations in this codebase are genuinely different code paths with
different framing parameters.

---

## 5. The machine learning models

Both models are `sklearn.pipeline.Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(...))])`.

### EQ model (`train_eq_model.py`)

- **Hyperparameters:** `RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)`
- **Targets:** 7 per-band normalized energies (Sub-bass, Bass, Low-mid, Mid, Upper-mid, Presence, Air), unitless 0–1. Produced by `extract_band_energies()`, which is the same STFT-mean/max-normalize formula as `eq_model.py: analyze_frequency_bands()` — so the targets are the *actual measured frequency balance of professionally mixed MUSDB18 tracks*, not hand-picked values.
- **Train/test split:** track-level. `_split_tracks()` permutes all 150 MUSDB18 tracks (100 from `train/` + 50 from `test/`, loaded together) with `np.random.default_rng(random_state=42)`, holds out `round(150*0.2) = 30` tracks for validation, keeps 120 for training. Segmentation (30 s window, 15 s hop → 50% overlap, via `_segment_track()`) happens *after* the split, so no segment from a held-out track leaks into training.

### Compression model (`train_compression_model.py`)

- **Hyperparameters:** `RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)`
- **Targets:** `ratio` (unitless, 1.0–16.0), `threshold_db` (dB, −40 to −6), `attack_ms` (ms, 1–100), `release_ms` (ms, 50–1000).
- **How targets are derived — important:** these are **not observed from real mixes**. `derive_compression_targets()` computes them analytically from each segment's own measured crest factor / RMS / onset rate:
  ```python
  ratio        = clip(1.0 + max(0, crest_factor_db - 4.0) * 0.55, 1.0, 16.0)
  threshold_db = clip(rms_db + 4.0, -40.0, -6.0)
  attack_ms    = clip(80.0 / (1.0 + onset_rate * 0.9), 1.0, 100.0)
  release_ms   = clip(attack_ms * 4.0, 50.0, 1000.0)
  ```
  The RandomForest is therefore learning a feature-rich approximation of this fixed hand-written formula (from the 20 input features to the formula's own output), not a mapping learned from human engineers' actual compressor settings.
- **Train/test split:** identical mechanism and parameters to the EQ model — same `_split_tracks()` code, `val_frac=0.2`, `random_state=42`, `segment_dur=30`, `hop_dur=15`.

### Saving / loading

`joblib.dump(bundle, path, compress=3)` where
`bundle = {"pipeline", "band_names" (EQ) / "target_names" (compression), "feature_names", "n_features"}`.
Loaded via `joblib.load(path)`. Default paths: `models/eq_model.joblib`,
`models/compression_model.joblib`. A `models/eq_model_demo.joblib` also
exists, produced by `train_eq_model.py --demo` (40 synthetic sine-wave
clips, no MUSDB18 needed) — this is not used by the running app. No
`models/compression_model_demo.joblib` is present, even though
`train_compression_model.py` has an equivalent `--demo` path that would
produce one — that demo run doesn't appear to have been executed (or its
output was deleted).

### Fallback if a model file is missing

- `eq_model.get_ideal_energy()` wraps the ML call in a bare `try/except Exception` and falls back to the hardcoded 7-value `IDEAL_ENERGY` dict.
- `compression_model.get_compression_params()` does the same, falling back to the *same analytic formula* as `derive_compression_targets()`, but using a **hardcoded placeholder `onset_rate = 2.0`** instead of measuring it — the fallback path comment explains this is "a neutral default without librosa onset detection."

---

## 6. Loudness measurement — `mastering_model.py`

Implements ITU-R BS.1770-4 / EBU R128 **directly in code**, not via an
external loudness library (no `pyloudnorm` or similar in `requirements.txt`
or imports).

- **K-weighting:** two cascaded biquad IIR filters (`_apply_k_weighting`) with hardcoded coefficients (`_K_B1/_K_A1`, `_K_B2/_K_A2`) explicitly documented as "valid at 48 kHz" — hence the mandatory resample to 48 kHz before use.
- **Integrated loudness (`_integrated_lufs`):** 400 ms blocks, 100 ms hop (75% overlap), mean-square per block → `-0.691 + 10*log10(ms)`. Two-stage gating: absolute gate discards blocks below −70 LUFS; relative gate then discards blocks more than 10 LU below the (already-absolute-gated) mean — matching the BS.1770-4 gating scheme.
- **Loudness Range (`_lra`):** 3 s sliding windows, 100 ms hop, same two-stage gating but with a −20 LU relative gate, then LRA = 95th percentile − 10th percentile of the gated window loudness values (EBU R128 definition).
- **True peak:** 4× oversampling via `scipy.signal.resample_poly(y_48k, 4, 1)` then peak magnitude in dB — a direct, simplified implementation of the BS.1770-4 Annex 2 true-peak method (4× oversample minimum).
- **Platform targets compared against** (`STREAMING_TARGETS`): Spotify −14 LUFS, Apple Music −16 LUFS, YouTube −14 LUFS, Tidal −14 LUFS, Amazon Music −14 LUFS. `TRUE_PEAK_LIMIT = -1.0` dBTP, `LRA_MIN = 6.0`, `LRA_MAX = 18.0` LU.

---

## 7. Source separation — `stem_separator.py`

- **Model/library:** Demucs (`demucs==4.0.1`), specifically the pretrained `htdemucs` model, loaded via `demucs.pretrained.get_model("htdemucs")`.
- **Invocation:** the Demucs **Python API** is called directly (`demucs.apply.apply_model`) rather than its CLI — the module's own docstring explains this bypasses torchaudio's file-loading path, which requires `torchcodec` on newer torch/torchaudio builds. Input is resampled to `model.samplerate` (44100) if needed, duplicated into a fake stereo pair, and peak-normalized to [-1, 1] before inference (`shifts=0, split=True, overlap=0.25`); the normalization is undone on the output.
- **Stems produced:** whichever names are in `model.sources` for `htdemucs` — the code iterates `for i, name in enumerate(model.sources)` rather than hardcoding a list, but the rest of the app (`_MIXER_STEMS`, `_STEM_COLORS`, `_STEM_TITLES` in `stem_separator.py` and `app.py`) hardcodes `drums, bass, vocals, other` as the expected four — this matches `htdemucs`'s actual sources but is not asserted/validated anywhere in code.
- **Caching:** `@st.cache_data(show_spinner=False)` on `app.py: run_stem_separation(file_bytes)`, keyed on the raw uploaded file bytes — Streamlit's own in-process/on-disk cache, not a custom persistence layer. No stems are ever written to `models/` or `data/`. `stem_separator.py` also defines `save_audio_to_temp()` (writes a numpy array to a temp WAV via `soundfile`), but grep across the whole repo shows it is **never called anywhere** — dead code.
- **Stem mixer:** implemented in `app.py: render_step_stems()` — per-stem mute/solo/volume controls (`st.session_state` keys `mute_*`/`solo_*`/`vol_*`), six presets from `stem_separator.get_preset_combinations()` (Full Mix, Drumless, Instrumental, Vocals Only, Bass + Drums, Other Only), combined via `stem_separator.combine_stems()` (mutes zero out a stem, solo overrides mutes, each active stem scaled by its volume multiplier, result peak-normalized to 0.95), with a live `st.audio` preview and a WAV download button.

---

## 8. The user interface

- **Framework:** Streamlit (`st.set_page_config(page_title="Mix Assistant", page_icon="🎚️", layout="wide")`).
- **Actual structure — a 6-step linear wizard, not tabs.** Navigation is driven by `st.session_state["current_step"]` (1–6) and rendered by `render_progress_indicator()` as six buttons (`STEP_LABELS = ["Overview","Stems","EQ & Tone","Dynamics","Loudness","Export"]`). Completed steps are clickable to jump back, which invalidates (sets to `None`) all downstream cached audio state (`eq_audio`/`comp_audio`/`final_audio`) so re-visiting a step forces the later steps to be redone. Genuine `st.tabs()` are used in exactly one place — the Waveforms/Frequency Spectra sub-view inside Step 2 — not for top-level navigation.
- **Screens a user moves through:** Overview (score/grade, health cards, top-3 priorities) → Stems (Demucs separation, per-stem EQ/dynamics, mixer) → EQ & Tone → Dynamics & Compression → Loudness & Mastering → Export & Summary.
- **How recommendations are presented and applied — this is a real interaction, not a static display.** On first visit to a step, `eq_suggested`/`comp_suggested` flags gate a one-time copy of the ML/rule-based recommendation values into the slider widgets' session-state keys. The user can then freely drag any slider away from that starting point; "↺ Flat"/"↺ Reset" zero everything; "💡 ML Suggest"/"💡 ML Values" reload the model's values on demand. Every step computes a live before/after preview by running the *current* slider values through `audio_processor.apply_eq` / `apply_compression` / `apply_limiter` and rendering two `st.audio` players (original vs. processed) with peak/RMS captions — so the user genuinely auditions the effect before committing. "Apply & Continue →" bakes the current settings into WAV bytes stored in `st.session_state` and becomes the input to the next step's live preview (EQ → Dynamics → Loudness are chained, each building on the previous step's applied output, not all working from the untouched original).
- **Caching strategy:** three functions carry `@st.cache_data(show_spinner=False)`:
  - `run_full_analysis(file_bytes)` — caches the entire initial analysis dict (`R`), keyed on the raw uploaded bytes.
  - `decode_audio_bytes(audio_bytes)` — caches the librosa decode of intermediate WAV bytes passed between steps, avoiding re-decoding on every Streamlit rerun (which happens on every widget interaction).
  - `run_stem_separation(file_bytes)` — caches the multi-minute Demucs pass, keyed on the raw uploaded bytes.

---

## 9. Technologies

From `requirements.txt` (the file is **UTF-16 encoded** with a BOM — an
unusual choice for a text file in a Python repo; it opens as garbled text
in tools that assume UTF-8/ASCII, including this read). Exact pinned
versions, cross-checked against what's actually imported in `app.py`/`src/`:

**Directly imported by project code:**
| Package | Version | Used for |
|---|---|---|
| streamlit | 1.57.0 | UI framework |
| numpy | 2.4.4 | array math throughout |
| librosa | 0.11.0 | audio loading, all spectral/rhythm features |
| scikit-learn | 1.8.0 | `StandardScaler`, `RandomForestRegressor`, `Pipeline` |
| joblib | 1.5.3 | model (de)serialization |
| scipy | 1.17.1 | biquad filters (`signal.lfilter`), resampling, WAV write |
| matplotlib | 3.10.9 | all plots |
| soundfile | 0.13.1 | WAV writing in `stem_separator.py` |
| demucs | 4.0.1 | stem separation (`htdemucs`) |
| torch | 2.11.0 | required by Demucs, imported lazily inside `stem_separator.py` (not at module top level) |

**Pinned in `requirements.txt` but not directly imported anywhere in `app.py` or `src/`:**
`tensorflow==2.21.0`, `keras==3.14.0`, `pandas==3.0.2`, `torchaudio==2.11.0`,
`openunmix==1.3.0` — these may be transitive dependencies of Streamlit or
Demucs (torchaudio in particular is a normal Demucs dependency), but grep
across the whole codebase finds no direct `import tensorflow`, `import
keras`, `import pandas`, `import torchaudio`, or `import openunmix`
anywhere in project code.

The installed venv was spot-checked and matches the pinned versions for
`sklearn`, `librosa`, `numpy`, and `matplotlib` exactly.

---

## 10. Honest gaps

Blunt, in order of how much it matters for a "what does this system
actually do" thesis chapter:

1. **The compression model's targets are a formula, not ground truth.** `derive_compression_targets()` is a hand-written analytic mapping from crest factor / RMS / onset rate to compressor settings. The RandomForest is trained to reproduce this formula from 20 audio features — it is *not* trained on real engineers' compressor choices. Describing it as "learning what compression professional mixes use" would be inaccurate; it's more accurately "learning a feature-rich approximation of a rule-based heuristic." The EQ model does **not** have this problem — its targets are directly measured from real professionally-mixed tracks.

2. **`dynamic_range` in the compression model's feature vector is mathematically wrong relative to its own comment.** `train_compression_model.py: extract_compression_features()` line ~118 computes `crest_factor_db - (rms_db - peak_db)`, which algebraically equals `2 * crest_factor_db`, not `crest_factor_db` as the adjacent comment states. The model was trained on this value under a misleading name; it's also redundant with the `crest_factor_db` feature already present. Not necessarily harmful to model performance (RF handles a scaled/duplicate feature fine) but the code and its comment disagree, and the feature name doesn't describe what it computes.

3. **Reproducing the reported R² required exact assumptions about training configuration that are not recorded anywhere.** No training report, log, or config file survives from the original training runs (`models/training_report.txt`, which both training scripts are coded to write, does not exist in the repo). I had to *assume* `segment_dur=30.0` (the CLI default) was used. This assumption reproduced the compression model's R² almost exactly (0.918 vs. reported 0.91) but the EQ model's reproduced R² (0.677) differs from the reported 0.72 by more than a rounding error — see the evaluation report in `results/model_metrics.txt` for the full discussion. If your thesis states 0.72 for the EQ model, that number cannot currently be independently verified from what's in the repo.

4. **No sample-rate standardization at inference.** `load_audio()` uses `sr=None`, so a user's upload is analyzed and fed to both ML models at whatever sample rate the file has natively — but both models were trained exclusively on 44100 Hz MUSDB18 audio. Nothing in the code resamples uploads to 44100 Hz before feature extraction. This is a real train/inference mismatch risk, not merely a style note.

5. **Everything is mono.** There is no stereo or spatial processing anywhere — confirmed at §3. If your thesis or any diagram implies stereo-aware EQ/compression/mastering, that is not what the code does.

6. **Dead code exists and should not be described as part of the working pipeline:**
   - `stem_separator.py: save_audio_to_temp()` — defined, never called.
   - `feature_extraction.py: get_mfcc()` and `plot_spectrogram()` — defined, never imported or called anywhere (only `load_audio`, `get_basic_info`, `plot_waveform`, `get_spectral_features` are actually used, per `app.py`'s import line).
   - `models/eq_model_demo.joblib` — output of a synthetic-data demo run, not used by the running app.
   - The equivalent `compression_model_demo.joblib` that `train_compression_model.py --demo` would produce does not exist in the repo — that demo path appears never to have been run (or its output was removed).

7. **No automated tests.** `tests/` exists but is completely empty. Nothing in the repo verifies feature-extraction correctness, the fallback paths, or model-loading behavior automatically.

8. **`requirements.txt` includes large ML frameworks (`tensorflow`, `keras`) that no project code imports.** Either they're unused leftovers from earlier prototyping, or they're pulled in for a reason not visible anywhere in `app.py`/`src/` (e.g. a notebook that no longer exists — `notebooks/` is empty). This should not be described as the project depending on TensorFlow/Keras for anything functional.

9. **The Demucs stem-name mapping is implicit, not enforced.** The app hardcodes `drums/bass/vocals/other` as the four stems everywhere in the UI, but `stem_separator.py` reads the actual names from `model.sources` at runtime rather than asserting they match. This works correctly for `htdemucs` in practice but is not defensively coded.

10. **`requirements.txt` is UTF-16 encoded**, which is atypical for this kind of file and caused it to render as spaced-out garbage text in a plain read — worth knowing if you regenerate or diff this file for your appendix.

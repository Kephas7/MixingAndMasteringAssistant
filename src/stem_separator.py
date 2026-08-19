"""
Stem Separation Module for MMMA (Music Mixing and Mastering Assistant)
Separates audio into drums, bass, vocals, and other stems using the Demucs
Python API directly (bypasses torchaudio file loading, which requires
torchcodec on newer torch/torchaudio builds).
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Optional


def save_audio_to_temp(y: np.ndarray, sr: int) -> str:
    """Save a numpy audio array to a temporary WAV file and return the path."""
    import os
    import tempfile
    import soundfile as sf
    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "input.wav")
    sf.write(wav_path, y.astype(np.float32), sr)
    return wav_path


def separate_stems(y: np.ndarray, sr: int, output_dir: str = None) -> Optional[dict]:
    """
    Separate audio into drums, bass, vocals, and other using the Demucs Python API.

    Uses demucs.pretrained.get_model + demucs.apply.apply_model so that audio
    is passed as a tensor — torchaudio file-loading (which requires torchcodec
    on newer installs) is bypassed entirely.

    Returns dict with keys matching model.sources (drums, bass, other, vocals)
    as mono float32 numpy arrays, or None if separation fails.
    """
    try:
        import torch
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as exc:
        print(f"[stem_separator] Import error: {exc}")
        return None

    try:
        model = get_model("htdemucs")
        model.eval()

        target_sr = model.samplerate  # 44100
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # htdemucs expects stereo — duplicate mono to stereo
        y_f32 = y.astype(np.float32)
        y_stereo = np.stack([y_f32, y_f32], axis=0)  # (2, samples)

        # Normalise to [-1, 1] so the model operates at a stable scale
        max_val = float(np.abs(y_stereo).max()) or 1.0
        y_norm  = y_stereo / max_val

        # (batch=1, channels=2, samples)
        wav = torch.from_numpy(y_norm).float().unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(
                model, wav,
                shifts=0,       # 0 = faster; 1 = better quality but doubles time
                split=True,
                overlap=0.25,
                progress=False,
            )
        # sources: (1, n_sources, 2, samples)

        stems: dict = {}
        for i, name in enumerate(model.sources):
            stem_np     = sources[0, i].numpy()          # (2, samples)
            stem_mono   = stem_np.mean(axis=0) * max_val  # rescale, convert to mono
            stems[name] = stem_mono.astype(np.float32)

        print(f"[stem_separator] Separation complete — {list(stems.keys())}")
        return stems

    except Exception as exc:
        import traceback
        print(f"[stem_separator] Error during separation: {exc}")
        traceback.print_exc()
        return None


def analyze_stems(stems: dict, sr: int) -> dict:
    """
    Run EQ and dynamics analysis on each stem.

    Returns nested dict:
        { stem_name: { "band_energies", "ideal_energy", "eq_recs", "dynamics" } }
    """
    from eq_model import analyze_frequency_bands, get_ideal_energy, get_eq_recommendations
    from compression_model import analyze_dynamics

    result: dict = {}
    for stem_name, y in stems.items():
        band_energies = analyze_frequency_bands(y, sr)
        ideal_energy  = get_ideal_energy(y, sr)
        eq_recs       = get_eq_recommendations(band_energies, ideal_energy=ideal_energy)
        dynamics      = analyze_dynamics(y, sr)
        result[stem_name] = {
            "band_energies": band_energies,
            "ideal_energy":  ideal_energy,
            "eq_recs":       eq_recs,
            "dynamics":      dynamics,
        }
    return result


# Plain-English consequence phrases keyed by (band, direction)
_CONSEQUENCES = {
    ("Sub-bass",  "Cut"):   "is likely contributing to sub-bass rumble that muddies the mix",
    ("Sub-bass",  "Boost"): "may explain why the {stem} lacks physical weight and impact",
    ("Bass",      "Cut"):   "is likely contributing to muddiness in your mix",
    ("Bass",      "Boost"): "may explain why the {stem} sounds thin or hollow",
    ("Low-mid",   "Cut"):   "is making the {stem} sound boxy or congested",
    ("Low-mid",   "Boost"): "may make the {stem} feel thin and lack body",
    ("Mid",       "Cut"):   "is adding a nasal or harsh quality to the {stem}",
    ("Mid",       "Boost"): "may explain why the {stem} sounds recessed in the mix",
    ("Upper-mid", "Cut"):   "is causing ear fatigue and harshness",
    ("Upper-mid", "Boost"): "may explain why the {stem} lacks definition and cut-through",
    ("Presence",  "Cut"):   "is causing sibilance or a harsh bright edge",
    ("Presence",  "Boost"): "may explain why the {stem} sounds dull or muffled",
    ("Air",       "Cut"):   "is making the {stem} sound brittle or overly bright",
    ("Air",       "Boost"): "may make the {stem} sound closed-in and lacking openness",
}


def get_stem_insights(stem_analyses: dict) -> list:
    """
    Identify the most severe EQ issue per stem and return plain-English insights.

    Only includes stems where the worst dB deviation > 2.0 dB.
    Returns at most 5 insights, ordered by severity (largest dB first).
    """
    insights_raw: list = []

    for stem_name, analysis in stem_analyses.items():
        eq_recs    = analysis["eq_recs"]
        worst_db   = 0.0
        worst_band = None
        worst_dir  = None

        for band, rec in eq_recs.items():
            if "Boost" in rec:
                db = float(rec.split(":")[1].strip().replace(" dB", ""))
                direction = "Boost"
            elif "Cut" in rec:
                db = float(rec.split(":")[1].strip().replace(" dB", ""))
                direction = "Cut"
            else:
                continue

            if db > worst_db:
                worst_db   = db
                worst_band = band
                worst_dir  = direction

        if worst_db <= 2.0 or worst_band is None:
            continue

        key        = (worst_band, worst_dir)
        consequence = _CONSEQUENCES.get(key, "needs attention")
        consequence = consequence.format(stem=f"{stem_name} stem")

        if worst_dir == "Cut":
            text = (
                f"Your {stem_name} stem has excessive energy in the "
                f"{worst_band} band ({worst_dir} {worst_db:.1f} dB recommended) "
                f"— {consequence}"
            )
        else:
            text = (
                f"Your {stem_name} stem needs more {worst_band} "
                f"({worst_dir} {worst_db:.1f} dB in {worst_band} band recommended) "
                f"— {consequence}"
            )

        insights_raw.append((worst_db, text))

    insights_raw.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in insights_raw[:5]]


# Per-stem display colours
_STEM_COLORS = {
    "drums":  "#E74C3C",
    "bass":   "#3498DB",
    "vocals": "#2ECC71",
    "other":  "#F39C12",
}
_STEM_TITLES = {
    "drums":  "Drums",
    "bass":   "Bass",
    "vocals": "Vocals",
    "other":  "Other",
}


def combine_stems(
    stems: dict,
    active_stems: list,
    volumes: dict,
) -> np.ndarray:
    """
    Combines selected stems into a single audio signal.

    Stems not in active_stems are muted. Each included stem is scaled by its
    volume multiplier (1.0 = 100 %, 0.5 = 50 %, 0.0 = silent). The result is
    normalised so the peak sits at 0.95, preventing clipping.
    """
    if not stems:
        return np.zeros(1, dtype=np.float32)

    ref_len = len(next(iter(stems.values())))

    if not active_stems:
        return np.zeros(ref_len, dtype=np.float32)

    combined = np.zeros(ref_len, dtype=np.float32)
    for name in active_stems:
        if name not in stems:
            continue
        vol = volumes.get(name, 1.0)
        arr = stems[name]
        n   = min(len(arr), ref_len)
        combined[:n] += arr[:n] * vol

    peak = float(np.abs(combined).max())
    if peak > 1e-6:
        combined = combined * (0.95 / peak)

    return combined.astype(np.float32)


def get_preset_combinations() -> dict:
    """Returns predefined stem combinations as presets."""
    return {
        "Full Mix":     ["drums", "bass", "vocals", "other"],
        "Drumless":     ["bass",  "vocals", "other"],
        "Instrumental": ["drums", "bass",   "other"],
        "Vocals Only":  ["vocals"],
        "Bass + Drums": ["drums", "bass"],
        "Other Only":   ["other"],
    }


def plot_stem_waveforms(stems: dict, sr: int) -> plt.Figure:
    """2×2 grid of waveforms — one subplot per stem."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")

    for ax, (stem_name, y) in zip(axes.flatten(), stems.items()):
        color = _STEM_COLORS.get(stem_name, "#888")
        ax.set_facecolor("#0e1117")
        librosa.display.waveshow(y, sr=sr, ax=ax, color=color, alpha=0.85)
        ax.set_title(_STEM_TITLES.get(stem_name, stem_name.capitalize()),
                     color="white", fontweight="bold", fontsize=11)
        ax.set_xlabel("Time (s)", color="#888", fontsize=8)
        ax.set_ylabel("Amplitude", color="#888", fontsize=8)
        ax.tick_params(colors="#888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#444")

    plt.tight_layout()
    return fig


def plot_stem_spectra(stems: dict, sr: int) -> plt.Figure:
    """2×2 grid of frequency spectra — one subplot per stem."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.patch.set_facecolor("#0e1117")

    for ax, (stem_name, y) in zip(axes.flatten(), stems.items()):
        color = _STEM_COLORS.get(stem_name, "#888")
        ax.set_facecolor("#0e1117")

        S      = np.abs(librosa.stft(y))
        freqs  = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0] * 2 - 2)
        S_db   = librosa.amplitude_to_db(S, ref=np.max)
        magnitude = np.mean(S_db, axis=1)

        mask = (freqs >= 20) & (freqs <= 20000)
        ax.plot(freqs[mask], magnitude[mask], color=color, linewidth=1.5, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlim([20, 20000])
        ax.set_title(_STEM_TITLES.get(stem_name, stem_name.capitalize()),
                     color="white", fontweight="bold", fontsize=11)
        ax.set_xlabel("Frequency (Hz)", color="#888", fontsize=8)
        ax.set_ylabel("Magnitude (dB)", color="#888", fontsize=8)
        ax.tick_params(colors="#888", labelsize=8)
        ax.grid(True, which="both", alpha=0.15, color="white")
        for sp in ax.spines.values():
            sp.set_color("#444")

    plt.tight_layout()
    return fig

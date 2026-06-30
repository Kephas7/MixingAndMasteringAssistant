"""
Interactive Audio Processor for MMMA.
Provides EQ, compression, and limiting that update in real-time as the user
adjusts sliders, using scipy biquad filters and vectorized processing.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, Tuple

# ── EQ band definitions ────────────────────────────────────────────────────────
# Each band has a center frequency, filter type, and Q factor.
# Sub-bass and Air use shelves; all others use peaking bells.
EQ_BANDS = {
    "Sub-bass":  {"freq": 40,    "type": "lowshelf",  "Q": 0.707},
    "Bass":      {"freq": 150,   "type": "peaking",   "Q": 1.0},
    "Low-mid":   {"freq": 375,   "type": "peaking",   "Q": 1.0},
    "Mid":       {"freq": 1000,  "type": "peaking",   "Q": 1.0},
    "Upper-mid": {"freq": 4000,  "type": "peaking",   "Q": 1.0},
    "Presence":  {"freq": 9000,  "type": "peaking",   "Q": 1.0},
    "Air":       {"freq": 16000, "type": "highshelf", "Q": 0.707},
}


# ── Biquad filter design (Audio EQ Cookbook — R. Bristow-Johnson) ──────────────

def _peaking(f0: float, gain_db: float, Q: float, sr: int):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * Q)
    b = np.array([1 + alpha*A,  -2*cw,  1 - alpha*A])
    a = np.array([1 + alpha/A,  -2*cw,  1 - alpha/A])
    return b / a[0], a / a[0]


def _low_shelf(f0: float, gain_db: float, sr: int):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    # S=1 (standard Butterworth shelf slope): alpha = sin(w0)/2 * sqrt(2)
    alpha = sw / 2 * np.sqrt(2)
    sqA = np.sqrt(A)
    b = np.array([
         A * ((A+1) - (A-1)*cw + 2*sqA*alpha),
        2*A * ((A-1) - (A+1)*cw),
         A * ((A+1) - (A-1)*cw - 2*sqA*alpha),
    ])
    a = np.array([
        (A+1) + (A-1)*cw + 2*sqA*alpha,
       -2 * ((A-1) + (A+1)*cw),
        (A+1) + (A-1)*cw - 2*sqA*alpha,
    ])
    return b / a[0], a / a[0]


def _high_shelf(f0: float, gain_db: float, sr: int):
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt(2)
    sqA = np.sqrt(A)
    b = np.array([
         A * ((A+1) + (A-1)*cw + 2*sqA*alpha),
        -2*A * ((A-1) + (A+1)*cw),
         A * ((A+1) + (A-1)*cw - 2*sqA*alpha),
    ])
    a = np.array([
        (A+1) - (A-1)*cw + 2*sqA*alpha,
        2 * ((A-1) - (A+1)*cw),
        (A+1) - (A-1)*cw - 2*sqA*alpha,
    ])
    return b / a[0], a / a[0]


def _design(band_name: str, gain_db: float, sr: int):
    band = EQ_BANDS[band_name]
    f0 = min(band["freq"], sr * 0.45)  # stay below Nyquist
    btype = band["type"]
    if btype == "peaking":
        return _peaking(f0, gain_db, band["Q"], sr)
    if btype == "lowshelf":
        return _low_shelf(f0, gain_db, sr)
    return _high_shelf(f0, gain_db, sr)


# ── Public processing functions ────────────────────────────────────────────────

def apply_eq(y: np.ndarray, sr: int, band_gains: Dict[str, float]) -> np.ndarray:
    """Apply 7-band parametric EQ. band_gains maps band name → gain in dB."""
    out = y.copy()
    for band, gain_db in band_gains.items():
        if abs(gain_db) < 0.05:
            continue
        b, a = _design(band, gain_db, sr)
        out = signal.lfilter(b, a, out)
    return out


def apply_compression(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 50.0,
    release_ms: float = 200.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """
    Dynamic range compressor using a vectorized peak-follower envelope.

    Uses an averaged attack/release smoothing coefficient — a good approximation
    that avoids a slow Python loop and is fast enough for real-time preview.
    """
    if ratio <= 1.05 and abs(makeup_db) < 0.05:
        return y.copy()

    makeup = 10 ** (makeup_db / 20.0)
    attack_coeff  = np.exp(-1.0 / max(attack_ms  * 0.001 * sr, 1))
    release_coeff = np.exp(-1.0 / max(release_ms * 0.001 * sr, 1))
    smooth = (attack_coeff + release_coeff) / 2

    # Fast envelope via scipy IIR (equivalent to exponential moving average)
    envelope = signal.lfilter([1 - smooth], [1, -smooth], np.abs(y))

    envelope_db = 20 * np.log10(np.maximum(envelope, 1e-10))
    gain_db = np.where(
        envelope_db > threshold_db,
        (threshold_db - envelope_db) * (1.0 - 1.0 / ratio),
        0.0,
    )
    return y * (10 ** (gain_db / 20.0)) * makeup


def apply_limiter(y: np.ndarray, ceiling_db: float = -0.1) -> np.ndarray:
    """Brick-wall limiter — hard-clips to ceiling_db."""
    ceiling = 10 ** (ceiling_db / 20.0)
    return np.clip(y, -ceiling, ceiling)


def export_audio(y: np.ndarray, sr: int) -> bytes:
    """Return WAV bytes suitable for st.audio and st.download_button."""
    buf = io.BytesIO()
    # Convert float32 to int16 PCM
    y_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    from scipy.io.wavfile import write as wav_write
    wav_write(buf, sr, y_int16)
    buf.seek(0)
    return buf.read()


# ── Visualization ──────────────────────────────────────────────────────────────

def compute_eq_curve(band_gains: Dict[str, float], sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Combined frequency response of all active EQ bands."""
    worN = np.logspace(np.log10(20), np.log10(min(sr / 2 - 100, 20000)), 800)
    H = np.ones(len(worN), dtype=complex)
    for band, gain_db in band_gains.items():
        if abs(gain_db) < 0.05:
            continue
        b, a = _design(band, gain_db, sr)
        _, h = signal.freqz(b, a, worN=worN, fs=sr)
        H *= h
    return worN, 20 * np.log10(np.abs(H) + 1e-10)


def plot_eq_curve(band_gains: Dict[str, float], sr: int) -> plt.Figure:
    """Live EQ curve — shows what the current EQ settings are doing to the frequency response."""
    freqs, magnitude = compute_eq_curve(band_gains, sr)

    fig, ax = plt.subplots(figsize=(11, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Zero (flat) reference
    ax.axhline(0, color="#555", linewidth=1, zorder=1)

    # Shade boost/cut regions
    ax.fill_between(freqs, 0, magnitude, where=magnitude >= 0, alpha=0.35, color="#FF8C42", zorder=2)
    ax.fill_between(freqs, 0, magnitude, where=magnitude <  0, alpha=0.35, color="#4D96FF", zorder=2)
    ax.plot(freqs, magnitude, color="white", linewidth=2, zorder=3)

    # Vertical band markers
    band_freqs  = [40, 150, 375, 1000, 4000, 9000, 16000]
    band_labels = ["Sub\nbass", "Bass", "Low\nMid", "Mid", "Upper\nMid", "Presence", "Air"]
    for f, label in zip(band_freqs, band_labels):
        ax.axvline(f, color="#444", linewidth=0.8, linestyle="--", zorder=0)

    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ax.set_ylim(-14, 14)
    ax.set_xlabel("Frequency (Hz)", color="#aaa", fontsize=9)
    ax.set_ylabel("Gain (dB)",      color="#aaa", fontsize=9)
    ax.set_title("EQ Curve — live preview", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.grid(alpha=0.15, color="white", which="both")

    # Add Hz labels at standard points
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"],
                       color="#aaa", fontsize=8)

    plt.tight_layout(pad=0.5)
    return fig


def plot_compressor_curve(threshold_db: float, ratio: float, makeup_db: float) -> plt.Figure:
    """
    Classic compressor transfer function (knee diagram).
    Shows input level vs output level, with the gain reduction region shaded.
    """
    x = np.linspace(-60, 0, 300)
    above = x > threshold_db

    # Compressed output
    y_comp = np.where(above, threshold_db + (x - threshold_db) / ratio, x)
    y_comp += makeup_db

    # Uncompressed (1:1) with makeup
    y_ref = x + makeup_db

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Shade the gain reduction area (only where compression is active)
    ax.fill_between(x, y_ref, y_comp, where=above, alpha=0.2, color="#FF6B6B", label="Gain reduction")

    # 1:1 reference
    ax.plot(x, y_ref, color="#555", linewidth=1.5, linestyle="--", label="No compression")

    # Compressed curve
    color = "#4D96FF" if ratio > 1.05 else "#6BCB77"
    ax.plot(x, y_comp, color=color, linewidth=2.5, label=f"{ratio:.1f}:1  +{makeup_db:.0f} dB makeup")

    # Threshold line
    ax.axvline(threshold_db, color="#FFD93D", linewidth=1.2, linestyle=":", alpha=0.9)
    ax.text(threshold_db + 1, -56, f"{threshold_db:.0f} dB",
            color="#FFD93D", fontsize=8, va="bottom")

    ax.set_xlim(-60, 0)
    ax.set_ylim(-60, min(max(makeup_db + 10, 10), 20))
    ax.set_xlabel("Input Level (dB)",  color="#aaa", fontsize=9)
    ax.set_ylabel("Output Level (dB)", color="#aaa", fontsize=9)
    ax.set_title("Compressor",         color="white", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", framealpha=0.8)
    ax.tick_params(colors="#aaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.grid(alpha=0.2, color="white")
    ax.plot([-60, 0], [-60, 0], color="#333", linewidth=0.5)  # diagonal guide

    plt.tight_layout(pad=0.5)
    return fig

"""
Mastering Analysis Model for MMMA (Music Mixing and Mastering Assistant)
Measures integrated loudness (LUFS), true peak (dBTP), and loudness range (LRA)
per ITU-R BS.1770-4 / EBU R128, then compares against streaming platform targets.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict

# Streaming platform loudness targets (integrated LUFS)
STREAMING_TARGETS = {
    "Spotify":       -14.0,
    "Apple Music":   -16.0,
    "YouTube":       -14.0,
    "Tidal":         -14.0,
    "Amazon Music":  -14.0,
}

TRUE_PEAK_LIMIT = -1.0   # dBTP ceiling for streaming delivery
LRA_MIN        = 6.0     # LU — below this the mix is likely over-compressed
LRA_MAX        = 18.0    # LU — above this is too dynamic for most commercial contexts

# ITU-R BS.1770-4 K-weighting filter coefficients (valid at 48 kHz)
_K_B1 = np.array([1.53512485958697, -2.69169618940638,  1.19839281085285])
_K_A1 = np.array([1.0,              -1.69065929318241,  0.73248077421585])
_K_B2 = np.array([1.0,              -2.0,               1.0])
_K_A2 = np.array([1.0,              -1.99004745483398,  0.99007225036603])


# ---------------------------------------------------------------------------
# Internal helpers (all expect 48 kHz mono input)
# ---------------------------------------------------------------------------

def _apply_k_weighting(y_48k: np.ndarray) -> np.ndarray:
    y = signal.lfilter(_K_B1, _K_A1, y_48k)
    y = signal.lfilter(_K_B2, _K_A2, y)
    return y


def _integrated_lufs(y_k: np.ndarray, sr: int = 48000) -> float:
    """Gated integrated loudness per ITU-R BS.1770-4."""
    block = int(0.4 * sr)   # 400 ms analysis window
    hop   = int(0.1 * sr)   # 100 ms hop (75 % overlap)

    loudness_blocks = []
    for i in range(0, len(y_k) - block + 1, hop):
        ms = np.mean(y_k[i : i + block] ** 2)
        if ms > 0:
            loudness_blocks.append(-0.691 + 10 * np.log10(ms))

    if not loudness_blocks:
        return -70.0

    blocks = np.array(loudness_blocks)

    # Absolute gate: discard blocks below -70 LUFS
    gated = blocks[blocks >= -70.0]
    if len(gated) == 0:
        return -70.0

    # Relative gate: discard blocks 10 LU below the ungated mean
    ungated_mean = -0.691 + 10 * np.log10(np.mean(10 ** (gated / 10)))
    gated = gated[gated >= ungated_mean - 10.0]
    if len(gated) == 0:
        return -70.0

    return round(-0.691 + 10 * np.log10(np.mean(10 ** (gated / 10))), 1)


def _lra(y_k: np.ndarray, sr: int = 48000) -> float:
    """Loudness Range per EBU R128 (3 s sliding window, -20 LU relative gate)."""
    window = int(3.0 * sr)
    hop    = int(0.1 * sr)

    windows = []
    for i in range(0, len(y_k) - window + 1, hop):
        ms = np.mean(y_k[i : i + window] ** 2)
        if ms > 0:
            windows.append(-0.691 + 10 * np.log10(ms))

    if len(windows) < 2:
        return 0.0

    wins  = np.array(windows)
    gated = wins[wins >= -70.0]
    if len(gated) < 2:
        return 0.0

    mean_lufs = -0.691 + 10 * np.log10(np.mean(10 ** (gated / 10)))
    gated     = gated[gated >= mean_lufs - 20.0]
    if len(gated) < 2:
        return 0.0

    return round(float(np.percentile(gated, 95) - np.percentile(gated, 10)), 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_loudness(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Measure integrated loudness (LUFS), true peak (dBTP), and loudness range (LRA).

    Resamples to 48 kHz internally (required for K-weighting coefficients).

    Parameters
    ----------
    y  : mono audio signal
    sr : original sample rate

    Returns
    -------
    Dict with keys: integrated_lufs, true_peak_dbtp, lra
    """
    y_48k = librosa.resample(y, orig_sr=sr, target_sr=48000) if sr != 48000 else y.copy()

    y_k        = _apply_k_weighting(y_48k)
    integrated = _integrated_lufs(y_k)
    lra        = _lra(y_k)

    # True peak: 4x oversample to detect inter-sample peaks
    y_up      = signal.resample_poly(y_48k, 4, 1)
    peak      = np.max(np.abs(y_up))
    true_peak = round(20 * np.log10(max(peak, 1e-10)), 1)

    return {
        "integrated_lufs":  integrated,
        "true_peak_dbtp":   true_peak,
        "lra":              lra,
    }


def get_mastering_recommendations(loudness: Dict[str, float]) -> Dict[str, str]:
    """
    Derive mastering recommendations from loudness metrics.

    Returns a dict containing both human-readable assessment strings and
    status keys (used to look up educational explanations).
    """
    lufs       = loudness["integrated_lufs"]
    true_peak  = loudness["true_peak_dbtp"]
    lra        = loudness["lra"]
    target     = -14.0
    diff       = lufs - target
    result: Dict[str, str] = {}

    # --- Integrated loudness ---
    if diff > 1.0:
        result["loudness_assessment"] = (
            f"Too loud by {abs(diff):.1f} LU — streaming platforms will normalize this down"
        )
        result["loudness_status"] = "too_loud"
    elif diff < -3.0:
        result["loudness_assessment"] = (
            f"Too quiet by {abs(diff):.1f} LU — will sound weak next to other tracks"
        )
        result["loudness_status"] = "too_quiet"
    else:
        result["loudness_assessment"] = "Loudness is well-targeted for streaming"
        result["loudness_status"]     = "good_loudness"

    result["gain_adjustment"] = f"{target - lufs:+.1f} dB to reach -14 LUFS"

    # --- True peak ---
    if true_peak > TRUE_PEAK_LIMIT:
        result["true_peak_assessment"] = (
            f"True peak {true_peak} dBTP exceeds -1.0 dBTP — may distort after encoding"
        )
        result["true_peak_status"] = "true_peak_exceeded"
    else:
        result["true_peak_assessment"] = f"True peak {true_peak} dBTP — within safe limits"
        result["true_peak_status"]     = "true_peak_ok"

    # --- LRA ---
    if lra < LRA_MIN:
        result["lra_assessment"] = f"LRA {lra} LU — very low, mix may sound flat and lifeless"
        result["lra_status"]     = "lra_low"
    elif lra > LRA_MAX:
        result["lra_assessment"] = f"LRA {lra} LU — high, volume will feel inconsistent in a playlist"
        result["lra_status"]     = "lra_high"
    else:
        result["lra_assessment"] = f"LRA {lra} LU — healthy dynamic range"
        result["lra_status"]     = "lra_good"

    return result


MASTERING_EXPLANATIONS: Dict[str, str] = {
    "too_loud": (
        "When your track exceeds -14 LUFS, Spotify, YouTube, and most platforms automatically "
        "turn it down using a limiter. This normalization often degrades the sound — an "
        "over-limited track loses punch and clarity when digitally attenuated. Focus on a "
        "great-sounding mix rather than chasing maximum loudness."
    ),
    "too_quiet": (
        "A track significantly below -14 LUFS will sound noticeably quieter than surrounding "
        "tracks in a playlist, even after normalization. Listeners often perceive quieter "
        "tracks as lower quality. Apply a limiter (ceiling -1.0 dBTP) and use make-up gain "
        "to raise the integrated loudness to the target range."
    ),
    "good_loudness": (
        "Your integrated loudness is well-targeted for streaming. The track should sit "
        "comfortably next to commercial releases without being turned up or down significantly "
        "by platform normalization algorithms."
    ),
    "true_peak_exceeded": (
        "True peak differs from regular peak metering — it measures inter-sample peaks that "
        "only appear after the digital signal is converted to analog or encoded to MP3/AAC. "
        "Exceeding -1.0 dBTP causes audible distortion on many devices. Set your limiter "
        "ceiling to -1.0 dBTP (not 0.0 dBFS) to prevent this."
    ),
    "true_peak_ok": (
        "True peak is within the standard -1.0 dBTP limit. Your track is safe from "
        "inter-sample clipping after lossy encoding to MP3 or AAC."
    ),
    "lra_low": (
        "A very low Loudness Range means the quiet and loud sections of your track are nearly "
        "the same volume — a sign of heavy limiting or over-compression. The mix may feel "
        "flat and fatiguing over time. Ease the limiter threshold and let the dynamics breathe."
    ),
    "lra_high": (
        "A high Loudness Range means there's significant variation between the quietest and "
        "loudest moments. While natural for classical or jazz, it feels inconsistent in a "
        "streaming playlist. A gentle mastering compressor (2:1, slow attack) before the "
        "limiter can tighten this up without squashing the life out of the mix."
    ),
    "lra_good": (
        "Your Loudness Range is in a healthy zone — the mix has enough dynamic variation to "
        "feel expressive and musical without being unpredictable in a streaming playlist context."
    ),
}


def get_mastering_explanation(status_key: str) -> str:
    """Return the educational explanation for a mastering status key."""
    return MASTERING_EXPLANATIONS.get(status_key, "")


def plot_loudness_vs_targets(loudness: Dict[str, float]) -> plt.Figure:
    """
    Horizontal loudness meter showing the track's integrated LUFS against
    streaming platform targets with color-coded acceptable zones.
    """
    lufs    = loudness["integrated_lufs"]
    x_min   = -30
    x_max   = -5

    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Zone shading
    ax.axvspan(x_min, -17,    alpha=0.2, color="#4D96FF", label="Too quiet")
    ax.axvspan(-17,   -11,    alpha=0.2, color="#6BCB77", label="Acceptable zone (-17 to -11 LUFS)")
    ax.axvspan(-11,   x_max,  alpha=0.2, color="#FF6B6B", label="Too loud")

    # Platform target lines
    platform_colors = ["#1DB954", "#FC3C44", "#FF2D55", "#00B9FF", "#FF9900"]
    for (platform, target), color in zip(STREAMING_TARGETS.items(), platform_colors):
        ax.axvline(
            target, color=color, linewidth=1.5, linestyle="--", alpha=0.9,
            label=f"{platform} ({target} LUFS)"
        )

    # Track loudness marker
    ax.axvline(lufs, color="white",     linewidth=5, zorder=5)
    ax.axvline(lufs, color="orangered", linewidth=3, zorder=6,
               label=f"Your track ({lufs} LUFS)")

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Integrated Loudness (LUFS)", fontsize=11, fontweight="bold", color="white")
    ax.set_title("Loudness vs. Streaming Platform Targets", fontsize=13, fontweight="bold", color="white")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8, ncol=2,
              labelcolor="white", facecolor="#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    return fig

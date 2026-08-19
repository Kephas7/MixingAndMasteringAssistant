"""
Compression Analysis Model for MMMA (Music Mixing and Mastering Assistant)
Analyzes dynamic range of audio signals and provides compression recommendations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Compression ratio presets (used for recommendations)
COMPRESSION_RATIOS = [1.5, 2.0, 4.0, 8.0, 16.0]

# Dynamic range thresholds for compression recommendations (in dB)
# These determine when compression is needed and how aggressive it should be
LOW_DYNAMIC_RANGE = 6      # Track is over-compressed (little dynamic variation)
MEDIUM_DYNAMIC_RANGE = 12  # Track has some dynamic range
HIGH_DYNAMIC_RANGE = 20    # Track has good dynamic range
EXTREME_DYNAMIC_RANGE = 30 # Track needs compression to control peaks

# Attack time options (in milliseconds)
ATTACK_TIMES = {
    "fast": 10,
    "medium": 50,
    "slow": 100,
}

# Release time options (in milliseconds)
RELEASE_TIMES = {
    "fast": 50,
    "medium": 200,
    "slow": 500,
}

# Crest factor thresholds (in dB)
# Crest factor = peak amplitude / RMS (ratio), converted to dB
CREST_FACTOR_LOW = 6       # Low crest factor: well-controlled dynamics
CREST_FACTOR_MEDIUM = 12   # Medium crest factor: moderate peaks
CREST_FACTOR_HIGH = 18     # High crest factor: significant peaks that may clip

# Educational explanations keyed to compression assessment outcomes
COMPRESSION_EXPLANATIONS = {
    "over_compressed": (
        "Your track has very little dynamic range — the loud and quiet parts are nearly the same "
        "volume. This is known as 'over-compression' and results in a flat, fatiguing sound that "
        "lacks emotional impact and punch. Adding more compression will make this worse. "
        "Instead, revisit your gain staging and use lighter compression on individual tracks, "
        "aiming for 3–6 dB of gain reduction at most."
    ),
    "well_balanced_minimal": (
        "Your dynamics are healthy and need almost no intervention. The mix has natural variation "
        "that feels expressive without being unpredictable. If you want to add cohesion, consider "
        "light parallel compression — blend a heavily compressed signal with the dry signal at "
        "low levels to add glue while preserving the original dynamics."
    ),
    "well_balanced_light": (
        "Your dynamics are in good shape but would benefit from gentle compression. A 2:1 ratio "
        "with a medium attack (40–60 ms) allows the initial transient through before compression "
        "kicks in, preserving the punch of drums and percussion while evening out the sustained "
        "body of the sound. Aim for 2–4 dB of gain reduction on the loudest passages."
    ),
    "moderate_peaks": (
        "Your mix has moderate peaks noticeably louder than the average level. Compression will "
        "even out these volume spikes so the mix sits consistently in the listener's ear. Start "
        "with a 4:1 ratio and set the threshold until you see 3–6 dB of gain reduction on peaks. "
        "A medium attack (30–50 ms) preserves transient punch while controlling the sustained energy."
    ),
    "high_peaks": (
        "Your mix has significant peaks — the loudest moments are much louder than average. "
        "Without compression these spikes will sound jarring and may clip on streaming platforms. "
        "Use an 8:1 ratio with a fast attack (5–15 ms) to tame the peaks, then apply make-up "
        "gain to restore the average loudness. Follow with a limiter to catch any remaining "
        "peaks before export."
    ),
}


def analyze_dynamics(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze dynamic range and related metrics of an audio signal.
    
    Computes various dynamic metrics including peak amplitude, RMS energy,
    dynamic range, crest factor, and loudness variation over time. These metrics
    form the basis for compression recommendations.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal (time-domain samples)
    sr : int
        Sample rate (samples per second)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary containing:
        - 'peak_amplitude': Maximum absolute amplitude in the signal
        - 'peak_db': Peak amplitude in dB (20*log10(peak))
        - 'rms_energy': Root mean square energy of entire signal
        - 'rms_db': RMS energy in dB (20*log10(rms))
        - 'dynamic_range': Peak dB - RMS dB (difference)
        - 'crest_factor_db': Peak to RMS ratio in dB
        - 'loudness_range': Variation in loudness over time (std dev of frame loudness)
        - 'mean_loudness': Average loudness across time
        
    Example:
    --------
    >>> y, sr = librosa.load('audio.wav')
    >>> dynamics = analyze_dynamics(y, sr)
    >>> print(f"Dynamic Range: {dynamics['dynamic_range']:.1f} dB")
    Dynamic Range: 24.3 dB
    """
    # Calculate peak amplitude
    peak_amplitude = np.max(np.abs(y))
    
    # Avoid log(0) by adding small epsilon value
    peak_amplitude = max(peak_amplitude, 1e-10)
    
    # Convert peak to dB (20*log10 for amplitude)
    peak_db = 20 * np.log10(peak_amplitude)
    
    # Calculate RMS energy
    rms_energy = np.sqrt(np.mean(y ** 2))
    rms_energy = max(rms_energy, 1e-10)
    
    # Convert RMS to dB
    rms_db = 20 * np.log10(rms_energy)
    
    # Dynamic range = difference between peak and RMS in dB
    dynamic_range = peak_db - rms_db
    
    # Crest factor = peak / RMS ratio, converted to dB
    crest_factor_linear = peak_amplitude / rms_energy
    crest_factor_db = 20 * np.log10(crest_factor_linear)
    
    # Calculate loudness over time using frame-based RMS
    # Split audio into frames and compute RMS for each frame
    frame_length = 2048  # ~46ms at 44.1kHz, standard for loudness analysis
    n_frames = len(y) // frame_length
    
    frame_rms = []
    for i in range(n_frames):
        frame = y[i * frame_length:(i + 1) * frame_length]
        frame_rms_val = np.sqrt(np.mean(frame ** 2))
        frame_rms_val = max(frame_rms_val, 1e-10)
        frame_db = 20 * np.log10(frame_rms_val)
        frame_rms.append(frame_db)
    
    # Calculate loudness range as standard deviation of frame loudness
    if len(frame_rms) > 0:
        loudness_range = np.std(frame_rms)
        mean_loudness = np.mean(frame_rms)
    else:
        loudness_range = 0.0
        mean_loudness = rms_db
    
    dynamics = {
        "peak_amplitude": peak_amplitude,
        "peak_db": peak_db,
        "rms_energy": rms_energy,
        "rms_db": rms_db,
        "dynamic_range": dynamic_range,
        "crest_factor_db": crest_factor_db,
        "loudness_range": loudness_range,
        "mean_loudness": mean_loudness,
    }
    
    return dynamics


def get_compression_params(
    y: np.ndarray,
    sr: int,
    model_path: str = "models/compression_model.joblib",
) -> Dict[str, float]:
    """
    Predict compression parameters using the trained ML model.

    Falls back to analytically derived values (from dynamics) if the model
    file has not been generated yet.

    Returns dict with: ratio, threshold_db, attack_ms, release_ms
    """
    try:
        from train_compression_model import predict_compression_params
        return predict_compression_params(y, sr, model_path=model_path)
    except Exception:
        pass
    # Fallback: derive analytically from signal dynamics
    dynamics = analyze_dynamics(y, sr)
    crest = dynamics["crest_factor_db"]
    rms_db = dynamics["rms_db"]
    onset_rate = 2.0  # neutral default without librosa onset detection
    ratio        = float(np.clip(1.0 + max(0.0, crest - 4.0) * 0.55, 1.0, 16.0))
    threshold_db = float(np.clip(rms_db + 4.0, -40.0, -6.0))
    attack_ms    = float(np.clip(80.0 / (1.0 + onset_rate * 0.9), 1.0, 100.0))
    release_ms   = float(np.clip(attack_ms * 4.0, 50.0, 1000.0))
    return {"ratio": ratio, "threshold_db": threshold_db,
            "attack_ms": attack_ms, "release_ms": release_ms}


def get_compression_recommendations(
    dynamics: Dict[str, float],
    ml_params: Dict[str, float] = None,
) -> Dict[str, str]:
    """
    Generate compression recommendations based on dynamic range analysis.

    When ml_params is provided (output of get_compression_params()), the ML
    predictions are used for ratio, attack, and release; assessment is still
    derived from measured dynamics so it remains accurate. Without ml_params
    the original rule-based logic is used as a fallback.

    Parameters:
    -----------
    dynamics  : Dict[str, float]  — from analyze_dynamics()
    ml_params : Dict[str, float]  — optional, from get_compression_params()

    Returns:
    --------
    Dict[str, str] with keys: compression_ratio, attack_time, release_time,
    compression_assessment, crest_factor, dynamic_range
    """
    crest_factor  = dynamics["crest_factor_db"]
    dynamic_range = dynamics["dynamic_range"]

    # ── Assessment (always rule-based — ML cannot observe over-compression) ───
    if dynamic_range < LOW_DYNAMIC_RANGE:
        assessment = "Over-compressed - dynamics are limited"
    elif crest_factor < CREST_FACTOR_LOW:
        assessment = "Well-balanced - minimal compression needed"
    elif crest_factor < CREST_FACTOR_MEDIUM:
        assessment = "Well-balanced - light compression recommended"
    elif crest_factor < CREST_FACTOR_HIGH:
        assessment = "Needs compression - moderate peaks detected"
    else:
        assessment = "Needs compression - high peaks detected"

    if ml_params is not None:
        # ── ML-derived ratio / attack / release ──────────────────────────────
        ratio      = ml_params["ratio"]
        attack_ms  = ml_params["attack_ms"]
        release_ms = ml_params["release_ms"]

        # Override assessment label for over-compressed check
        if dynamic_range < LOW_DYNAMIC_RANGE:
            ratio = 1.5  # Don't recommend further compression

        # Human-readable labels
        attack_label  = "fast" if attack_ms < 20 else ("medium" if attack_ms < 60 else "slow")
        release_label = "fast" if release_ms < 100 else ("medium" if release_ms < 350 else "slow")

        ratio_str   = f"{ratio:.1f}:1"
        attack_str  = f"{attack_ms:.0f} ms ({attack_label})"
        release_str = f"{release_ms:.0f} ms ({release_label})"

    else:
        # ── Fallback: original rule-based logic ───────────────────────────────
        loudness_range = dynamics["loudness_range"]

        if crest_factor < CREST_FACTOR_LOW:
            compression_ratio = 1.5
        elif crest_factor < CREST_FACTOR_MEDIUM:
            compression_ratio = 2.0
        elif crest_factor < CREST_FACTOR_HIGH:
            compression_ratio = 4.0
        else:
            compression_ratio = 8.0
        if dynamic_range < LOW_DYNAMIC_RANGE:
            compression_ratio = 1.5

        attack_key  = "fast" if loudness_range > 8 else ("medium" if loudness_range > 4 else "slow")
        release_key = "slow" if dynamic_range > MEDIUM_DYNAMIC_RANGE else "medium"

        ratio_str   = f"{compression_ratio}:1"
        attack_str  = f"{ATTACK_TIMES[attack_key]} ms ({attack_key})"
        release_str = f"{RELEASE_TIMES[release_key]} ms ({release_key})"

    return {
        "compression_ratio":    ratio_str,
        "attack_time":          attack_str,
        "release_time":         release_str,
        "compression_assessment": assessment,
        "crest_factor":         f"{crest_factor:.1f} dB",
        "dynamic_range":        f"{dynamic_range:.1f} dB",
    }


def plot_dynamic_range(y: np.ndarray, sr: int) -> plt.Figure:
    """
    Plot dynamic range over time showing peak and RMS envelopes.
    
    Creates a dual-axis plot displaying frame-based RMS energy (representing
    average loudness) and peak amplitude envelope over time. Visually illustrates
    the dynamic characteristics of the audio signal and helps identify peaks
    and low-level sections.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal (time-domain samples)
    sr : int
        Sample rate (samples per second)
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure object containing the dynamic range visualization.
        Can be displayed with plt.show() or saved with fig.savefig().
        
    Example:
    --------
    >>> y, sr = librosa.load('audio.wav')
    >>> fig = plot_dynamic_range(y, sr)
    >>> fig.savefig('dynamics.png', dpi=150, bbox_inches='tight')
    """
    # Frame-based analysis parameters
    frame_length = 2048  # ~46ms at 44.1kHz
    hop_length = 512     # 75% overlap
    
    # Calculate number of frames
    n_frames = 1 + (len(y) - frame_length) // hop_length
    
    # Initialize arrays to store frame metrics
    frame_rms = np.zeros(n_frames)
    frame_peak = np.zeros(n_frames)
    frame_times = np.zeros(n_frames)
    
    # Process each frame
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        
        # Handle last frame which may be shorter
        if end > len(y):
            frame = y[start:]
        else:
            frame = y[start:end]
        
        # Calculate RMS for this frame
        rms_val = np.sqrt(np.mean(frame ** 2))
        frame_rms[i] = 20 * np.log10(max(rms_val, 1e-10))
        
        # Calculate peak amplitude for this frame
        peak_val = np.max(np.abs(frame))
        frame_peak[i] = 20 * np.log10(max(peak_val, 1e-10))
        
        # Calculate time in seconds for this frame
        frame_times[i] = (start + hop_length / 2) / sr
    
    # Create figure with subplots
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot RMS energy (loudness) on primary y-axis
    color_rms = 'steelblue'
    ax1.plot(frame_times, frame_rms, linewidth=2, color=color_rms, label='RMS Energy (Loudness)', alpha=0.8)
    ax1.fill_between(frame_times, frame_rms, alpha=0.2, color=color_rms)
    ax1.set_xlabel("Time (seconds)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("RMS Energy (dB)", fontsize=12, fontweight='bold', color=color_rms)
    ax1.tick_params(axis='y', labelcolor=color_rms)
    
    # Create secondary y-axis for peak amplitude
    ax2 = ax1.twinx()
    color_peak = 'orangered'
    ax2.plot(frame_times, frame_peak, linewidth=2, color=color_peak, label='Peak Amplitude', alpha=0.8, linestyle='--')
    ax2.set_ylabel("Peak Amplitude (dB)", fontsize=12, fontweight='bold', color=color_peak)
    ax2.tick_params(axis='y', labelcolor=color_peak)
    
    # Add title
    plt.title("Dynamic Range Analysis Over Time", fontsize=14, fontweight='bold', pad=20)
    
    # Configure grid
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add horizontal reference lines
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='0 dB Reference')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


def get_compression_explanation(assessment: str) -> str:
    """Return an educational explanation for a compression assessment string."""
    if "Over-compressed" in assessment:
        return COMPRESSION_EXPLANATIONS["over_compressed"]
    if "high peaks" in assessment:
        return COMPRESSION_EXPLANATIONS["high_peaks"]
    if "moderate peaks" in assessment:
        return COMPRESSION_EXPLANATIONS["moderate_peaks"]
    if "minimal" in assessment:
        return COMPRESSION_EXPLANATIONS["well_balanced_minimal"]
    return COMPRESSION_EXPLANATIONS["well_balanced_light"]

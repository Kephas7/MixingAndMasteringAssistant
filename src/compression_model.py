"""
Compression Analysis Model for MMMA (Music Mixing and Mastering Assistant)
Analyzes dynamic range of audio signals and provides compression recommendations.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Dict, Tuple

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


def get_compression_recommendations(dynamics: Dict[str, float]) -> Dict[str, str]:
    """
    Generate compression recommendations based on dynamic range analysis.
    
    Evaluates crest factor and dynamic range metrics to recommend appropriate
    compression settings including ratio, attack/release times, and overall
    compression assessment (over-compressed, well-balanced, or needs compression).
    
    Parameters:
    -----------
    dynamics : Dict[str, float]
        Dynamic metrics from analyze_dynamics().
    
    Returns:
    --------
    Dict[str, str]
        Dictionary containing:
        - 'compression_ratio': Recommended ratio (e.g., "2:1", "4:1")
        - 'attack_time': Recommended attack in milliseconds (e.g., "10 ms (fast)")
        - 'release_time': Recommended release in milliseconds (e.g., "200 ms (medium)")
        - 'compression_assessment': Overall assessment of current compression state
        
    Example:
    --------
    >>> dynamics = analyze_dynamics(y, sr)
    >>> recs = get_compression_recommendations(dynamics)
    >>> print(recs['compression_ratio'])
    '4:1'
    >>> print(recs['compression_assessment'])
    'Needs compression - high peaks detected'
    """
    recommendations = {}
    
    crest_factor = dynamics["crest_factor_db"]
    dynamic_range = dynamics["dynamic_range"]
    
    # Determine compression ratio based on crest factor
    # Higher crest factor means more aggressive compression needed
    if crest_factor < CREST_FACTOR_LOW:
        # Very low crest factor: minimal compression needed
        compression_ratio = 1.5
        assessment = "Well-balanced - minimal compression needed"
    elif crest_factor < CREST_FACTOR_MEDIUM:
        # Low to medium: light to moderate compression
        compression_ratio = 2.0
        assessment = "Well-balanced - light compression recommended"
    elif crest_factor < CREST_FACTOR_HIGH:
        # Medium to high: moderate to aggressive compression
        compression_ratio = 4.0
        assessment = "Needs compression - moderate peaks detected"
    else:
        # High crest factor: aggressive compression needed
        compression_ratio = 8.0
        assessment = "Needs compression - high peaks detected"
    
    # Over-compression detection: if dynamic range is very low,
    # the track is already heavily compressed
    if dynamic_range < LOW_DYNAMIC_RANGE:
        assessment = "Over-compressed - dynamics are limited"
        compression_ratio = 1.5  # No additional compression needed
    
    # Attack time determination based on dynamic range variation
    # High loudness range variation suggests dynamic content needing fast attack
    loudness_range = dynamics["loudness_range"]
    if loudness_range > 8:  # High variation
        attack_time = "fast"
    elif loudness_range > 4:
        attack_time = "medium"
    else:
        attack_time = "slow"
    
    # Release time determination based on type of content
    # Music with sustained notes benefits from slower release
    # Percussive content benefits from faster release
    if dynamic_range > MEDIUM_DYNAMIC_RANGE:
        release_time = "slow"  # Preserve sustain on dynamic content
    else:
        release_time = "medium"
    
    # Format recommendation strings
    recommendations["compression_ratio"] = f"{compression_ratio}:1"
    recommendations["attack_time"] = f"{ATTACK_TIMES[attack_time]} ms ({attack_time})"
    recommendations["release_time"] = f"{RELEASE_TIMES[release_time]} ms ({release_time})"
    recommendations["compression_assessment"] = assessment
    
    # Add additional metrics for user reference
    recommendations["crest_factor"] = f"{crest_factor:.1f} dB"
    recommendations["dynamic_range"] = f"{dynamic_range:.1f} dB"
    
    return recommendations


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

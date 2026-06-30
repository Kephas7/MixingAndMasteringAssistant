"""
Equalization Analysis Model for MMMA (Music Mixing and Mastering Assistant)
Analyzes frequency content and provides EQ recommendations based on industry standards.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import Dict

# Define frequency bands in Hz (standard audio engineering ranges)
FREQUENCY_BANDS = {
    "Sub-bass": (20, 60),
    "Bass": (60, 250),
    "Low-mid": (250, 500),
    "Mid": (500, 2000),
    "Upper-mid": (2000, 6000),
    "Presence": (6000, 12000),
    "Air": (12000, 20000),
}

# Reference ideal energy balance (normalized to 0-1)
# These represent typical energy distribution in a professional mix
# Based on industry standards for well-balanced audio
IDEAL_ENERGY = {
    "Sub-bass": 0.5,
    "Bass": 1.0,
    "Low-mid": 0.9,
    "Mid": 1.0,
    "Upper-mid": 0.85,
    "Presence": 0.75,
    "Air": 0.5,
}

# Threshold for EQ recommendations in dB
BOOST_THRESHOLD = -2  # Below this, recommend boost
CUT_THRESHOLD = 2     # Above this, recommend cut

# Educational explanations for each band and action direction
EQ_EXPLANATIONS = {
    "Sub-bass": {
        "boost": (
            "Sub-bass adds physical weight and impact — essential for hip-hop, EDM, and trap. "
            "Boosting below 60 Hz makes the mix feel powerful on subwoofers, but too much causes "
            "muddiness on earbuds and laptop speakers where this range is barely reproduced."
        ),
        "cut": (
            "Excess sub-bass below 60 Hz rarely translates to small speakers and makes the kick "
            "and bass feel indistinct. A high-pass filter at 30–40 Hz cleans this up without "
            "affecting perceived bass weight."
        ),
        "neutral": "Sub-bass energy is balanced — good foundation without excess rumble.",
    },
    "Bass": {
        "boost": (
            "The bass region gives your mix warmth and body. Boosting here adds fullness to kick "
            "drums, bass guitars, and low-end synths. If the mix sounds thin or hollow on big "
            "speakers, this is where to add energy."
        ),
        "cut": (
            "Excess bass between 60–250 Hz is the most common cause of a muddy mix — instruments "
            "lose clarity and the low end sounds congested. Cutting here will immediately separate "
            "elements and add punch to the kick and bass."
        ),
        "neutral": "Bass energy is well-balanced — good weight without muddiness.",
    },
    "Low-mid": {
        "boost": (
            "The low-mids add body and warmth to instruments. Boosting adds fullness, but this "
            "range is easy to over-do — small increases of 1–2 dB go a long way. Too much and "
            "you'll introduce the 'boxiness' problem described below."
        ),
        "cut": (
            "Excess low-mids make mixes sound 'boxy' or 'honky' — like audio coming from a "
            "cardboard box. Cutting here is one of the most effective ways to add clarity and "
            "create space for vocals and lead instruments to sit."
        ),
        "neutral": "Low-mids are balanced — instruments have body without boxiness.",
    },
    "Mid": {
        "boost": (
            "The mids carry the core of most instruments. Boosting adds presence and 'forwardness', "
            "making elements cut through a dense mix. Great for bringing a buried vocal or lead "
            "synth to the front without raising its overall volume."
        ),
        "cut": (
            "Excess mids cause a nasal or harsh quality. If anything in your mix sounds grating "
            "or becomes fatiguing over time, reducing this region often helps. Subtle cuts of "
            "2–3 dB here can open up the mix considerably."
        ),
        "neutral": "Midrange is well-balanced — instruments have presence without harshness.",
    },
    "Upper-mid": {
        "boost": (
            "The upper-mids control the attack and articulation of instruments. Boosting here makes "
            "snares crack, guitars bite, and vocals cut through clearly. Essential for getting "
            "individual elements to stand out in a dense arrangement."
        ),
        "cut": (
            "High upper-mid energy is the leading cause of ear fatigue — the mix sounds harsh or "
            "piercing after a few minutes of listening. A gentle cut of 1–3 dB here is often the "
            "difference between a mix that's enjoyable long-term and one that wears you out."
        ),
        "neutral": "Upper-mids are balanced — good clarity and attack without harshness.",
    },
    "Presence": {
        "boost": (
            "The presence region adds shimmer and definition. Boosting here makes cymbals sparkle, "
            "gives vocals that professional 'airy' quality, and helps the mix translate on earbuds "
            "and phones where this range is often emphasized."
        ),
        "cut": (
            "Excess presence causes sibilance — that piercing 'ssss' sound on vocals and the harsh "
            "edge on cymbals. A gentle cut here will smooth things out. For vocals specifically, "
            "a de-esser is more surgical than a broadband EQ cut."
        ),
        "neutral": "Presence is balanced — good sparkle and definition without harshness.",
    },
    "Air": {
        "boost": (
            "The air band adds an open, extended top-end that makes mixes sound professional and "
            "polished. A gentle high-shelf boost of 1–2 dB here is a classic mastering move. "
            "Subtlety is key — the goal is 'open', not 'bright'."
        ),
        "cut": (
            "Too much air above 12 kHz makes the mix sound brittle or harsh, especially on digital "
            "formats and streaming. A gentle high-shelf cut will warm things up. This band should "
            "be present and supportive, not the dominant character of the mix."
        ),
        "neutral": "Air frequencies are balanced — open top-end without brittleness.",
    },
}


def analyze_frequency_bands(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze energy levels across 7 frequency bands using FFT.

    Computes STFT magnitude (linear amplitude) for each band and normalizes
    the result to a 0–1 scale relative to the loudest band. Working in linear
    amplitude space (not dB) avoids sign-inversion issues when normalizing.

    Parameters:
    -----------
    y : np.ndarray
        Audio signal (time-domain samples)
    sr : int
        Sample rate (samples per second)

    Returns:
    --------
    Dict[str, float]
        Band names mapped to normalized amplitude values (0–1).
    """
    # STFT magnitude — linear amplitude, all values ≥ 0
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0] * 2 - 2)

    band_energies = {}
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        mask = (freqs >= low_freq) & (freqs < high_freq)
        band_energies[band_name] = float(np.mean(S[mask, :])) if np.any(mask) else 0.0

    # Normalize to 0–1 relative to the dominant band
    max_energy = max(band_energies.values()) if band_energies else 1.0
    if max_energy > 0:
        band_energies = {k: v / max_energy for k, v in band_energies.items()}
    else:
        band_energies = {k: 0.0 for k in band_energies}

    return band_energies


def get_eq_recommendations(band_energies: Dict[str, float]) -> Dict[str, str]:
    """
    Compare band energies against ideal reference mix balance and provide EQ recommendations.
    
    Calculates the dB difference between actual and ideal energy for each band,
    then provides EQ adjustment recommendations (Boost, Cut, or Neutral) based on
    threshold values. Positive dB values indicate the band is above ideal (should cut),
    negative values indicate the band is below ideal (should boost).
    
    Parameters:
    -----------
    band_energies : Dict[str, float]
        Energy values for each band from analyze_frequency_bands().
        Expected to be normalized 0-1 values.
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping band names to recommendation strings.
        Format examples: "Boost: 3.5 dB", "Cut: 2.1 dB", "Neutral".
        
    Example:
    --------
    >>> energies = {'Bass': 0.95, 'Mid': 0.8, ...}
    >>> recs = get_eq_recommendations(energies)
    >>> print(recs['Bass'])
    'Cut: 1.4 dB'
    """
    recommendations = {}
    
    for band_name, energy in band_energies.items():
        # Get ideal energy for this band
        ideal = IDEAL_ENERGY.get(band_name, 1.0)
        
        # Calculate ratio between actual and ideal energy
        if ideal > 0:
            ratio = energy / ideal
        else:
            ratio = 1.0
        
        # Convert ratio to dB scale: 20 * log10(ratio)
        # Clamp ratio to avoid log of zero or negative numbers
        ratio = max(ratio, 0.01)
        db_diff = 20 * np.log10(ratio)
        
        # Cap to a practical EQ range — no real EQ adjustment exceeds ±12 dB
        db_diff = max(-12.0, min(12.0, db_diff))

        # Generate recommendation based on dB difference threshold
        # Positive db_diff means energy is above ideal -> should cut
        # Negative db_diff means energy is below ideal -> should boost
        if db_diff > CUT_THRESHOLD:
            recommendations[band_name] = f"Cut: {db_diff:.1f} dB"
        elif db_diff < BOOST_THRESHOLD:
            recommendations[band_name] = f"Boost: {abs(db_diff):.1f} dB"
        else:
            recommendations[band_name] = "Neutral"
    
    return recommendations


def plot_frequency_spectrum(y: np.ndarray, sr: int) -> plt.Figure:
    """
    Plot the full frequency spectrum of the audio signal with EQ band markers.
    
    Creates a logarithmic frequency spectrum plot showing magnitude response across
    the entire audible range (20 Hz - 20 kHz), with colored region markers for each
    EQ band defined in FREQUENCY_BANDS. Useful for visual analysis of frequency content.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal (time-domain samples)
    sr : int
        Sample rate (samples per second)
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure object containing the spectrum plot.
        Can be displayed with plt.show() or saved with fig.savefig().
        
    Example:
    --------
    >>> y, sr = librosa.load('audio.wav')
    >>> fig = plot_frequency_spectrum(y, sr)
    >>> fig.savefig('spectrum.png', dpi=150, bbox_inches='tight')
    """
    # Compute STFT magnitude spectrum
    S = np.abs(librosa.stft(y))
    
    # Convert to dB scale for better visual representation
    S_db = librosa.power_to_db(S**2, ref=np.max)
    
    # Get frequency bins for the STFT
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0] * 2 - 2)
    
    # Average magnitude across time dimension to get overall spectrum
    magnitude = np.mean(S_db, axis=1)
    
    # Create figure with appropriate size for frequency analysis
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the spectrum using logarithmic scale for y-axis (magnitude in dB)
    ax.semilogy(freqs, magnitude, linewidth=2.5, color='steelblue', label='Spectrum')
    
    # Set axis labels and title
    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Magnitude (dB)", fontsize=12, fontweight='bold')
    ax.set_title("Audio Frequency Spectrum Analysis", fontsize=14, fontweight='bold')
    
    # Configure grid for easier reading
    ax.grid(True, which="both", alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set x-axis to audible range
    ax.set_xlim([20, 20000])
    
    # Color palette for band visualization
    colors = ['#FF6B6B', '#FFA500', '#FFD93D', '#6BCB77', '#4D96FF', '#9D84B7', '#FF006E']
    
    # Add band region markers with semi-transparent background and dashed borders
    for (band_name, (low_freq, high_freq)), color in zip(FREQUENCY_BANDS.items(), colors):
        # Semi-transparent background for each band
        ax.axvspan(low_freq, high_freq, alpha=0.15, color=color, label=band_name)
        
        # Dashed border lines at band edges for clarity
        ax.axvline(low_freq, color=color, linestyle='--', alpha=0.4, linewidth=1)
        ax.axvline(high_freq, color=color, linestyle='--', alpha=0.4, linewidth=1)
    
    # Add legend showing all frequency bands
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


def get_eq_explanation(band: str, recommendation: str) -> str:
    """Return an educational explanation for an EQ recommendation string."""
    explanations = EQ_EXPLANATIONS.get(band, {})
    if "Boost" in recommendation:
        return explanations.get("boost", "")
    elif "Cut" in recommendation:
        return explanations.get("cut", "")
    return explanations.get("neutral", "")
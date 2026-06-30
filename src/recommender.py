"""
Unified Recommender for MMMA (Music Mixing and Mastering Assistant)
Synthesizes EQ, compression, and mastering results into a mix quality score
and a prioritized, actionable top-3 recommendation list.
"""

from typing import Dict, List


def _score_to_grade(score: int) -> str:
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _score_to_label(score: int) -> str:
    if score >= 90: return "Professional"
    if score >= 80: return "Nearly There"
    if score >= 70: return "Good Foundation"
    if score >= 60: return "Needs Work"
    return "Major Issues"


def generate_mix_report(
    eq_recommendations: Dict[str, str],
    compression_recs:   Dict[str, str],
    mastering_recs:     Dict[str, str],
) -> Dict:
    """
    Synthesize all analysis results into a mix quality score and a top-3
    priority action list.

    Parameters
    ----------
    eq_recommendations : output of get_eq_recommendations()
    compression_recs   : output of get_compression_recommendations()
    mastering_recs     : output of get_mastering_recommendations()

    Returns
    -------
    Dict with keys:
        score      : int 0-100
        grade      : str "A"–"F"
        label      : str descriptive label
        summary    : str one-sentence overview
        strengths  : List[str] things working well
        priorities : List[dict] top 3 issues, each with rank/category/action/why
    """
    score     = 100
    issues:   List[Dict] = []
    strengths: List[str] = []

    # -----------------------------------------------------------------------
    # Mastering issues (highest stakes — affect streaming delivery directly)
    # -----------------------------------------------------------------------
    loudness_status  = mastering_recs.get("loudness_status", "")
    true_peak_status = mastering_recs.get("true_peak_status", "")
    lra_status       = mastering_recs.get("lra_status", "")

    if true_peak_status == "true_peak_exceeded":
        score -= 20
        issues.append({
            "priority": 5,
            "category": "Mastering",
            "action":   "Lower your limiter ceiling to -1.0 dBTP",
            "why": (
                "Your true peak exceeds -1.0 dBTP. After MP3/AAC encoding, inter-sample "
                "peaks above this limit cause audible distortion on most playback devices. "
                "Set your limiter's output ceiling to -1.0 dBTP before export."
            ),
        })
    else:
        strengths.append("True peak is within the -1.0 dBTP streaming limit")

    if loudness_status == "too_loud":
        score -= 14
        gain = mastering_recs.get("gain_adjustment", "")
        issues.append({
            "priority": 4,
            "category": "Mastering",
            "action":   f"Reduce loudness ({gain}) before export",
            "why": (
                "Streaming platforms turn down anything louder than -14 LUFS using a "
                "limiter, which often degrades the sound. Targeting the platform's native "
                "level sounds better than being normalized down."
            ),
        })
    elif loudness_status == "too_quiet":
        score -= 12
        gain = mastering_recs.get("gain_adjustment", "")
        issues.append({
            "priority": 4,
            "category": "Mastering",
            "action":   f"Increase loudness ({gain}) to target -14 LUFS",
            "why": (
                "Tracks significantly below -14 LUFS sound noticeably quieter than "
                "surrounding tracks in a playlist. Apply a limiter with a -1.0 dBTP "
                "ceiling and use make-up gain to bring the integrated loudness to target."
            ),
        })
    else:
        strengths.append("Integrated loudness is well-targeted for streaming")

    if lra_status == "lra_low":
        score -= 10
        issues.append({
            "priority": 3,
            "category": "Mastering",
            "action":   "Ease the limiter to allow more dynamic range",
            "why": (
                "A very low Loudness Range (LRA) is a sign of over-limiting. The mix sounds "
                "flat and fatiguing because the loud and quiet moments are nearly the same "
                "volume. Raise the limiter threshold and let the dynamics breathe."
            ),
        })
    elif lra_status == "lra_high":
        score -= 8
        issues.append({
            "priority": 2,
            "category": "Mastering",
            "action":   "Apply a gentle mastering compressor before the limiter",
            "why": (
                "A high Loudness Range means the volume swings are too wide for a consistent "
                "playlist experience. A gentle 2:1 compressor with a slow attack before the "
                "limiter will tighten this without squashing the life from the mix."
            ),
        })
    else:
        strengths.append("Loudness Range (LRA) is healthy")

    # -----------------------------------------------------------------------
    # Compression issues
    # -----------------------------------------------------------------------
    assessment = compression_recs.get("compression_assessment", "")
    if "Over-compressed" in assessment:
        score -= 18
        issues.append({
            "priority": 4,
            "category": "Compression",
            "action":   "Reduce compression — dynamics are already too limited",
            "why": (
                "The dynamic range is so narrow the mix sounds flat and lifeless. "
                "Adding more compression will make it worse. Re-examine gain staging "
                "and aim for no more than 3–6 dB of gain reduction per compressor."
            ),
        })
    elif "high peaks" in assessment:
        score -= 12
        ratio  = compression_recs.get("compression_ratio", "")
        attack = compression_recs.get("attack_time", "")
        issues.append({
            "priority": 3,
            "category": "Compression",
            "action":   f"Compress at {ratio} ratio with {attack} attack",
            "why": (
                "Significant peaks make the mix sound inconsistent and risk clipping after "
                "encoding. A higher ratio with a fast attack tames the peaks while make-up "
                "gain restores the average loudness."
            ),
        })
    elif "moderate peaks" in assessment:
        score -= 6
        ratio = compression_recs.get("compression_ratio", "")
        issues.append({
            "priority": 2,
            "category": "Compression",
            "action":   f"Apply light compression: {ratio} ratio",
            "why": (
                "Moderate peaks are creating volume inconsistency. Light compression will "
                "even these out so the mix sits consistently without sacrificing the "
                "natural dynamics of the performance."
            ),
        })
    else:
        strengths.append("Dynamics and compression are well-balanced")

    # -----------------------------------------------------------------------
    # EQ issues — surface the single worst band as a priority item
    # -----------------------------------------------------------------------
    actionable_eq = {b: r for b, r in eq_recommendations.items() if r != "Neutral"}
    n_eq = len(actionable_eq)

    # Count-based deduction
    if n_eq <= 0:
        pass
    elif n_eq <= 2:
        score -= 5
    elif n_eq <= 4:
        score -= 10
    else:
        score -= 18

    # Find the most severe band
    worst_band = None
    worst_db   = 0.0
    for band, rec in actionable_eq.items():
        try:
            db_val = float(rec.split(":")[1].strip().replace(" dB", ""))
        except (IndexError, ValueError):
            db_val = 0.0
        if db_val > worst_db:
            worst_db   = db_val
            worst_band = (band, rec, db_val)

    # Severity bonus deduction
    if worst_db > 10:
        score -= 8
    elif worst_db > 6:
        score -= 5
    elif worst_db > 3:
        score -= 3

    if worst_band:
        band, rec, db_val = worst_band
        direction = "Boost" if "Boost" in rec else "Cut"
        other_count = n_eq - 1
        tail = f" ({other_count} other band{'s' if other_count != 1 else ''} also need attention)" if other_count > 0 else ""
        issues.append({
            "priority": 3 if db_val > 6 else (2 if db_val > 3 else 1),
            "category": "EQ",
            "action":   f"{direction} the {band} band by {db_val:.1f} dB{tail}",
            "why": (
                f"The {band} band has the largest frequency imbalance in your mix "
                f"({direction.lower()} of {db_val:.1f} dB). Addressing the biggest EQ "
                "imbalances first has the greatest perceptual impact. See the EQ section "
                "below for the full band-by-band breakdown."
            ),
        })
    else:
        strengths.append("Frequency balance is well-distributed across all bands")

    # -----------------------------------------------------------------------
    # Assemble final report
    # -----------------------------------------------------------------------
    issues.sort(key=lambda x: x["priority"], reverse=True)
    top_3 = issues[:3]
    for i, issue in enumerate(top_3, start=1):
        issue["rank"] = i

    score = max(0, min(100, score))

    if score >= 90:
        summary = "Your mix is in excellent shape — ready for release."
    elif score >= 80:
        summary = "Solid mix with a few areas to polish before release."
    elif score >= 70:
        summary = "Good foundation — focus on the priorities below to get release-ready."
    elif score >= 60:
        summary = "Several areas need attention. Work through the priorities below."
    else:
        summary = "Multiple significant issues detected. Work through the priorities below one at a time."

    return {
        "score":      score,
        "grade":      _score_to_grade(score),
        "label":      _score_to_label(score),
        "summary":    summary,
        "strengths":  strengths,
        "priorities": top_3,
    }

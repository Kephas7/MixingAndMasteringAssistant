import streamlit as st
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_extraction import load_audio, get_basic_info, plot_waveform, plot_spectrogram, get_spectral_features
from eq_model import analyze_frequency_bands, get_eq_recommendations, plot_frequency_spectrum, get_eq_explanation
from compression_model import analyze_dynamics, get_compression_recommendations, plot_dynamic_range, get_compression_explanation
from mastering_model import (analyze_loudness, get_mastering_recommendations, get_mastering_explanation,
                              plot_loudness_vs_targets, STREAMING_TARGETS)
from recommender import generate_mix_report

st.set_page_config(page_title="Mix Assistant", page_icon="🎚️", layout="wide")

# ── UI Helpers ─────────────────────────────────────────────────────────────────

# Human-readable names and perceptual descriptions for each EQ band
BAND_FRIENDLY = {
    "Sub-bass":  ("Deep Bass",   "20–60 Hz",   "The chest-rumble of kicks and sub-bass synths"),
    "Bass":      ("Bass",        "60–250 Hz",  "Warmth and weight — kick drum body, bass guitar"),
    "Low-mid":   ("Low Mids",    "250–500 Hz", "Can make a mix sound 'boxy' if overdone"),
    "Mid":       ("Mids",        "500–2 kHz",  "Where vocals and most instruments live"),
    "Upper-mid": ("Upper Mids",  "2–6 kHz",    "Clarity and bite — helps things cut through"),
    "Presence":  ("Presence",    "6–12 kHz",   "Sparkle, sibilance, cymbal shimmer"),
    "Air":       ("Air",         "12–20 kHz",  "The open, polished top-end sheen"),
}


def _eq_health(eq_recs):
    n = sum(1 for r in eq_recs.values() if r != "Neutral")
    if n == 0:  return "good",    "Frequency balance is excellent"
    if n <= 2:  return "warning", f"{n} frequency area{'s' if n > 1 else ''} need attention"
    return             "bad",     f"{n} of 7 frequency areas are out of balance"


def _comp_health(comp_recs):
    a = comp_recs["compression_assessment"]
    if "Over-compressed" in a: return "bad",     "Dynamics are too squashed — the track sounds flat"
    if "high peaks"      in a: return "warning",  "Loud spikes need taming"
    if "moderate peaks"  in a: return "warning",  "Light compression would help even things out"
    return                            "good",     "Dynamics are well-balanced"


def _loud_health(master_recs):
    ls = master_recs["loudness_status"]
    tp = master_recs["true_peak_status"]
    lr = master_recs["lra_status"]
    if ls == "too_loud" or tp == "true_peak_exceeded": return "bad",     master_recs["loudness_assessment"]
    if ls == "too_quiet" or lr != "lra_good":          return "warning",  master_recs["loudness_assessment"]
    return                                                    "good",     "Your track is streaming-ready 🎉"


def _card(label, status, summary, action=None):
    """Colored status card with optional action line."""
    c = {"good": "#6BCB77", "warning": "#FFD93D", "bad": "#FF6B6B"}[status]
    i = {"good": "✅",       "warning": "⚠️",       "bad": "🔴"}[status]
    act = f'<div style="margin-top:6px;font-size:13px;color:{c};font-weight:600;">→ {action}</div>' if action else ""
    st.markdown(f"""
    <div style="padding:16px 18px;border-radius:10px;border-left:5px solid {c};
                background:{c}18;margin:6px 0;">
        <span style="font-weight:700;font-size:15px;">{i}&nbsp; {label}</span>
        <div style="color:#ccc;font-size:13px;margin-top:4px;">{summary}</div>
        {act}
    </div>""", unsafe_allow_html=True)


def _plot_eq_balance(eq_recs):
    """Horizontal bar chart: deviation from ideal. Easier to read than raw values."""
    bands = list(BAND_FRIENDLY.keys())
    deviations, colors = [], []
    for band in bands:
        rec = eq_recs.get(band, "Neutral")
        if "Boost" in rec:
            deviations.append(-float(rec.split(":")[1].strip().replace(" dB", "")))
            colors.append("#4D96FF")
        elif "Cut" in rec:
            deviations.append( float(rec.split(":")[1].strip().replace(" dB", "")))
            colors.append("#FF6B6B")
        else:
            deviations.append(0.0)
            colors.append("#6BCB77")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    labels = [f"{BAND_FRIENDLY[b][0]}  {BAND_FRIENDLY[b][1]}" for b in bands]
    y = range(len(bands))
    ax.barh(y, deviations, color=colors, height=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.axvline(0, color="white", linewidth=1.5, alpha=0.5, zorder=4)
    ax.axvspan(-13, -2, alpha=0.08, color="#4D96FF")
    ax.axvspan( -2,  2, alpha=0.12, color="#6BCB77")
    ax.axvspan(  2, 13, alpha=0.08, color="#FF6B6B")
    ax.set_xlim(-13, 13)
    ax.set_xlabel("← Too Low   ·   Just Right   ·   Too High →", color="white", fontsize=10)
    ax.set_title("Frequency Balance", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.grid(axis="x", alpha=0.2, color="white")
    plt.tight_layout()
    return fig


# ── Landing page ───────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
    <span style="font-size:42px;">🎚️</span>
    <div>
        <h1 style="margin:0;padding:0;font-size:32px;">Mix Assistant</h1>
        <p style="margin:0;color:#888;font-size:15px;">
            Upload your track → get plain-English feedback on EQ, dynamics, and streaming loudness
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

uploaded_file = st.file_uploader(
    "Drop your track here — WAV, MP3, or FLAC",
    type=["wav", "mp3", "flac"],
    label_visibility="visible",
)

# ── No file: welcome screen ────────────────────────────────────────────────────

if uploaded_file is None:
    st.markdown("#### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **1️⃣  Upload your track**
        Drop a WAV, MP3, or FLAC file above.
        Any length works — even a rough draft.
        """)
    with c2:
        st.markdown("""
        **2️⃣  We analyse it**
        Frequency balance, dynamics, and
        loudness are measured automatically.
        """)
    with c3:
        st.markdown("""
        **3️⃣  Get actionable feedback**
        Plain-English explanations of what to fix
        and *why* — not just raw numbers.
        """)

    st.divider()
    st.markdown("#### What you'll get")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🎛️ **EQ & Tone**\nIs your mix bright, dark, muddy, or thin? We tell you which frequencies to adjust and what they sound like.")
    with c2:
        st.info("🔊 **Dynamics & Compression**\nDoes your track have punch and energy? We check if your mix is over-compressed or needs taming.")
    with c3:
        st.info("🎚️ **Streaming Readiness**\nWill your track sound quiet on Spotify? We compare against platform loudness targets and flag any issues.")

    st.stop()

# ── File uploaded: run analysis ────────────────────────────────────────────────

st.audio(uploaded_file)

with st.spinner("Analysing your track — this usually takes 15–30 seconds…"):
    y, sr         = load_audio(uploaded_file)
    info          = get_basic_info(y, sr)
    spectral      = get_spectral_features(y, sr)
    band_energies = analyze_frequency_bands(y, sr)
    eq_recs       = get_eq_recommendations(band_energies)
    dynamics      = analyze_dynamics(y, sr)
    comp_recs     = get_compression_recommendations(dynamics)
    loudness      = analyze_loudness(y, sr)
    master_recs   = get_mastering_recommendations(loudness)
    mix_report    = generate_mix_report(eq_recs, comp_recs, master_recs)

# Derive simple health statuses once
eq_status,   eq_summary   = _eq_health(eq_recs)
comp_status, comp_summary = _comp_health(comp_recs)
loud_status, loud_summary = _loud_health(master_recs)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_overview, tab_eq, tab_dynamics, tab_loudness, tab_technical = st.tabs([
    "🏆  Overview",
    "🎛️  EQ & Tone",
    "🔊  Dynamics",
    "🎚️  Loudness",
    "🔬  Technical",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    score = mix_report["score"]
    grade = mix_report["grade"]
    label = mix_report["label"]
    color = "#6BCB77" if score >= 80 else ("#FFD93D" if score >= 60 else "#FF6B6B")

    col_grade, col_health = st.columns([1, 2])

    # Grade card
    with col_grade:
        st.markdown(f"""
        <div style="text-align:center;padding:28px 16px;border-radius:14px;
                    background:{color}20;border:2px solid {color};">
            <div style="font-size:64px;font-weight:900;color:{color};line-height:1;">{grade}</div>
            <div style="font-size:28px;font-weight:700;color:{color};">{score}/100</div>
            <div style="font-size:13px;color:#aaa;margin-top:6px;">{label}</div>
        </div>
        <p style="text-align:center;color:#aaa;font-size:13px;margin-top:10px;">
            {mix_report['summary']}
        </p>
        """, unsafe_allow_html=True)

    # Quick health summary
    with col_health:
        st.markdown("#### Your mix at a glance")

        # Pick a one-liner action for each area
        eq_action   = "See the EQ & Tone tab for details"
        comp_action = "See the Dynamics tab for details"
        loud_action = "See the Loudness tab for details"

        _card("EQ & Tone",          eq_status,   eq_summary,   eq_action)
        _card("Dynamics",           comp_status, comp_summary, comp_action)
        _card("Streaming Loudness", loud_status, loud_summary, loud_action)

    st.divider()

    # Top priorities
    if mix_report["priorities"]:
        st.markdown("#### 🎯 Top priorities — do these first")
        for p in mix_report["priorities"]:
            badge = "#FF6B6B" if p["priority"] >= 4 else ("#FFD93D" if p["priority"] >= 3 else "#4D96FF")
            with st.expander(f"#{p['rank']}  {p['category']} — {p['action']}"):
                st.markdown(p["why"])

    # Strengths
    if mix_report["strengths"]:
        st.markdown("#### ✅ What's already working")
        for s in mix_report["strengths"]:
            st.success(s)

    # Basic track info (small, at the bottom of overview)
    st.divider()
    st.markdown("#### Track info")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", f"{info['Duration (s)']}s")
    c2.metric("Tempo",    f"{info['Tempo (BPM)']} BPM")
    c3.metric("Sample Rate", f"{info['Sample Rate (Hz)']} Hz")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EQ & TONE
# ══════════════════════════════════════════════════════════════════════════════

with tab_eq:
    st.markdown("### 🎛️ EQ & Tone")
    st.caption("Does your mix sound bright, dark, muddy, or thin? This section shows which frequency ranges are out of balance.")

    _card("EQ & Tone", eq_status, eq_summary)
    st.markdown("")

    # EQ balance chart — the main visual
    st.markdown("#### Frequency balance")
    st.caption("Bars pointing left = that range is too quiet. Bars pointing right = too loud. Green = just right.")
    st.pyplot(_plot_eq_balance(eq_recs))

    st.divider()

    # Band-by-band breakdown
    actionable = {b: r for b, r in eq_recs.items() if r != "Neutral"}
    neutral    = {b: r for b, r in eq_recs.items() if r == "Neutral"}

    if actionable:
        st.markdown("#### What to fix")
        for band, rec in actionable.items():
            fname, frange, fdesc = BAND_FRIENDLY[band]
            direction = "Boost" if "Boost" in rec else "Cut"
            db_val    = rec.split(":")[1].strip()
            icon      = "🔵" if direction == "Boost" else "🔴"

            with st.expander(f"{icon} **{fname}** ({frange}) — {direction} {db_val}"):
                st.caption(f"*{fdesc}*")
                st.markdown(get_eq_explanation(band, rec))

    if neutral:
        with st.expander(f"✅ {len(neutral)} band{'s' if len(neutral) > 1 else ''} that are already balanced"):
            for band in neutral:
                fname, frange, _ = BAND_FRIENDLY[band]
                st.success(f"✅ {fname} ({frange}) — no adjustment needed")

    st.divider()

    with st.expander("📈 Full frequency spectrum (advanced)"):
        st.caption("The detailed frequency content of your track across the full 20 Hz – 20 kHz range.")
        st.pyplot(plot_frequency_spectrum(y, sr))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

with tab_dynamics:
    st.markdown("### 🔊 Dynamics")
    st.caption("Does your track have energy, punch, and life — or does it sound flat and over-processed?")

    _card("Dynamics", comp_status, comp_summary)

    st.divider()

    # Compression recommendation in plain language
    assessment = comp_recs["compression_assessment"]
    ratio      = comp_recs["compression_ratio"]
    attack     = comp_recs["attack_time"]
    release    = comp_recs["release_time"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### What this means")
        st.markdown(get_compression_explanation(assessment))

    with col2:
        st.markdown("#### Recommended settings")
        if "Over-compressed" not in assessment:
            st.metric("Ratio",   ratio,  help="How aggressively the compressor reduces loud peaks")
            st.metric("Attack",  attack, help="How quickly the compressor reacts to a loud sound")
            st.metric("Release", release,help="How quickly the compressor lets go after the peak passes")
        else:
            st.info("No compression recommended — ease off what you already have.")

    st.divider()

    # Dynamic metrics in plain language
    st.markdown("#### Dynamic range metrics")
    st.caption("These numbers describe how much your track's volume varies over time.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Dynamic Range",
        comp_recs["dynamic_range"],
        help="The gap between the loudest and quietest moments (in dB). "
             "Below 6 dB = likely over-compressed. Above 20 dB = may need taming."
    )
    col2.metric(
        "Peak Level",
        f"{dynamics['peak_db']:.1f} dB",
        help="The single loudest moment in your track."
    )
    col3.metric(
        "Average Level",
        f"{dynamics['rms_db']:.1f} dB",
        help="The average loudness of your track over time (RMS)."
    )
    col4.metric(
        "Volume Variation",
        f"{dynamics['loudness_range']:.1f} dB",
        help="How much the loudness varies throughout the track. "
             "High variation = dramatic quiet/loud swings."
    )

    st.divider()

    with st.expander("📉 Dynamic range over time (advanced)"):
        st.caption("Blue = average loudness, orange = peak level. Large gaps mean high dynamic range.")
        st.pyplot(plot_dynamic_range(y, sr))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOUDNESS
# ══════════════════════════════════════════════════════════════════════════════

with tab_loudness:
    st.markdown("### 🎚️ Streaming Loudness")
    st.caption(
        "Streaming platforms automatically adjust track volume to a target level. "
        "If your track is too loud, they'll turn it down (often making it sound worse). "
        "If it's too quiet, it'll sound weak next to other tracks."
    )

    _card("Streaming Loudness", loud_status, loud_summary)

    st.divider()

    # Three key metrics with plain-language tooltips
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Integrated Loudness",
        f"{loudness['integrated_lufs']:.1f} LUFS",
        help="LUFS (Loudness Units Full Scale) is the standard way streaming platforms "
             "measure loudness. Spotify targets –14 LUFS. Lower = quieter."
    )
    col2.metric(
        "True Peak",
        f"{loudness['true_peak_dbtp']:.1f} dBTP",
        help="The maximum signal level including inter-sample peaks (spikes that happen "
             "between digital samples). Should stay below –1.0 dBTP to avoid distortion "
             "after MP3/AAC encoding."
    )
    col3.metric(
        "Loudness Range (LRA)",
        f"{loudness['lra']:.1f} LU",
        help="How much the loudness varies across the whole track. "
             "6–18 LU is a healthy range for most music."
    )

    st.divider()

    # Loudness explanation
    st.markdown("#### What to do")
    st.markdown(get_mastering_explanation(master_recs["loudness_status"]))

    if master_recs["loudness_status"] != "good_loudness":
        st.info(f"💡 Suggested adjustment: **{master_recs['gain_adjustment']}**")

    # True peak
    tp_status = master_recs["true_peak_status"]
    if tp_status == "true_peak_exceeded":
        st.error(f"⚠️ **True Peak issue:** {master_recs['true_peak_assessment']}")
        st.markdown(get_mastering_explanation("true_peak_exceeded"))
    else:
        st.success(f"✅ **True Peak:** {master_recs['true_peak_assessment']}")

    # LRA
    lra_status = master_recs["lra_status"]
    if lra_status != "lra_good":
        st.warning(f"⚠️ **Loudness Range:** {master_recs['lra_assessment']}")
        st.markdown(get_mastering_explanation(lra_status))
    else:
        st.success(f"✅ **Loudness Range:** {master_recs['lra_assessment']}")

    st.divider()

    # Per-platform readiness
    st.markdown("#### Platform readiness")
    lufs = loudness["integrated_lufs"]
    cols = st.columns(len(STREAMING_TARGETS))
    for col, (platform, target) in zip(cols, STREAMING_TARGETS.items()):
        diff = lufs - target
        if abs(diff) <= 1.5:
            col.success(f"✅\n\n**{platform}**\n\nTarget: {target} LUFS\n\nReady")
        elif diff < -1.5:
            col.warning(f"🔇\n\n**{platform}**\n\nTarget: {target} LUFS\n\nToo quiet ({diff:+.1f} LU)")
        else:
            col.error(f"⚠️\n\n**{platform}**\n\nTarget: {target} LUFS\n\nToo loud ({diff:+.1f} LU)")

    st.divider()

    with st.expander("📊 Loudness meter chart (advanced)"):
        st.pyplot(plot_loudness_vs_targets(loudness))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TECHNICAL
# ══════════════════════════════════════════════════════════════════════════════

with tab_technical:
    st.markdown("### 🔬 Technical Details")
    st.caption("The raw data behind your analysis. Useful if you want to dig deeper or share with an engineer.")

    # Track info
    st.markdown("#### Track info")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration",    f"{info['Duration (s)']}s")
    col2.metric("Tempo",       f"{info['Tempo (BPM)']} BPM")
    col3.metric("Sample Rate", f"{info['Sample Rate (Hz)']} Hz")
    col4.metric("RMS Energy",  f"{info['RMS Energy']:.4f}")

    st.divider()

    # Spectral features
    st.markdown("#### Spectral features")
    st.caption(
        "These describe the overall character of your frequency content. "
        "Centroid = average 'brightness' (higher = brighter). "
        "Bandwidth = how spread out the frequencies are. "
        "Rolloff = the point below which 85% of the energy sits."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Spectral Centroid",   f"{spectral['Spectral Centroid (Hz)']} Hz")
    col2.metric("Spectral Bandwidth",  f"{spectral['Spectral Bandwidth (Hz)']} Hz")
    col3.metric("Spectral Rolloff",    f"{spectral['Spectral Rolloff (Hz)']} Hz")

    st.divider()

    # Raw EQ band values
    st.markdown("#### Raw EQ band values")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Normalised energy per band (0 = silent, 1 = dominant)")
        for band, energy in band_energies.items():
            st.progress(float(energy), text=f"{band}: {energy:.3f}")
    with col2:
        st.caption("Raw recommendations")
        for band, rec in eq_recs.items():
            if "Boost" in rec:   st.success(f"🔺 {band}: {rec}")
            elif "Cut" in rec:   st.error(f"🔻 {band}: {rec}")
            else:                st.info(f"✅ {band}: {rec}")

    st.divider()

    # Waveform & spectrogram
    st.markdown("#### Waveform")
    st.pyplot(plot_waveform(y, sr))

    st.markdown("#### Mel spectrogram")
    st.caption("A visual map of your track's frequency content over time. Brighter = louder.")
    st.pyplot(plot_spectrogram(y, sr))

import streamlit as st
import sys
import os
import io
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_extraction import load_audio, get_basic_info, plot_waveform, get_spectral_features
from eq_model import (analyze_frequency_bands, get_eq_recommendations, get_ideal_energy,
                      plot_frequency_spectrum, get_eq_explanation)
from compression_model import (analyze_dynamics, get_compression_params,
                                get_compression_recommendations, plot_dynamic_range,
                                get_compression_explanation)
from mastering_model import (analyze_loudness, get_mastering_recommendations,
                              get_mastering_explanation, plot_loudness_vs_targets,
                              STREAMING_TARGETS)
from recommender import generate_mix_report
from audio_processor import (apply_eq, apply_compression, apply_limiter,
                              export_audio, plot_eq_curve, plot_compressor_curve)
from stem_separator import (separate_stems, analyze_stems, get_stem_insights,
                             plot_stem_waveforms, plot_stem_spectra,
                             combine_stems, get_preset_combinations)

st.set_page_config(page_title="Mix Assistant", page_icon="🎚️", layout="wide")

# ── Cached analysis ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_full_analysis(file_bytes: bytes) -> dict:
    buf           = io.BytesIO(file_bytes)
    y, sr         = load_audio(buf)
    info          = get_basic_info(y, sr)
    spectral      = get_spectral_features(y, sr)
    band_energies = analyze_frequency_bands(y, sr)
    ideal_energy  = get_ideal_energy(y, sr)
    eq_recs       = get_eq_recommendations(band_energies, ideal_energy=ideal_energy)
    dynamics      = analyze_dynamics(y, sr)
    ml_comp       = get_compression_params(y, sr)
    comp_recs     = get_compression_recommendations(dynamics, ml_params=ml_comp)
    loudness      = analyze_loudness(y, sr)
    master_recs   = get_mastering_recommendations(loudness)
    mix_report    = generate_mix_report(eq_recs, comp_recs, master_recs)
    orig_bytes    = export_audio(y, sr)
    return dict(
        y=y, sr=sr, info=info, spectral=spectral,
        band_energies=band_energies, eq_recs=eq_recs,
        dynamics=dynamics, ml_comp=ml_comp, comp_recs=comp_recs,
        loudness=loudness, master_recs=master_recs,
        mix_report=mix_report, orig_bytes=orig_bytes,
    )


@st.cache_data(show_spinner=False)
def decode_audio_bytes(audio_bytes: bytes):
    """Decode stored WAV bytes back to numpy — cached so step renders don't re-decode."""
    buf = io.BytesIO(audio_bytes)
    return load_audio(buf)


@st.cache_data(show_spinner=False)
def run_stem_separation(file_bytes: bytes):
    """Run Demucs stem separation — cached to avoid re-running the 2-5 min process."""
    buf = io.BytesIO(file_bytes)
    y, sr = load_audio(buf)
    stems = separate_stems(y, sr)
    if stems is None:
        return None
    stem_analyses = analyze_stems(stems, sr)
    insights      = get_stem_insights(stem_analyses)
    stem_bytes    = {name: export_audio(arr, sr) for name, arr in stems.items()}
    return {
        "stems":         stems,
        "sr":            sr,
        "stem_analyses": stem_analyses,
        "insights":      insights,
        "stem_bytes":    stem_bytes,
    }


# ── Constants ────────────────────────────────────────────────────────────────

BAND_FRIENDLY = {
    "Sub-bass":  ("Deep Bass",   "20–60 Hz",   "Chest-rumble: kicks, sub-bass synths"),
    "Bass":      ("Bass",        "60–250 Hz",  "Warmth and weight — kick body, bass guitar"),
    "Low-mid":   ("Low Mids",    "250–500 Hz", "Can sound 'boxy' if overdone"),
    "Mid":       ("Mids",        "500–2 kHz",  "Where vocals and most instruments live"),
    "Upper-mid": ("Upper Mids",  "2–6 kHz",    "Clarity and bite — helps things cut through"),
    "Presence":  ("Presence",    "6–12 kHz",   "Sparkle, sibilance, cymbal shimmer"),
    "Air":       ("Air",         "12–20 kHz",  "The open, polished top-end sheen"),
}

STEP_LABELS = ["Overview", "Stems",  "EQ & Tone", "Dynamics", "Loudness", "Export"]
STEP_ICONS  = ["🔍",       "🎵",     "🎛️",        "🔊",       "🎚️",       "📦"]

# ── Session state ────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "current_step":   1,
        "stems":          None,
        "active_stems":   ["drums", "bass", "vocals", "other"],
        "stem_volumes":   {"drums": 1.0, "bass": 1.0, "vocals": 1.0, "other": 1.0},
        "eq_gains":       {},
        "eq_audio":       None,
        "comp_params":    {},
        "comp_audio":     None,
        "final_audio":    None,
        "eq_suggested":   False,
        "comp_suggested": False,
        "uploaded_name":  None,
        "lim_ceiling":    -1.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_all():
    for k in ["current_step", "stems", "active_stems", "stem_volumes",
              "eq_gains", "eq_audio", "comp_params",
              "comp_audio", "final_audio", "eq_suggested", "comp_suggested", "lim_ceiling"]:
        if k in st.session_state:
            del st.session_state[k]
    for band in BAND_FRIENDLY:
        st.session_state.pop(f"eq_{band}", None)
    for k in ["comp_threshold", "comp_ratio", "comp_attack", "comp_release", "comp_makeup"]:
        st.session_state.pop(k, None)
    for s in ["drums", "bass", "vocals", "other"]:
        for prefix in ("mute_", "solo_", "vol_"):
            st.session_state.pop(f"{prefix}{s}", None)
    _init_state()


# ── Health helpers ───────────────────────────────────────────────────────────

def _eq_health(eq_recs):
    n = sum(1 for r in eq_recs.values() if r != "Neutral")
    if n == 0:  return "good",    "Frequency balance is excellent"
    if n <= 2:  return "warning", f"{n} frequency area{'s' if n > 1 else ''} need attention"
    return             "bad",     f"{n} of 7 frequency areas are out of balance"


def _comp_health(comp_recs):
    a = comp_recs["compression_assessment"]
    if "Over-compressed" in a: return "bad",     "Dynamics are too squashed — sounds flat"
    if "high peaks"      in a: return "warning",  "Loud spikes need taming"
    if "moderate peaks"  in a: return "warning",  "Light compression would help"
    return                            "good",     "Dynamics are well-balanced"


def _loud_health(master_recs):
    ls = master_recs["loudness_status"]
    tp = master_recs["true_peak_status"]
    lr = master_recs["lra_status"]
    if ls == "too_loud" or tp == "true_peak_exceeded":
        return "bad",    master_recs["loudness_assessment"]
    if ls == "too_quiet" or lr != "lra_good":
        return "warning", master_recs["loudness_assessment"]
    return "good", "Your track is streaming-ready"


# ── UI helpers ───────────────────────────────────────────────────────────────

def _card(label, status, summary, action=None):
    c = {"good": "#6BCB77", "warning": "#FFD93D", "bad": "#FF6B6B"}[status]
    i = {"good": "✅",       "warning": "⚠️",       "bad": "🔴"}[status]
    act = (f'<div style="margin-top:6px;font-size:13px;color:{c};font-weight:600;">'
           f'→ {action}</div>') if action else ""
    st.markdown(
        f'<div style="padding:16px 18px;border-radius:10px;border-left:5px solid {c};'
        f'background:{c}18;margin:6px 0;">'
        f'<span style="font-weight:700;font-size:15px;">{i}&nbsp; {label}</span>'
        f'<div style="color:#ccc;font-size:13px;margin-top:4px;">{summary}</div>'
        f'{act}</div>',
        unsafe_allow_html=True,
    )


def _parse_eq_gain(rec_str: str) -> float:
    """'Boost: 3.5 dB' → +3.5  |  'Cut: 2.1 dB' → -2.1  |  'Neutral' → 0.0"""
    if "Boost" in rec_str:
        return  float(rec_str.split(":")[1].strip().replace(" dB", ""))
    if "Cut"   in rec_str:
        return -float(rec_str.split(":")[1].strip().replace(" dB", ""))
    return 0.0


def _peak_rms(sig):
    peak = 20 * np.log10(max(float(np.max(np.abs(sig))), 1e-10))
    rms  = 20 * np.log10(max(float(np.sqrt(np.mean(sig ** 2))), 1e-10))
    return peak, rms


# ── Progress indicator ───────────────────────────────────────────────────────

def render_progress_indicator():
    step = st.session_state["current_step"]
    cols = st.columns(6)
    for i, (col, label, icon) in enumerate(zip(cols, STEP_LABELS, STEP_ICONS), start=1):
        with col:
            if i < step:
                if st.button(f"{icon} {label} ✓", key=f"step_btn_{i}",
                             use_container_width=True):
                    # Invalidate downstream state based on which step we're going back to
                    if i <= 2:
                        st.session_state["stems"]       = None
                        st.session_state["eq_audio"]    = None
                        st.session_state["comp_audio"]  = None
                        st.session_state["final_audio"] = None
                    elif i <= 3:
                        st.session_state["eq_audio"]    = None
                        st.session_state["comp_audio"]  = None
                        st.session_state["final_audio"] = None
                    elif i <= 4:
                        st.session_state["comp_audio"]  = None
                        st.session_state["final_audio"] = None
                    elif i <= 5:
                        st.session_state["final_audio"] = None
                    st.session_state["current_step"] = i
                    st.rerun()
            elif i == step:
                st.button(f"{icon} {label}", key=f"step_btn_{i}",
                          use_container_width=True, type="primary", disabled=True)
            else:
                st.button(f"{icon} {label}", key=f"step_btn_{i}",
                          use_container_width=True, disabled=True)
    st.divider()


# ── Step 1: Overview ─────────────────────────────────────────────────────────

def render_step1(R):
    mix_report  = R["mix_report"]
    info        = R["info"]
    eq_recs     = R["eq_recs"]
    comp_recs   = R["comp_recs"]
    master_recs = R["master_recs"]
    y, sr       = R["y"], R["sr"]

    score  = mix_report["score"]
    grade  = mix_report["grade"]
    label  = mix_report["label"]
    color  = "#6BCB77" if score >= 80 else ("#FFD93D" if score >= 60 else "#FF6B6B")

    eq_status,   eq_sum   = _eq_health(eq_recs)
    comp_status, comp_sum = _comp_health(comp_recs)
    loud_status, loud_sum = _loud_health(master_recs)

    st.markdown("## 🔍 Step 1 — Overview")
    st.caption("Here's what we found in your track. Hit Start Mixing to work through each area step by step.")

    # Track metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",    f"{info['Duration (s)']}s")
    c2.metric("Tempo",       f"{info['Tempo (BPM)']} BPM")
    c3.metric("Sample Rate", f"{info['Sample Rate (Hz)']} Hz")
    c4.metric("RMS Energy",  f"{info['RMS Energy']:.4f}")

    with st.expander("Waveform", expanded=False):
        st.pyplot(plot_waveform(y, sr))

    st.markdown("")

    # Score card + health cards
    col_grade, col_health = st.columns([1, 2])

    with col_grade:
        st.markdown(
            f'<div style="text-align:center;padding:28px 16px;border-radius:14px;'
            f'background:{color}20;border:2px solid {color};">'
            f'<div style="font-size:64px;font-weight:900;color:{color};line-height:1;">{grade}</div>'
            f'<div style="font-size:28px;font-weight:700;color:{color};">{score}/100</div>'
            f'<div style="font-size:13px;color:#aaa;margin-top:6px;">{label}</div>'
            f'</div>'
            f'<p style="text-align:center;color:#aaa;font-size:13px;margin-top:10px;">'
            f'{mix_report["summary"]}</p>',
            unsafe_allow_html=True,
        )

    with col_health:
        st.markdown("#### Your mix at a glance")
        _card("EQ & Tone",          eq_status,   eq_sum)
        _card("Dynamics",           comp_status, comp_sum)
        _card("Streaming Loudness", loud_status, loud_sum)

    # Top priorities
    if mix_report["priorities"]:
        st.markdown("#### 🎯 Top priorities — do these first")
        for p in mix_report["priorities"]:
            with st.expander(f"#{p['rank']}  {p['category']} — {p['action']}"):
                st.markdown(p["why"])

    if mix_report["strengths"]:
        st.markdown("#### ✅ What's already working")
        for s in mix_report["strengths"]:
            st.success(s)

    st.markdown("")
    if st.button("▶  Start Mixing", type="primary", use_container_width=True, key="start_mixing"):
        st.session_state["current_step"] = 2
        st.rerun()


# ── Step 2: Stem Analysis ────────────────────────────────────────────────────

def render_step_stems(file_bytes: bytes):
    st.markdown("## 🎵 Step 2 — Stem Analysis")
    st.caption(
        "Separate your track into drums, bass, vocals, and other to see how each element "
        "is balanced. Uses Demucs and may take 2–5 minutes on first run — result is cached."
    )

    col_skip, _ = st.columns([1, 3])
    with col_skip:
        if st.button("⏭ Skip Stem Analysis", key="stems_skip", use_container_width=True):
            st.session_state["current_step"] = 3
            st.rerun()

    st.markdown("")

    with st.spinner("Separating stems — this can take 2–5 minutes on first run…"):
        result = run_stem_separation(file_bytes)

    if result is None:
        st.error(
            "Stem separation failed. Demucs may not be installed or encountered an error. "
            "You can skip this step and continue with EQ."
        )
        if st.button("Continue to EQ & Tone →", type="primary",
                     key="stems_continue_err", use_container_width=True):
            st.session_state["current_step"] = 3
            st.rerun()
        return

    stems         = result["stems"]
    sr            = result["sr"]
    stem_analyses = result["stem_analyses"]
    insights      = result["insights"]
    stem_bytes    = result["stem_bytes"]

    st.session_state["stems"] = True

    # Insights
    if insights:
        st.markdown("#### 🔍 Key Insights")
        for insight in insights:
            st.info(insight)
    else:
        st.success("✅ All stems look well-balanced — no major EQ issues detected.")

    st.markdown("")

    tab_wave, tab_spec = st.tabs(["Waveforms", "Frequency Spectra"])
    with tab_wave:
        st.pyplot(plot_stem_waveforms(stems, sr))
    with tab_spec:
        st.pyplot(plot_stem_spectra(stems, sr))

    st.markdown("#### Per-Stem Details")
    stem_icons = {"drums": "🥁", "bass": "🎸", "vocals": "🎤", "other": "🎹"}
    for stem_name, analysis in stem_analyses.items():
        icon = stem_icons.get(stem_name, "🎵")
        with st.expander(f"{icon} {stem_name.capitalize()} — EQ & Dynamics", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**EQ Recommendations**")
                for band, rec in analysis["eq_recs"].items():
                    if "Boost" in rec:
                        st.markdown(f"🔵 **{band}**: {rec}")
                    elif "Cut" in rec:
                        st.markdown(f"🔴 **{band}**: {rec}")
                    else:
                        st.markdown(f"✅ **{band}**: {rec}")
            with c2:
                st.markdown("**Dynamics**")
                dyn = analysis["dynamics"]
                st.caption(f"Dynamic Range: {dyn['dynamic_range']:.1f} dB")
                st.caption(f"Peak: {dyn['peak_db']:.1f} dB")
                st.caption(f"RMS: {dyn['rms_db']:.1f} dB")
            if stem_name in stem_bytes:
                st.audio(stem_bytes[stem_name], format="audio/wav")

    # ── Stem Mixer ───────────────────────────────────────────────────────────
    st.subheader("🎛️ Stem Mixer")
    st.caption(
        "Mute or solo individual stems to isolate elements of your mix. "
        "Download any combination as WAV."
    )

    _MIXER_STEMS  = ["drums", "bass", "vocals", "other"]
    _MIXER_ICONS  = {"drums": "🥁", "bass": "🎸", "vocals": "🎤", "other": "🎹"}
    _MIXER_COLORS = {"drums": "#E74C3C", "bass": "#3498DB",
                     "vocals": "#2ECC71", "other": "#F39C12"}

    presets       = get_preset_combinations()
    preset_meta   = [
        ("Full Mix",     "🎵 Full Mix"),
        ("Drumless",     "🥁 Drumless"),
        ("Instrumental", "🎤 Instrumental"),
        ("Vocals Only",  "🎙️ Vocals Only"),
        ("Bass + Drums", "🎸 Bass + Drums"),
        ("Other Only",   "🎹 Other Only"),
    ]

    # Preset callbacks — run before next render, so widget keys can be set safely
    def _make_preset_cb(preset_stems):
        def _cb():
            for s in _MIXER_STEMS:
                st.session_state[f"mute_{s}"] = s not in preset_stems
                st.session_state[f"solo_{s}"] = False
                st.session_state[f"vol_{s}"]  = 100.0
            st.session_state["active_stems"]  = list(preset_stems)
            st.session_state["stem_volumes"]  = {s: 1.0 for s in _MIXER_STEMS}
        return _cb

    preset_cols = st.columns(6)
    for col, (pname, plabel) in zip(preset_cols, preset_meta):
        with col:
            safe_key = pname.lower().replace(" ", "_").replace("+", "and")
            st.button(plabel, key=f"preset_{safe_key}",
                      use_container_width=True,
                      on_click=_make_preset_cb(presets[pname]))

    # Per-stem controls
    st.markdown("")
    volumes: dict = {}
    for stem_name in _MIXER_STEMS:
        if stem_name not in stems:
            continue
        c_name, c_mute, c_solo, c_vol, c_color = st.columns([1, 1, 1, 3, 1])
        with c_name:
            st.markdown(
                f"<div style='padding-top:6px;font-weight:700;'>"
                f"{_MIXER_ICONS[stem_name]} {stem_name.capitalize()}</div>",
                unsafe_allow_html=True,
            )
        with c_mute:
            st.checkbox("Mute", key=f"mute_{stem_name}")
        with c_solo:
            st.checkbox("Solo", key=f"solo_{stem_name}")
        with c_vol:
            st.slider("", 0.0, 150.0, 100.0, 1.0,
                      key=f"vol_{stem_name}", format="%g%%")
        with c_color:
            clr = _MIXER_COLORS[stem_name]
            st.markdown(
                f'<div style="width:22px;height:22px;border-radius:4px;'
                f'background:{clr};margin-top:8px;"></div>',
                unsafe_allow_html=True,
            )
        volumes[stem_name] = st.session_state.get(f"vol_{stem_name}", 100.0) / 100.0

    # Solo logic: any solo → only soloed stems play; else respect mutes
    any_solo = any(st.session_state.get(f"solo_{s}", False)
                   for s in _MIXER_STEMS if s in stems)
    if any_solo:
        active_now = [s for s in _MIXER_STEMS
                      if s in stems and st.session_state.get(f"solo_{s}", False)]
    else:
        active_now = [s for s in _MIXER_STEMS
                      if s in stems and not st.session_state.get(f"mute_{s}", False)]

    # Status badge
    if len(active_now) >= 2:
        badge = "  ".join(f"{_MIXER_ICONS[s]} {s.capitalize()}" for s in active_now)
        st.success(f"Now playing: {badge}")
    elif len(active_now) == 1:
        s = active_now[0]
        st.info(f"Solo mode: {_MIXER_ICONS[s]} {s.capitalize()} only")
    else:
        st.warning("All stems muted — nothing to preview")

    # Combine + preview
    combined       = combine_stems(stems, active_now, volumes)
    combined_bytes = export_audio(combined, sr)

    st.markdown("**▶ Preview Current Combination**")
    st.audio(combined_bytes, format="audio/wav")

    # Smart filename
    _name_map = {
        tuple(sorted(["drums", "bass", "vocals", "other"])): "full_mix.wav",
        tuple(sorted(["bass",  "vocals", "other"])):          "drumless.wav",
        tuple(sorted(["drums", "bass",   "other"])):          "instrumental.wav",
        ("vocals",):                                          "vocals_only.wav",
        tuple(sorted(["drums", "bass"])):                     "bass_and_drums.wav",
        ("other",):                                           "other_only.wav",
    }
    dl_name = _name_map.get(tuple(sorted(active_now)), "custom_mix.wav")
    if not active_now:
        dl_name = "silence.wav"

    st.download_button(
        "⬇ Download This Combination as WAV",
        data=combined_bytes,
        file_name=dl_name,
        mime="audio/wav",
        use_container_width=True,
    )

    st.info(
        "🎧 **Learning tip:** Try muting the bass stem and listen to how the kick drum "
        "sounds without it. Then unmute and notice how they interact in the low frequencies. "
        "This is how mixing engineers use stem isolation to identify frequency masking "
        "and balance issues."
    )

    st.markdown("")
    if st.button("Continue to EQ & Tone →", type="primary",
                 key="stems_continue", use_container_width=True):
        st.session_state["current_step"] = 3
        st.rerun()


# ── Step 3: EQ & Tone ────────────────────────────────────────────────────────

def render_step2(R):
    eq_recs    = R["eq_recs"]
    y, sr      = R["y"], R["sr"]
    orig_bytes = R["orig_bytes"]

    st.markdown("## 🎛️ Step 3 — EQ & Tone")
    st.caption(
        "Adjust the frequency balance of your track. "
        "Sliders are pre-loaded with ML-derived suggestions — tweak as you like."
    )

    # Load ML suggestions into slider session state on first visit
    if not st.session_state["eq_suggested"]:
        for band, rec in eq_recs.items():
            gain = max(-12.0, min(12.0, _parse_eq_gain(rec)))
            st.session_state[f"eq_{band}"] = gain
        st.session_state["eq_suggested"] = True

    col_left, col_right = st.columns([1, 1])

    # ── Left: analysis ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Frequency Spectrum")
        st.pyplot(plot_frequency_spectrum(y, sr))

        st.markdown("#### Band Recommendations")
        for band, (fname, frange, fdesc) in BAND_FRIENDLY.items():
            rec = eq_recs.get(band, "Neutral")
            if "Boost" in rec:
                icon, clr = "🔵", "#4D96FF"
            elif "Cut" in rec:
                icon, clr = "🔴", "#FF6B6B"
            else:
                icon, clr = "✅", "#6BCB77"
            st.markdown(
                f'<div style="padding:6px 12px;border-radius:6px;border-left:3px solid {clr};'
                f'background:{clr}18;margin:3px 0;">'
                f'<span style="font-weight:600;">{icon} {fname}</span> '
                f'<span style="color:#888;font-size:12px;">{frange}</span> — '
                f'<span style="color:{clr};">{rec}</span></div>',
                unsafe_allow_html=True,
            )
            with st.expander(f"Why? — {fname}", expanded=False):
                st.caption(fdesc)
                st.markdown(get_eq_explanation(band, rec))

    # ── Right: sliders + curve + before/after ─────────────────────────────
    with col_right:
        st.markdown("#### EQ Sliders")
        st.caption("Drag right to boost, left to cut. 0 dB = no change.")

        band_gains = {}
        eq_cols = st.columns(7)
        for col, (band, (fname, frange, _)) in zip(eq_cols, BAND_FRIENDLY.items()):
            with col:
                gain = st.slider(
                    fname, -12.0, 12.0,
                    value=float(st.session_state.get(f"eq_{band}", 0.0)),
                    step=0.5, key=f"eq_{band}",
                    label_visibility="collapsed",
                    help=f"{fname} ({frange})",
                )
                band_gains[band] = gain
                clr = "#FF8C42" if gain > 0.1 else ("#4D96FF" if gain < -0.1 else "#888")
                st.markdown(
                    f"<div style='text-align:center;font-size:11px;font-weight:700;color:{clr};'>"
                    f"{gain:+.1f}</div>"
                    f"<div style='text-align:center;font-size:9px;color:#666;'>{fname}</div>",
                    unsafe_allow_html=True,
                )

        # Live EQ curve
        st.pyplot(plot_eq_curve(band_gains, sr))

        # Live before/after preview
        st.markdown("#### Before / After")
        y_eq       = apply_eq(y, sr, band_gains)
        eq_preview = export_audio(y_eq, sr)
        orig_peak, orig_rms = _peak_rms(y)
        eq_peak,   eq_rms   = _peak_rms(y_eq)

        ba1, ba2 = st.columns(2)
        with ba1:
            st.caption("**Original**")
            st.audio(orig_bytes, format="audio/wav")
            st.caption(f"Peak {orig_peak:.1f} dB · RMS {orig_rms:.1f} dB")
        with ba2:
            st.caption("**EQ Preview**")
            st.audio(eq_preview, format="audio/wav")
            st.caption(f"Peak {eq_peak:.1f} dB · RMS {eq_rms:.1f} dB")

        # on_click callbacks run before the next render — safe to set widget keys there
        def _reset_eq_cb():
            for band in BAND_FRIENDLY:
                st.session_state[f"eq_{band}"] = 0.0

        def _load_ml_eq_cb():
            for band, rec in eq_recs.items():
                st.session_state[f"eq_{band}"] = max(-12.0, min(12.0, _parse_eq_gain(rec)))

        # Action buttons
        st.markdown("")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.button("↺ Flat", key="eq_reset_flat", use_container_width=True,
                      on_click=_reset_eq_cb)
        with b2:
            st.button("💡 ML Suggest", key="eq_load_ml", use_container_width=True,
                      on_click=_load_ml_eq_cb)
        with b3:
            if st.button("Apply & Continue →", type="primary", key="eq_apply",
                         use_container_width=True):
                gains = {band: float(st.session_state.get(f"eq_{band}", 0.0))
                         for band in BAND_FRIENDLY}
                y_applied = apply_eq(y, sr, gains)
                st.session_state["eq_gains"]    = gains
                st.session_state["eq_audio"]    = export_audio(y_applied, sr)
                st.session_state["comp_audio"]  = None
                st.session_state["final_audio"] = None
                st.session_state["current_step"] = 4
                st.rerun()


# ── Step 3: Dynamics & Compression ──────────────────────────────────────────

def render_step3(R):
    dynamics   = R["dynamics"]
    comp_recs  = R["comp_recs"]
    ml_comp    = R["ml_comp"]
    y_orig, sr = R["y"], R["sr"]
    orig_bytes = R["orig_bytes"]

    # Use EQ-processed audio as source if available
    eq_audio_bytes = st.session_state.get("eq_audio")
    if eq_audio_bytes:
        y_src, sr_src = decode_audio_bytes(eq_audio_bytes)
        src_bytes = eq_audio_bytes
    else:
        y_src, sr_src = y_orig, sr
        src_bytes = orig_bytes

    st.markdown("## 🔊 Step 4 — Dynamics & Compression")
    st.caption(
        "Control how much the loudest moments stand out from the quieter ones. "
        "Sliders are pre-loaded with ML-derived settings."
    )

    # Load ML suggestions on first visit to step 3
    if not st.session_state["comp_suggested"]:
        st.session_state["comp_threshold"] = float(np.clip(ml_comp.get("threshold_db", -20.0), -60.0, 0.0))
        st.session_state["comp_ratio"]     = float(np.clip(ml_comp.get("ratio",        2.0),   1.0, 20.0))
        st.session_state["comp_attack"]    = float(np.clip(ml_comp.get("attack_ms",    50.0),  1.0, 200.0))
        st.session_state["comp_release"]   = float(np.clip(ml_comp.get("release_ms",   200.0), 10.0, 2000.0))
        st.session_state["comp_makeup"]    = 0.0
        st.session_state["comp_suggested"] = True

    col_left, col_right = st.columns([1, 1])

    # ── Left: analysis ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Dynamic Range Over Time")
        st.pyplot(plot_dynamic_range(y_src, sr_src))

        comp_status, comp_sum = _comp_health(comp_recs)
        _card("Dynamics Assessment", comp_status, comp_sum)

        st.markdown("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Dynamic Range", f"{dynamics['dynamic_range']:.1f} dB",
                  help="Gap between loudest and quietest moments. < 6 dB = over-compressed.")
        m2.metric("Peak Level",    f"{dynamics['peak_db']:.1f} dB")
        m3.metric("RMS Level",     f"{dynamics['rms_db']:.1f} dB")

        st.markdown("#### What this means")
        st.info(get_compression_explanation(comp_recs["compression_assessment"]))

    # ── Right: sliders + curve + before/after ────────────────────────────────
    with col_right:
        st.markdown("#### Compressor Settings")

        comp_threshold = st.slider(
            "Threshold", -60.0, 0.0,
            value=float(st.session_state.get("comp_threshold", -20.0)),
            step=1.0, key="comp_threshold", format="%.0f dB",
            help="Level at which compression starts. Lower = more compression.",
        )
        comp_ratio = st.slider(
            "Ratio", 1.0, 20.0,
            value=float(st.session_state.get("comp_ratio", 2.0)),
            step=0.5, key="comp_ratio", format="%.1f:1",
            help="1:1 = off. 2:1 = gentle. 4:1 = moderate. 8:1+ = heavy.",
        )
        comp_attack = st.slider(
            "Attack", 1.0, 200.0,
            value=float(st.session_state.get("comp_attack", 50.0)),
            step=1.0, key="comp_attack", format="%.0f ms",
            help="How quickly the compressor reacts. Fast = tighter. Slow = lets punch through.",
        )
        comp_release = st.slider(
            "Release", 10.0, 2000.0,
            value=float(st.session_state.get("comp_release", 200.0)),
            step=10.0, key="comp_release", format="%.0f ms",
            help="How quickly the compressor lets go after a peak. Slow = smoother.",
        )
        comp_makeup = st.slider(
            "Makeup Gain", 0.0, 20.0,
            value=float(st.session_state.get("comp_makeup", 0.0)),
            step=0.5, key="comp_makeup", format="%.1f dB",
            help="Boost the compressed signal to restore perceived loudness.",
        )

        # Live transfer curve
        st.pyplot(plot_compressor_curve(comp_threshold, comp_ratio, comp_makeup))

        # Live before/after preview
        st.markdown("#### Before / After")
        y_comp       = apply_compression(y_src, sr_src, comp_threshold, comp_ratio,
                                         comp_attack, comp_release, comp_makeup)
        comp_preview = export_audio(y_comp, sr_src)
        src_peak,  src_rms  = _peak_rms(y_src)
        comp_peak, comp_rms = _peak_rms(y_comp)

        ba1, ba2 = st.columns(2)
        with ba1:
            st.caption("**Before Compression**")
            st.audio(src_bytes, format="audio/wav")
            st.caption(f"Peak {src_peak:.1f} · RMS {src_rms:.1f} dB")
        with ba2:
            st.caption("**Compressed Preview**")
            st.audio(comp_preview, format="audio/wav")
            st.caption(f"Peak {comp_peak:.1f} · RMS {comp_rms:.1f} dB")

        # on_click callbacks run before the next render — safe to set widget keys there
        def _reset_comp_cb():
            st.session_state["comp_threshold"] = -20.0
            st.session_state["comp_ratio"]     = 1.0
            st.session_state["comp_attack"]    = 50.0
            st.session_state["comp_release"]   = 200.0
            st.session_state["comp_makeup"]    = 0.0

        def _load_ml_comp_cb():
            st.session_state["comp_threshold"] = float(np.clip(ml_comp.get("threshold_db", -20.0), -60.0, 0.0))
            st.session_state["comp_ratio"]     = float(np.clip(ml_comp.get("ratio",        2.0),   1.0, 20.0))
            st.session_state["comp_attack"]    = float(np.clip(ml_comp.get("attack_ms",    50.0),  1.0, 200.0))
            st.session_state["comp_release"]   = float(np.clip(ml_comp.get("release_ms",   200.0), 10.0, 2000.0))

        # Action buttons
        st.markdown("")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.button("↺ Reset", key="comp_reset", use_container_width=True,
                      on_click=_reset_comp_cb)
        with b2:
            st.button("💡 ML Values", key="comp_load_ml", use_container_width=True,
                      on_click=_load_ml_comp_cb)
        with b3:
            if st.button("Apply & Continue →", type="primary", key="comp_apply",
                         use_container_width=True):
                y_applied = apply_compression(y_src, sr_src, comp_threshold, comp_ratio,
                                              comp_attack, comp_release, comp_makeup)
                st.session_state["comp_params"] = {
                    "threshold_db": comp_threshold, "ratio": comp_ratio,
                    "attack_ms": comp_attack, "release_ms": comp_release,
                    "makeup_db": comp_makeup,
                }
                st.session_state["comp_audio"]  = export_audio(y_applied, sr_src)
                st.session_state["final_audio"] = None
                st.session_state["current_step"] = 5
                st.rerun()


# ── Step 4: Loudness & Mastering ─────────────────────────────────────────────

def render_step4(R):
    loudness    = R["loudness"]
    master_recs = R["master_recs"]
    y_orig, sr  = R["y"], R["sr"]
    orig_bytes  = R["orig_bytes"]

    comp_audio_bytes = st.session_state.get("comp_audio")
    if comp_audio_bytes:
        y_src, sr_src = decode_audio_bytes(comp_audio_bytes)
        src_bytes = comp_audio_bytes
    else:
        y_src, sr_src = y_orig, sr
        src_bytes = orig_bytes

    st.markdown("## 🎚️ Step 5 — Loudness & Mastering")
    st.caption(
        "Set the output ceiling and check streaming readiness. "
        "The limiter is always the last step — it catches peaks before export."
    )

    col_left, col_right = st.columns([1, 1])

    # ── Left: analysis ───────────────────────────────────────────────────────
    with col_left:
        st.markdown("#### Loudness Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Integrated LUFS", f"{loudness['integrated_lufs']:.1f}",
                  help="Spotify targets −14 LUFS. Lower = quieter.")
        m2.metric("True Peak",       f"{loudness['true_peak_dbtp']:.1f} dBTP",
                  help="Must stay below −1.0 dBTP to avoid clipping after MP3/AAC encoding.")
        m3.metric("LRA",             f"{loudness['lra']:.1f} LU",
                  help="Loudness Range. 6–18 LU is healthy for most music.")

        loud_status, loud_sum = _loud_health(master_recs)
        _card("Streaming Loudness", loud_status, loud_sum)

        st.markdown("#### Platform Readiness")
        lufs = loudness["integrated_lufs"]
        for platform, target in STREAMING_TARGETS.items():
            diff = lufs - target
            if abs(diff) <= 1.5:
                st.success(f"✅ **{platform}** — ready (target {target} LUFS)")
            elif diff < -1.5:
                st.warning(f"🔇 **{platform}** — {abs(diff):.1f} LU too quiet (target {target} LUFS)")
            else:
                st.error(f"⚠️ **{platform}** — {diff:.1f} LU too loud (target {target} LUFS)")

        st.markdown("#### What to do")
        st.info(get_mastering_explanation(master_recs["loudness_status"]))
        if master_recs["loudness_status"] != "good_loudness":
            st.caption(f"Suggested gain adjustment: **{master_recs['gain_adjustment']}**")

        if master_recs["true_peak_status"] == "true_peak_exceeded":
            st.error(f"⚠️ True Peak: {master_recs['true_peak_assessment']}")
        else:
            st.success(f"✅ True Peak: {master_recs['true_peak_assessment']}")

        with st.expander("📊 Loudness chart", expanded=False):
            st.pyplot(plot_loudness_vs_targets(loudness))

    # ── Right: limiter + before/after ────────────────────────────────────────
    with col_right:
        st.markdown("#### Output Limiter")
        st.caption("−1.0 dBTP is the streaming standard — prevents clipping after encoding.")

        lim_ceiling = st.slider(
            "Ceiling", -6.0, 0.0,
            value=float(st.session_state.get("lim_ceiling", -1.0)),
            step=0.1, key="lim_ceiling", format="%.1f dBTP",
            help="Absolute peak level of the output. −1.0 dBTP is the industry standard.",
        )

        platform_choice = st.selectbox(
            "Target platform",
            list(STREAMING_TARGETS.keys()),
            key="target_platform",
        )
        target_lufs = STREAMING_TARGETS[platform_choice]
        st.caption(f"**{platform_choice} target:** {target_lufs} LUFS  ·  "
                   f"**Your track:** {loudness['integrated_lufs']:.1f} LUFS")

        # Live preview
        y_limited     = apply_limiter(y_src, lim_ceiling)
        limited_bytes = export_audio(y_limited, sr_src)
        src_peak, _  = _peak_rms(y_src)
        lim_peak, _  = _peak_rms(y_limited)

        st.markdown(f"**Peak after limiter:** {lim_peak:.1f} dBTP")

        st.markdown("#### Before / After")
        ba1, ba2 = st.columns(2)
        with ba1:
            st.caption("**Before Limiter**")
            st.audio(src_bytes, format="audio/wav")
            st.caption(f"Peak {src_peak:.1f} dB")
        with ba2:
            st.caption("**Limited Preview**")
            st.audio(limited_bytes, format="audio/wav")
            st.caption(f"Peak {lim_peak:.1f} dBTP")

        st.markdown("")
        if st.button("Apply & Continue →", type="primary", key="lim_apply",
                     use_container_width=True):
            y_final = np.clip(apply_limiter(y_src, lim_ceiling), -1.0, 1.0)
            st.session_state["final_audio"] = export_audio(y_final, sr_src)
            st.session_state["current_step"] = 6
            st.rerun()


# ── Step 6: Export & Summary ─────────────────────────────────────────────────

def render_step5(R, filename_stem: str):
    orig_bytes  = R["orig_bytes"]
    mix_report  = R["mix_report"]

    final_audio = st.session_state.get("final_audio")
    if final_audio is None:
        st.warning("No processed audio found — go back and apply each step before exporting.")
        return

    st.markdown("## 📦 Step 6 — Export & Summary")

    # ── Before / After ────────────────────────────────────────────────────────
    st.markdown("### Before vs After")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("**Original**")
        st.audio(orig_bytes, format="audio/wav")
    with col2:
        st.caption("**Processed (EQ + Compression + Limiter)**")
        st.audio(final_audio, format="audio/wav")

    st.divider()

    # ── Processing chain summary ─────────────────────────────────────────────
    st.markdown("### Processing Chain")
    eq_gains    = st.session_state.get("eq_gains", {})
    comp_params = st.session_state.get("comp_params", {})
    lim_ceiling = st.session_state.get("lim_ceiling", -1.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🎛️ EQ**")
        if eq_gains:
            for band, gain in eq_gains.items():
                fname = BAND_FRIENDLY[band][0]
                if abs(gain) > 0.1:
                    arrow = "↑" if gain > 0 else "↓"
                    st.caption(f"{arrow} {fname}: {gain:+.1f} dB")
                else:
                    st.caption(f"→ {fname}: flat")
        else:
            st.caption("No EQ applied")

    with c2:
        st.markdown("**🔊 Compression**")
        if comp_params:
            st.caption(f"Threshold: {comp_params['threshold_db']:.0f} dB")
            st.caption(f"Ratio: {comp_params['ratio']:.1f}:1")
            st.caption(f"Attack: {comp_params['attack_ms']:.0f} ms")
            st.caption(f"Release: {comp_params['release_ms']:.0f} ms")
            st.caption(f"Makeup: +{comp_params.get('makeup_db', 0):.1f} dB")
        else:
            st.caption("No compression applied")

    with c3:
        st.markdown("**🔒 Limiter**")
        st.caption(f"Ceiling: {lim_ceiling:.1f} dBTP")

    st.divider()

    # ── What you learned ─────────────────────────────────────────────────────
    st.markdown("### What You Learned")

    strengths  = mix_report.get("strengths", [])
    priorities = mix_report.get("priorities", [])

    if strengths:
        st.markdown("**Strengths in your original mix:**")
        for s in strengths:
            st.success(s)

    if priorities:
        st.markdown("**Key areas that needed work:**")
        for p in priorities:
            st.info(f"**{p['category']}** — {p['action']}: {p['why']}")

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        "⬇️  Download Processed WAV",
        final_audio,
        file_name=f"{filename_stem}_processed.wav",
        mime="audio/wav",
        use_container_width=True,
        type="primary",
    )
    st.caption("Downloads as 16-bit PCM WAV at the original sample rate.")

    st.markdown("")
    if st.button("🔄 Start over with a new mix", key="start_over", use_container_width=True):
        _reset_all()
        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    _init_state()

    st.markdown(
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">'
        '<span style="font-size:42px;">🎚️</span>'
        '<div>'
        '<h1 style="margin:0;padding:0;font-size:32px;">Mix Assistant</h1>'
        '<p style="margin:0;color:#888;font-size:15px;">'
        'Upload your track and get guided through EQ, compression, and mastering — step by step.'
        '</p></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Drop your track here — WAV, MP3, or FLAC",
        type=["wav", "mp3", "flac"],
        label_visibility="visible",
    )

    if uploaded_file is None:
        st.markdown("#### How it works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                "**1️⃣  Upload your track**\n\n"
                "Drop a WAV, MP3, or FLAC file above. Any length works — even a rough draft."
            )
        with c2:
            st.markdown(
                "**2️⃣  We analyse it**\n\n"
                "Frequency balance, dynamics, and loudness are measured automatically."
            )
        with c3:
            st.markdown(
                "**3️⃣  Work through each step**\n\n"
                "EQ → Compression → Mastering. Each step explains what to do and why."
            )
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("🎛️ **EQ & Tone**\nIs your mix bright, dark, muddy, or thin? We show you which frequencies to adjust.")
        with c2:
            st.info("🔊 **Dynamics & Compression**\nDoes your track have punch, or does it sound flat and over-processed?")
        with c3:
            st.info("🎚️ **Streaming Readiness**\nWill it sound right on Spotify and Apple Music? We check loudness targets.")
        st.stop()

    # Detect new file — reset everything
    if uploaded_file.name != st.session_state.get("uploaded_name"):
        _reset_all()
        st.session_state["uploaded_name"] = uploaded_file.name

    # Run full analysis (cached after first load)
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    with st.spinner("Analysing your track — this takes 15–30 seconds on first load…"):
        R = run_full_analysis(file_bytes)

    # Progress indicator with back-navigation
    render_progress_indicator()

    # Route to current step
    step          = st.session_state["current_step"]
    filename_stem = uploaded_file.name.rsplit(".", 1)[0]

    if   step == 1: render_step1(R)
    elif step == 2: render_step_stems(file_bytes)
    elif step == 3: render_step2(R)
    elif step == 4: render_step3(R)
    elif step == 5: render_step4(R)
    elif step == 6: render_step5(R, filename_stem)


main()

import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_extraction import (
    load_audio, get_basic_info, plot_waveform,
    plot_spectrogram, get_spectral_features
)
from eq_model import analyze_frequency_bands, get_eq_recommendations, plot_frequency_spectrum, get_eq_explanation
from compression_model import analyze_dynamics, get_compression_recommendations, plot_dynamic_range, get_compression_explanation
from mastering_model import analyze_loudness, get_mastering_recommendations, get_mastering_explanation, plot_loudness_vs_targets

st.set_page_config(
    page_title="Music Mixing & Mastering Assistant",
    page_icon="🎚️",
    layout="wide"
)

st.title("🎚️ Music Mixing & Mastering Assistant")
st.subheader("Upload your track and get professional mixing and mastering recommendations")

st.divider()

uploaded_file = st.file_uploader(
    "Upload an audio file to get started",
    type=["wav", "mp3", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    with st.spinner("Analyzing your track..."):
        y, sr = load_audio(uploaded_file)
        info = get_basic_info(y, sr)
        spectral = get_spectral_features(y, sr)

    st.divider()
    st.subheader("📊 Track Analysis")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{info['Duration (s)']}s")
    col2.metric("Tempo", f"{info['Tempo (BPM)']} BPM")
    col3.metric("Sample Rate", f"{info['Sample Rate (Hz)']} Hz")
    col4.metric("RMS Energy", str(info['RMS Energy']))

    st.divider()
    st.subheader("🌊 Waveform")
    st.pyplot(plot_waveform(y, sr))

    st.divider()
    st.subheader("🎨 Mel Spectrogram")
    st.pyplot(plot_spectrogram(y, sr))

    st.divider()
    st.subheader("📡 Spectral Features")
    col1, col2, col3 = st.columns(3)
    col1.metric("Centroid", f"{spectral['Spectral Centroid (Hz)']} Hz")
    col2.metric("Bandwidth", f"{spectral['Spectral Bandwidth (Hz)']} Hz")
    col3.metric("Rolloff", f"{spectral['Spectral Rolloff (Hz)']} Hz")

    st.divider()
    st.subheader("🎛️ EQ Analysis")

    with st.spinner("Running EQ analysis..."):
        band_energies = analyze_frequency_bands(y, sr)
        recommendations = get_eq_recommendations(band_energies)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Frequency Band Energy**")
        for band, energy in band_energies.items():
            st.progress(float(abs(energy)), text=f"{band}: {energy:.2f}")

    with col2:
        st.markdown("**EQ Recommendations**")
        for band, rec in recommendations.items():
            if "Boost" in rec:
                st.success(f"🔺 {band}: {rec}")
            elif "Cut" in rec:
                st.error(f"🔻 {band}: {rec}")
            else:
                st.info(f"✅ {band}: {rec}")

    actionable_bands = {b: r for b, r in recommendations.items() if r != "Neutral"}
    if actionable_bands:
        st.markdown("**What this means for your mix:**")
        for band, rec in actionable_bands.items():
            icon = "🔺" if "Boost" in rec else "🔻"
            with st.expander(f"{icon} {band} — {rec}"):
                st.write(get_eq_explanation(band, rec))

    st.divider()
    st.subheader("📈 Frequency Spectrum")
    st.pyplot(plot_frequency_spectrum(y, sr))

    st.divider()
    st.subheader("🔊 Compression Analysis")

    with st.spinner("Analyzing dynamics..."):
        dynamics = analyze_dynamics(y, sr)
        compression_recs = get_compression_recommendations(dynamics)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dynamic Range", compression_recs["dynamic_range"])
    col2.metric("Crest Factor", compression_recs["crest_factor"])
    col3.metric("Attack Time", compression_recs["attack_time"])
    col4.metric("Release Time", compression_recs["release_time"])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        assessment = compression_recs["compression_assessment"]
        if "Over-compressed" in assessment:
            st.error(f"⚠️ {assessment}")
        elif "Needs compression" in assessment:
            st.warning(f"🔶 {assessment}")
        else:
            st.success(f"✅ {assessment}")
        st.metric("Recommended Ratio", compression_recs["compression_ratio"])
        st.markdown(get_compression_explanation(assessment))

    with col2:
        st.markdown("**Dynamic Metrics**")
        st.write(f"Peak Level: `{dynamics['peak_db']:.1f} dB`")
        st.write(f"RMS Level: `{dynamics['rms_db']:.1f} dB`")
        st.write(f"Loudness Range: `{dynamics['loudness_range']:.1f} dB`")
        st.write(f"Mean Loudness: `{dynamics['mean_loudness']:.1f} dB`")

    st.subheader("📉 Dynamic Range Over Time")
    st.pyplot(plot_dynamic_range(y, sr))

    st.divider()
    st.subheader("🎚️ Mastering & Loudness")

    with st.spinner("Analyzing loudness..."):
        loudness = analyze_loudness(y, sr)
        mastering_recs = get_mastering_recommendations(loudness)

    col1, col2, col3 = st.columns(3)
    col1.metric("Integrated Loudness", f"{loudness['integrated_lufs']} LUFS")
    col2.metric("True Peak", f"{loudness['true_peak_dbtp']} dBTP")
    col3.metric("Loudness Range (LRA)", f"{loudness['lra']} LU")

    st.divider()

    # Loudness assessment
    loudness_status = mastering_recs["loudness_status"]
    if loudness_status == "too_loud":
        st.error(f"⚠️ {mastering_recs['loudness_assessment']}")
    elif loudness_status == "too_quiet":
        st.warning(f"🔇 {mastering_recs['loudness_assessment']}")
    else:
        st.success(f"✅ {mastering_recs['loudness_assessment']}")
    st.caption(f"Suggested gain adjustment: {mastering_recs['gain_adjustment']}")
    st.markdown(get_mastering_explanation(loudness_status))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**True Peak**")
        tp_status = mastering_recs["true_peak_status"]
        if tp_status == "true_peak_exceeded":
            st.error(f"⚠️ {mastering_recs['true_peak_assessment']}")
        else:
            st.success(f"✅ {mastering_recs['true_peak_assessment']}")
        st.markdown(get_mastering_explanation(tp_status))

    with col2:
        st.markdown("**Loudness Range**")
        lra_status = mastering_recs["lra_status"]
        if lra_status == "lra_good":
            st.success(f"✅ {mastering_recs['lra_assessment']}")
        else:
            st.warning(f"🔶 {mastering_recs['lra_assessment']}")
        st.markdown(get_mastering_explanation(lra_status))

    st.subheader("📊 Loudness vs. Streaming Targets")
    st.pyplot(plot_loudness_vs_targets(loudness))

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎛️ **EQ & Frequency**\nAnalyze and balance your frequency spectrum")
    with col2:
        st.info("🔊 **Compression**\nControl dynamics and punch in your mix")
    with col3:
        st.info("📊 **Mastering**\nLoudness normalization and final polish")
    st.caption("Supported formats: WAV, MP3, FLAC")



    
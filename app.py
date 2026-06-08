import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from feature_extraction import (
    load_audio, get_basic_info, plot_waveform,
    plot_spectrogram, get_spectral_features
)

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

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎛️ **EQ & Frequency**\nAnalyze and balance your frequency spectrum")
    with col2:
        st.info("🔊 **Compression**\nControl dynamics and punch in your mix")
    with col3:
        st.info("📊 **Mastering**\nLoudness normalization and final polish")
    st.caption("Supported formats: WAV, MP3, FLAC")
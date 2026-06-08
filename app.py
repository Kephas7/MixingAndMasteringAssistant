import streamlit as st

st.set_page_config(
    page_title="Music Mixing & Mastering Assistant",
    page_icon="🎚️",
    layout="wide"
)

st.title("🎚️ Music Mixing & Mastering Assistant")
st.subheader("Upload your track and get professional mixing and mastering recommendations")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🎛️ **EQ & Frequency**\nAnalyze and balance your frequency spectrum")

with col2:
    st.info("🔊 **Compression**\nControl dynamics and punch in your mix")

with col3:
    st.info("📊 **Mastering**\nLoudness normalization and final polish")

st.divider()

uploaded_file = st.file_uploader("Upload an audio file to get started", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    st.success("File uploaded successfully! Analysis coming in Sprint 2.")
else:
    st.caption("Supported formats: WAV, MP3, FLAC")
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io

def load_audio(file):
    audio_bytes = file.read()
    audio_buffer = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_buffer, sr=None)
    return y, sr

def get_basic_info(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = float(np.atleast_1d(tempo)[0])
    rms = float(np.sqrt(np.mean(y**2)))
    return {
        "Duration (s)": round(duration, 2),
        "Sample Rate (Hz)": sr,
        "Tempo (BPM)": round(tempo_value, 1),
        "RMS Energy": round(rms, 4)
    }

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1DB954")
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time",
                                    y_axis="mel", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    plt.tight_layout()
    return fig

def get_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)

def get_spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return {
        "Spectral Centroid (Hz)": round(float(np.mean(centroid)), 2),
        "Spectral Bandwidth (Hz)": round(float(np.mean(bandwidth)), 2),
        "Spectral Rolloff (Hz)": round(float(np.mean(rolloff)), 2)
    }
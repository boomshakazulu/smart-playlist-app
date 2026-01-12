import argparse
import json
import math
from typing import Dict, Any

import librosa
import numpy as np


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def summarize_feature(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values.size else 0.0,
        "std": float(np.std(values)) if values.size else 0.0,
        "min": float(np.min(values)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
    }


def analyze_audio(path: str, sr: int = 22050) -> Dict[str, Any]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Harmonic / percussive split
    y_harm, y_perc = librosa.effects.hpss(y)

    # Core rhythm + energy
    # Use percussive component for more reliable tempo
    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if tempo.size else 0.0
    tempo = float(tempo) if tempo else 0.0

    # Robust tempo estimate using tempo candidates
    try:
        tempo_candidates = librosa.beat.tempo(y=y_perc, sr=sr, aggregate=None)
        if tempo_candidates is not None and len(tempo_candidates) > 0:
            tempo = float(np.median(tempo_candidates))
    except Exception:
        pass

    # Clamp to a sane range to avoid half/double errors
    if tempo > 200:
        tempo = tempo / 2.0
    if tempo < 60:
        tempo = tempo * 2.0
    tempo_strength = float(len(beat_frames) / max(1.0, duration)) if duration else 0.0

    rms = librosa.feature.rms(y=y)[0]
    rms_stats = summarize_feature(rms)
    rms_mean = rms_stats["mean"]
    energy = clamp(rms_mean * 10.0, 0.0, 1.0)

    # Harmonic / percussive split
    rms_harm = float(np.mean(librosa.feature.rms(y=y_harm))) if y_harm.size else 0.0
    rms_perc = float(np.mean(librosa.feature.rms(y=y_perc))) if y_perc.size else 0.0
    perc_ratio = clamp(rms_perc / (rms_harm + rms_perc + 1e-9), 0.0, 1.0)

    # Spectral descriptors
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    centroid_stats = summarize_feature(centroid)
    flatness_stats = summarize_feature(flatness)
    zcr_stats = summarize_feature(zcr)

    # Harmonic / timbre
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    chroma_mean = np.mean(np.max(chroma, axis=0)) if chroma.size else 0.0
    chroma_avg = np.mean(chroma, axis=1) if chroma.size else np.zeros(12)
    key_index = int(np.argmax(chroma_avg)) if chroma_avg.size else 0
    key_strength = float(chroma_avg[key_index] / (np.sum(chroma_avg) + 1e-9))
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_name = key_names[key_index]

    key_h = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_profile = np.mean(key_h, axis=1) if key_h.size else np.zeros(12)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_score = float(np.dot(key_profile, major_profile))
    minor_score = float(np.dot(key_profile, minor_profile))
    mode = "major" if major_score >= minor_score else "minor"
    mfcc_mean = np.mean(mfcc, axis=1).tolist() if mfcc.size else []
    mfcc_std = np.std(mfcc, axis=1).tolist() if mfcc.size else []

    brightness = clamp(centroid_stats["mean"] / (sr / 2.0), 0.0, 1.0)
    flatness_mean = flatness_stats["mean"]
    zcr_mean = clamp(zcr_stats["mean"] * 10.0, 0.0, 1.0)

    valence = clamp(
        0.5 * brightness + 0.3 * (1.0 - flatness_mean) + 0.2 * chroma_mean,
        0.0,
        1.0,
    )

    loudness = 20.0 * math.log10(rms_mean + 1e-9)

    return {
        "tempo": float(tempo),
        "energy": float(energy),
        "valence": float(valence),
        "duration": duration,
        "loudness": float(loudness),
        "rms": rms_stats,
        "spectral_centroid": centroid_stats,
        "spectral_bandwidth": summarize_feature(bandwidth),
        "spectral_rolloff": summarize_feature(rolloff),
        "spectral_flatness": flatness_stats,
        "spectral_contrast": summarize_feature(contrast),
        "zero_crossing_rate": zcr_stats,
        "chroma_peak": float(chroma_mean),
        "key": key_name,
        "mode": mode,
        "key_strength": float(key_strength),
        "tempo_strength": float(tempo_strength),
        "mfcc_mean": [float(x) for x in mfcc_mean],
        "mfcc_std": [float(x) for x in mfcc_std],
        "percussive_ratio": float(perc_ratio),
        "sample_rate": int(sr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    features = analyze_audio(args.path)
    print(json.dumps(features))


if __name__ == "__main__":
    main()

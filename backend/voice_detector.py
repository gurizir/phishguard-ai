import librosa
import numpy as np


def detect_ai_voice(audio_path):

    # Load audio
    y, sr = librosa.load(audio_path)

    # Pitch variation
    pitch = librosa.yin(y, fmin=50, fmax=300, sr=sr)
    pitch_var = np.var(pitch)

    # Loudness variation
    rms = librosa.feature.rms(y=y)[0]
    rms_var = np.var(rms)

    # Spectral flatness (noise vs natural voice)
    spectral = librosa.feature.spectral_flatness(y=y)[0]
    flatness = np.mean(spectral)

    # Heuristic score (0 to 3)
    score = 0

    if pitch_var < 10:
        score += 1

    if rms_var < 0.001:
        score += 1

    if flatness > 0.2:
        score += 1


    # Convert score to confidence (0.0 → 1.0)
    confidence = score / 3


    # Return dynamic probabilities
    if confidence >= 0.5:

        return [
            {"label": "fake", "score": round(confidence, 2)},
            {"label": "real", "score": round(1 - confidence, 2)}
        ]

    else:

        return [
            {"label": "real", "score": round(1 - confidence, 2)},
            {"label": "fake", "score": round(confidence, 2)}
        ]
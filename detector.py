import librosa
import numpy as np


def extract_features(file):
    """
    Extract audio features matching train_model.py exactly.
    """
    try:
        audio, sr = librosa.load(file, sr=22050, duration=5.0)

        if audio is None or len(audio) == 0:
            raise ValueError("Audio file is empty or corrupted")

        target_len = 22050 * 5
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        audio = audio[:target_len]

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std  = np.std(mfcc.T,  axis=0)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_mean = np.mean(mel.T, axis=0)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff  = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

        features = np.hstack((
            mfcc_mean,        # 40
            mfcc_std,         # 40
            chroma_mean,      # 12
            mel_mean,         # 128
            [zcr, spectral_centroid, spectral_rolloff]  # 3
        ))
        return features

    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")


def get_ai_reasoning(features, prob_fake):
    """
    Calibrated heuristic analysis tuned for real microphone recordings.
    Thresholds reflect typical live-capture acoustic profiles.
    """
    reasons = []

    # MFCC std across coefficients 40-79 — mic recordings have higher variance
    # Old threshold (< 0.7) was wrong — mic audio naturally has std of 2–15+
    mfcc_std_mean = np.mean(features[40:80])
    if mfcc_std_mean < 1.5:
        reasons.append("[!] Unusually low MFCC variance — possible AI monotone artifact")

    # Spectral rolloff — index 222
    # Old threshold (< 2000 Hz) was far too low; mic recordings are typically 3000–8000 Hz
    if len(features) > 222 and features[222] < 1500:
        reasons.append("[!] Compressed frequency response — bandwidth narrower than expected")

    # ZCR — index 220
    # Old threshold (< 0.05) flagged normal speech; typical ZCR is 0.02–0.15
    if len(features) > 220 and features[220] < 0.02:
        reasons.append("[!] Abnormally low zero-crossing rate — atypical phoneme transitions")

    # Spectral centroid — index 221
    # Very low centroid (< 500 Hz) suggests muffled or synthetic output
    if len(features) > 221 and features[221] < 500:
        reasons.append("[!] Spectral centroid below natural speech range")

    # If model confidence is borderline, note it
    if 0.45 <= prob_fake <= 0.65:
        reasons.append("[~] Confidence in borderline range — result may reflect audio quality")

    if not reasons:
        reasons.append("[OK] Voice displays natural organic variance and noise profile")

    return reasons


def predict(file, model, scaler):
    """
    Predict probability and return AI reasoning.
    Returns (prob_fake: float, reasoning: list) or raises RuntimeError.
    """
    try:
        features = extract_features(file)
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
        prob_fake = float(model.predict_proba(features_scaled)[0][1])

        reasoning = get_ai_reasoning(features, prob_fake)
        return prob_fake, reasoning

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
# updated

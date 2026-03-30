import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import io
from PIL import Image

def plot_feature_chart(file):
    """
    Creates a bar chart using sr=22050 consistency.
    """
    try:
        # Load like train_model.py
        audio, sr = librosa.load(file, sr=22050, res_type='kaiser_fast', duration=5.0)

        # Basic normalization for chart consistency
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        # Feature scores (0-100)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_score = float(np.clip(np.mean(np.abs(mfcc)) * 2, 0, 100))

        # Proxy for pitch stability
        try:
            cens = librosa.feature.spectral_centroid(y=audio, sr=sr)
            pitch_score = float(np.clip(np.std(cens) / 4, 0, 100))
        except: pitch_score = 0.0

        flux = float(np.clip(np.mean(librosa.onset.onset_strength(y=audio, sr=sr)) * 10, 0, 100))
        chroma = float(np.clip(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr)) * 100, 0, 100))
        zcr = float(np.clip(np.mean(librosa.feature.zero_crossing_rate(y=audio)) * 1000, 0, 100))

        features = ['MFCC', 'Pitch\nStability', 'Spectral\nFlux', 'Chroma\nPattern', 'Zero\nCrossing']
        scores   = [mfcc_score, pitch_score, flux, chroma, zcr]
        colors   = ['#7c3aed' if s > 50 else '#06b6d4' for s in scores]

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0f0f2e')
        ax.set_facecolor('#0f0f2e')

        bars = ax.bar(features, scores, color=colors, width=0.5, edgecolor='none')
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f'{score:.0f}', ha='center', va='bottom', color='white', fontweight='bold')

        ax.set_ylim(0, 115)
        ax.set_title('Feature Analysis Breakdown', color='white', fontweight='bold', pad=10)
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_yticks([0, 25, 50, 75, 100])

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f0f2e')
        plt.close()
        buf.seek(0)
        return np.array(Image.open(buf))

    except Exception:
        plt.close('all')
        blank = np.zeros((400, 800, 3), dtype=np.uint8)
        blank[:, :] = [15, 15, 46]
        return blank# updated

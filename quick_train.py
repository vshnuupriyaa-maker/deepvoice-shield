import os
import numpy as np
import librosa
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

REAL_FOLDER = "dataset/real"
FAKE_FOLDER = "dataset/fake"

# ── Feature extraction — MUST match detector.py exactly (223 features) ──
def extract_features(file):
    audio, sr = librosa.load(file, sr=22050, duration=5.0)

    target_len = 22050 * 5
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)   # 40
    mfcc_std  = np.std(mfcc.T,  axis=0)   # 40

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)  # 12

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_mean = np.mean(mel.T, axis=0)  # 128

    zcr               = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_rolloff  = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    return np.hstack((
        mfcc_mean, mfcc_std, chroma_mean, mel_mean,
        [zcr, spectral_centroid, spectral_rolloff]
    ))  # total = 40+40+12+128+3 = 223


# ── Load dataset — accepts BOTH .wav and .mp3 ──
AUDIO_EXTS = ('.wav', '.mp3', '.flac', '.ogg')

X, y = [], []

print("Loading REAL voices...")
for file in os.listdir(REAL_FOLDER):
    if file.lower().endswith(AUDIO_EXTS):
        path = os.path.join(REAL_FOLDER, file)
        try:
            X.append(extract_features(path))
            y.append(0)
            print(f"  ✅ REAL: {file}")
        except Exception as e:
            print(f"  ❌ Skipped {file}: {e}")

print("\nLoading FAKE voices...")
for file in os.listdir(FAKE_FOLDER):
    if file.lower().endswith(AUDIO_EXTS):
        path = os.path.join(FAKE_FOLDER, file)
        try:
            X.append(extract_features(path))
            y.append(1)
            print(f"  ✅ FAKE: {file}")
        except Exception as e:
            print(f"  ❌ Skipped {file}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\n📊 Dataset: {len(y)} samples | Real: {sum(y==0)} | Fake: {sum(y==1)}")

if sum(y==0) == 0 or sum(y==1) == 0:
    raise ValueError("❌ Need at least 1 real AND 1 fake sample. Check your dataset folders.")

# ── Scale ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train ──
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

if len(y) >= 6:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Test accuracy: {acc*100:.1f}%")
else:
    model.fit(X_scaled, y)
    acc = accuracy_score(y, model.predict(X_scaled))
    print(f"⚠ Small dataset — train accuracy: {acc*100:.1f}%")

# ── Save as SEPARATE files (matches app.py / detector.py) ──
os.makedirs("models", exist_ok=True)
joblib.dump(model,  "models/model.pkl")   # ← model only
joblib.dump(scaler, "models/scaler.pkl")  # ← scaler only

# Save accuracy for Command Center display
with open("models/meta.txt", "w") as f:
    f.write(f"accuracy: {acc*100:.1f}%\n")
    f.write(f"real: {sum(y==0)}\n")
    f.write(f"fake: {sum(y==1)}\n")

print("\n✅ model.pkl, scaler.pkl, meta.txt saved to models/")
print("   Restart app.py to load the new model.")
# updated

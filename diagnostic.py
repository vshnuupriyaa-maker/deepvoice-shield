
import os
import joblib
import numpy as np
import librosa
from detector import predict, extract_features
from visualizer import plot_spectrogram
from charts import plot_feature_chart

# ── Dynamic Model Loading ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

print("--- Diagnostic Start ---")
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model, scaler = None, None

sample_file = os.path.join(BASE_DIR, 'samples', 'real_voice.wav')
if not os.path.exists(sample_file):
    print(f"❌ Sample file not found: {sample_file}")
else:
    print(f"Testing with: {sample_file}")
    try:
        print("1. Running prediction...")
        prob_fake, reasoning = predict(sample_file, model, scaler)
        print(f"✅ Prediction: {prob_fake}, Reasoning: {reasoning}")
        
        print("2. Running spectrogram...")
        spec = plot_spectrogram(sample_file)
        print(f"✅ Spectrogram shape: {spec.shape}")
        
        print("3. Running feature chart...")
        chart = plot_feature_chart(sample_file)
        print(f"✅ Chart shape: {chart.shape}")
        
        print("4. Testing librosa.load at 16k...")
        y, sr = librosa.load(sample_file, sr=16000, duration=5.0)
        print(f"✅ Loaded audio: {len(y)} samples at {sr}Hz")
        
    except Exception as e:
        print(f"❌ Caught exception: {e}")
        import traceback
        traceback.print_exc()

print("--- Diagnostic End ---")
# updated

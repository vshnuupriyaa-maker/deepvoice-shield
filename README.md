[DeepVoiceShield_README.md](https://github.com/user-attachments/files/29243130/DeepVoiceShield_README.md)
# 🛡️ DeepVoice Shield Pro — Forensic Voice Authentication Engine

A real-time **deepfake voice detection system** powered by XGBoost and acoustic feature engineering. Analyses 223 acoustic biomarkers — MFCCs, spectral flux, pitch entropy, harmonic ratios — to classify any voice as **human** or **AI-synthesised** in under 3 seconds. Built with Python, Gradio, and Librosa.

> **Current Model Accuracy: 86.7%** (trained on 33 real + 41 fake samples with augmentation)

---

## ✨ Features

### 🏠 Command Center
- Live system status — model operational / untrained, sample counts, accuracy
- Detection pipeline overview (4-stage: Ingest → Extract → Classify → Report)
- Verdict classification guide and threat landscape context

### 🔬 Forensic Lab (Single File Analysis)
- **Upload or Record** — drag & drop audio file or capture live via microphone
- **Adjustable Sensitivity** — threshold slider from 0.3 to 0.8
- **Verdict Display** — animated GENUINE HUMAN ✓ or DEEPFAKE DETECTED ⚠ banner with confidence score
- **AI Forensic Observation** — plain-language explanation of detected artifacts (MFCC variance, spectral rolloff, ZCR, centroid)
- **Spectrogram View** — Mel spectrogram visualization of the audio
- **Feature Breakdown Chart** — bar chart of extracted acoustic features
- **Stats Panel** — Duration, Pitch/Magnitude (Hz), AI Confidence %, Threat level
- **PDF Report Download** — downloadable forensic report (falls back to .txt if fpdf2 unavailable)
- **Analysis History** — running table of all files analyzed in the session

### 🗂️ Batch Audit
- Upload multiple audio files at once
- Each file scored as **REAL / SUSPICIOUS / FAKE** based on threshold
- Results table with filename, verdict, confidence, and timestamp

---

## 🧠 How It Works

```
Audio File (.wav / .mp3)
        │
        ▼
Feature Extraction (223 features)
 ├── MFCC mean + std        (80 features)
 ├── Chroma STFT mean       (12 features)
 ├── Mel spectrogram mean   (128 features)
 └── ZCR + Centroid + Rolloff (3 features)
        │
        ▼
StandardScaler normalization
        │
        ▼
XGBoost Classifier
 └── predict_proba → synthetic probability score
        │
        ▼
Verdict + Forensic Reasoning + Report
```

**AI Forensic Reasoning flags:**
- Low MFCC variance (possible AI monotone artifact)
- Compressed frequency response (bandwidth narrower than natural speech)
- Abnormally low zero-crossing rate (atypical phoneme transitions)
- Spectral centroid below natural speech range
- Borderline confidence note (0.45–0.65 range)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3 | Core language |
| Gradio | Web UI framework |
| Librosa | Audio feature extraction |
| XGBoost | Classification model |
| Scikit-learn | Preprocessing & metrics |
| NumPy | Numerical computation |
| Matplotlib | Spectrogram & chart generation |
| fpdf2 | PDF report generation |
| Joblib | Model serialization |

---

## 📁 Project Structure

```
deepvoice-shield-main/
├── app.py                    # Gradio app — UI, tabs, analysis pipeline
├── detector.py               # Feature extraction + XGBoost prediction + forensic reasoning
├── train_model.py            # Model training with augmentation — saves model.pkl & scaler.pkl
├── quick_train.py            # Quick retrain shortcut
├── visualizer.py             # Mel spectrogram plot generation
├── charts.py                 # Feature breakdown bar chart
├── reporter.py               # PDF/TXT report generator
├── diagnostic.py             # Diagnostic tool for debugging audio issues
├── record_samples.py         # Microphone sample recording utility
├── download_real_voices.py   # Script to download real voice samples
├── requirements.txt          # Python dependencies
├── models/
│   ├── model.pkl             # Trained XGBoost classifier
│   ├── scaler.pkl            # StandardScaler (fitted on training data)
│   └── meta.txt              # Training metadata (accuracy, sample counts)
├── dataset/
│   ├── real/                 # Real human voice samples (.wav)
│   │   └── 33 samples (LibriSpeech + recorded)
│   └── fake/                 # AI-generated voice samples (.mp3)
│       └── 41 samples (ElevenLabs + AIVocal)
├── samples/
│   ├── real_voice.wav        # Sample real voice for quick testing
│   └── fake_voice.wav        # Sample fake voice for quick testing
└── assets/
    └── banner.png            # App banner image
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/vshnuupriyaa-maker/deepvoice-shield.git

# 2. Navigate into the project folder
cd deepvoice-shield-main

# 3. Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python train_model.py
```

This reads all audio from `dataset/real/` and `dataset/fake/`, applies augmentation (pitch shift, time stretch, noise injection, gain scaling — up to 9 variants per file), trains the XGBoost model, and saves `models/model.pkl`, `models/scaler.pkl`, and `models/meta.txt`.

### Run the App

```bash
python app.py
```

The Gradio app launches locally and also generates a public shareable link (`share=True`).

---

## 📊 Dataset

| Source | Type | Count |
|---|---|---|
| LibriSpeech (61-70968) | Real human voice | 11 files |
| Recorded samples (real_1–20) | Real human voice | 22 files |
| ElevenLabs (Roger voice) | AI-generated | 31 files |
| AIVocal | AI-generated | 10 files |

**After augmentation:** 33 real → ~297 variants · 41 fake → ~369 variants

> 💡 For best model accuracy, add 20+ real and 20+ fake audio files to `dataset/real/` and `dataset/fake/` before training.

---

## 🔊 Supported Audio Formats

WAV · MP3 · FLAC — analysed at **22,050 Hz**, 5-second window (padded if shorter)

---

## ⚙️ Augmentation Strategy

Each training sample is expanded into up to 9 variants:

| Technique | Effect |
|---|---|
| Original | Baseline |
| Pitch shift +1.5 semitones | Higher voice |
| Pitch shift −1.5 semitones | Lower voice |
| Time stretch ×1.1 | Faster speech |
| Time stretch ×0.9 | Slower speech |
| Mild white noise (σ=0.003) | Noisy environment |
| Stronger white noise (σ=0.008) | Noisier environment |
| Gain ×0.7 | Quieter recording |
| Gain ×1.3 (clipped) | Louder recording |

---

## 👩‍💻 Built By

**Arava Vishnu Priya Gopi**  
GitHub: [@vshnuupriyaa-maker](https://github.com/vshnuupriyaa-maker)

import gradio as gr
import joblib
import numpy as np
import os
import librosa
import datetime
import tempfile
import traceback
import sys
from detector import predict, extract_features
from visualizer import plot_spectrogram
from charts import plot_feature_chart
from reporter import generate_pdf_report

# ── Dynamic Model Loading ──
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m_path = os.path.join(base_dir, 'models', 'model.pkl')
    s_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    try:
        m = joblib.load(m_path)
        s = joblib.load(s_path)
        print("✅ Models reloaded successfully")
        return m, s
    except Exception:
        print("⚠ Models not found. Please train/calibrate first.")
        return None, None

model, scaler = load_models()

from reporter import HAS_FPDF
if not HAS_FPDF:
    print("WARNING: 'fpdf2' not found. Reports will be in .txt format.")

css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap');
:root{--primary:#8b5cf6;--secondary:#06b6d4;--accent:#f43f5e;--bg:#030712;--card:rgba(17,24,39,0.7);}
*{box-sizing:border-box;transition:all 0.25s cubic-bezier(0.4,0,0.2,1);}
body,.gradio-container{background:radial-gradient(circle at top right,#0f172a,#030712)!important;font-family:'Outfit',sans-serif!important;color:#f8fafc!important;}
.main-header{text-align:center;padding:2.5rem 0 1.5rem;}
.main-header h1{font-size:3.2rem;font-weight:800;margin:0;background:linear-gradient(to right,#a78bfa,#22d3ee,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-2px;}
.glass-card{background:var(--card)!important;backdrop-filter:blur(16px) saturate(180%);border:1px solid rgba(255,255,255,0.1)!important;border-radius:20px!important;box-shadow:0 8px 32px 0 rgba(0,0,0,0.8)!important;padding:1.5rem!important;}
.result-display{border-radius:20px;padding:2rem;margin-bottom:20px;border:2px solid rgba(255,255,255,0.1);box-shadow:0 4px 20px rgba(0,0,0,0.4);text-align:center;backdrop-filter:blur(10px);}
.result-genuine{background:linear-gradient(145deg,rgba(34,197,94,0.3),rgba(21,128,61,0.4))!important;border-color:#22c55e!important;color:#4ade80!important;box-shadow:0 0 30px rgba(34,197,94,0.3)!important;animation:pulse-glow-green 2s infinite ease-in-out!important;}
@keyframes pulse-glow-green{50%{box-shadow:0 0 50px rgba(34,197,94,0.5)!important;transform:scale(1.01);}}
.result-fake{background:linear-gradient(145deg,rgba(220,38,38,0.5),rgba(153,27,27,0.6))!important;border-color:#ef4444!important;color:white!important;box-shadow:0 0 40px rgba(220,38,38,0.5)!important;text-transform:uppercase;letter-spacing:2px;animation:pulse-glow-red 1.5s infinite ease-in-out!important;}
@keyframes pulse-glow-red{50%{box-shadow:0 0 60px rgba(220,38,38,0.7)!important;transform:scale(1.02);}}
.verdict-text{font-size:2.2rem;font-weight:800;margin-bottom:5px;}
.confidence-bar{font-family:'JetBrains Mono',monospace;font-size:1rem;opacity:0.9;font-weight:600;}
.status-badge{display:inline-block;padding:3px 10px;border-radius:100px;font-size:0.7rem;font-weight:800;text-transform:uppercase;margin-bottom:8px;border:1px solid currentColor;}
.primary-btn{background:linear-gradient(135deg,#7c3aed,#0891b2)!important;border:none!important;border-radius:12px!important;font-weight:700!important;color:white!important;box-shadow:0 4px 15px rgba(124,58,237,0.3)!important;padding:10px 20px!important;}
.primary-btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(124,58,237,0.5)!important;}
.info-card{background:rgba(6,182,212,0.05)!important;border:1px solid rgba(6,182,212,0.2)!important;border-radius:12px!important;padding:12px!important;}
.info-card label{color:#22d3ee!important;font-weight:800!important;text-transform:uppercase!important;font-size:0.75rem!important;letter-spacing:0.5px!important;}
.info-card input{background:rgba(0,0,0,0.2)!important;border:1px solid rgba(255,255,255,0.05)!important;color:#f8fafc!important;font-family:'JetBrains Mono',monospace!important;}
.obs-box{background:linear-gradient(135deg,rgba(6,182,212,0.1),rgba(139,92,246,0.1))!important;padding:20px;border-radius:16px;margin-top:20px;border:1px solid rgba(6,182,212,0.3)!important;color:#e2e8f0!important;}
.obs-box b{color:#22d3ee;font-size:1.1rem;text-transform:uppercase;letter-spacing:1px;}
.doc-box{background:transparent!important;border:none!important;margin-top:10px;}
.doc-box .gr-box,.doc-box .gr-form{background:transparent!important;border:none!important;}
.doc-box [data-testid="block-label"]{display:none!important;}
.doc-box .file-preview{background:rgba(255,255,255,0.05)!important;border:1px dashed rgba(255,255,255,0.2)!important;border-radius:12px!important;}
.history-table{border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.05);}
.viz-container{border-radius:16px;overflow:hidden;margin-top:10px;border:1px solid rgba(255,255,255,0.1);}
"""

# ─────────────────────────────────────────────
# CORE ANALYSIS
# ─────────────────────────────────────────────

def _run_analysis(audio_path, threshold, history):
    if history is None:
        history = []
    if audio_path is None or not os.path.exists(str(audio_path)):
        return ("<div class='verdict-text' style='color:#94a3b8'>AWAITING SAMPLE...</div>",
                None, None, "—", "—", "—", "—", history, None, "")
    try:
        global model, scaler
        if model is None:
            model, scaler = load_models()
        if model is None:
            return ("<div class='verdict-text' style='color:#f43f5e'>ERROR: Model Not Found — run train_model.py first</div>",
                    None, None, "—", "—", "—", "—", history, None, "")

        prob_fake, reasoning = predict(audio_path, model, scaler)
        prob_real = 1.0 - prob_fake

        if prob_fake >= threshold:
            label, icon, cls = "DEEPFAKE DETECTED", "⚠", "result-fake"
        else:
            label, icon, cls = "GENUINE HUMAN",     "✓", "result-genuine"

        verdict_html = f"""
        <div class="result-display {cls}">
            <div class="status-badge">{label}</div>
            <div class="verdict-text">{icon} {label}</div>
            <div class="confidence-bar">{prob_fake*100:.1f}% AI SYNTHETIC &bull; {prob_real*100:.1f}% HUMAN ORGANIC</div>
        </div>"""

        reasoning_html = (
            "<div class='obs-box'><b>AI Forensic Observation:</b><br>"
            + "<br>".join(reasoning) + "</div>"
        )

        spec_img   = plot_spectrogram(audio_path)
        feat_chart = plot_feature_chart(audio_path)

        y, sr = librosa.load(audio_path, sr=22050, duration=5.0)
        duration = f"{len(y)/sr:.2f}s"
        try:
            centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_pitch = f"{int(np.nanmean(centroid))} Hz"
        except Exception:
            avg_pitch = "—"

        ts    = datetime.datetime.now().strftime('%H%M%S')
        r_pdf = os.path.join(tempfile.gettempdir(), f"Report_{ts}.pdf")
        r_txt = os.path.join(tempfile.gettempdir(), f"Report_{ts}.txt")
        try:
            generate_pdf_report(os.path.basename(audio_path), prob_fake, label,
                                duration, avg_pitch, reasoning, r_pdf)
        except Exception:
            pass

        if os.path.exists(r_pdf):
            final_report = r_pdf
        else:
            with open(r_txt, "w", encoding="utf-8") as f:
                f.write(f"DEEPVOICE SHIELD REPORT\nResult: {label}\nConf: {prob_fake*100:.1f}%\n")
            final_report = r_txt

        if hasattr(history, 'values'):
            history = history.values.tolist()
        history = [[os.path.basename(audio_path), label,
                    f"{prob_fake*100:.1f}%",
                    datetime.datetime.now().strftime("%H:%M")]] + list(history)

        return (verdict_html, spec_img, feat_chart,
                duration, avg_pitch, f"{prob_fake*100:.1f}%", label,
                history, final_report, reasoning_html)

    except Exception as e:
        traceback.print_exc()
        if hasattr(history, 'values'):
            history = history.values.tolist()
        return (f"<div class='verdict-text' style='color:#f43f5e'>ANALYSIS FAILED: {str(e)}</div>",
                None, None, "—", "—", "—", "—", history, None, "")


def analyze_voice(audio_input, threshold=0.5, history=None):
    return _run_analysis(audio_input, threshold, history)


def batch_analyze(files, threshold=0.5):
    results = []
    if not files:
        return results
    for f in files:
        path = f.name if hasattr(f, 'name') else str(f)
        try:
            prob, _ = predict(path, model, scaler)
            v = ("FAKE" if prob >= threshold + 0.25
                 else ("SUSPICIOUS" if prob >= threshold else "REAL"))
            results.append([os.path.basename(path), v,
                             f"{prob*100:.1f}%",
                             datetime.datetime.now().strftime("%H:%M")])
        except Exception as e:
            results.append([os.path.basename(path), "ERROR", str(e)[:30], "—"])
    return results


# ─────────────────────────────────────────────
# COMMAND CENTER
# ─────────────────────────────────────────────

def get_live_stats():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_ok = os.path.exists(os.path.join(base_dir, 'models', 'model.pkl'))

    samples_dir = os.path.join(base_dir, 'samples')
    n_samples = fake_count = real_count = 0
    if os.path.exists(samples_dir):
        for fname in os.listdir(samples_dir):
            if fname.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                n_samples += 1
                if 'fake' in fname.lower(): fake_count += 1
                elif 'real' in fname.lower(): real_count += 1

    acc_str = "—"
    meta_path = os.path.join(base_dir, 'models', 'meta.txt')
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                for line in f:
                    if "accuracy" in line.lower():
                        acc_str = line.split(":")[-1].strip()
                        break
        except Exception:
            pass

    return model_ok, n_samples, fake_count, real_count, acc_str


def build_homepage_html():
    model_ok, n_samples, fake_count, real_count, acc_str = get_live_stats()

    model_dot   = "🟢" if model_ok else "🔴"
    model_label = "OPERATIONAL" if model_ok else "UNTRAINED"

    return f"""
<div style="font-family:'Outfit',sans-serif;color:#f8fafc;padding:0.5rem 0;">

  <!-- HERO -->
  <div style="background:linear-gradient(135deg,rgba(139,92,246,0.18),rgba(6,182,212,0.18));
       border:1px solid rgba(139,92,246,0.35);border-radius:20px;padding:2.5rem;
       text-align:center;margin-bottom:1.5rem;">
    <div style="font-size:0.7rem;font-weight:800;letter-spacing:4px;color:#a78bfa;margin-bottom:10px;">
      DEEPVOICE SHIELD PRO &nbsp;·&nbsp; HACKATHON EDITION
    </div>
    <h2 style="font-size:2.4rem;font-weight:800;margin:0 0 12px;
        background:linear-gradient(to right,#a78bfa,#22d3ee);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;">
      Forensic Voice Authentication Engine
    </h2>
    <p style="opacity:0.75;max-width:600px;margin:0 auto;font-size:0.98rem;line-height:1.8;">
      A <b style="color:#a78bfa;">real-time deepfake detection system</b> that analyses 223 acoustic
      biomarkers — MFCCs, spectral flux, pitch entropy, and harmonic ratios —
      to classify any voice as <b style="color:#4ade80;">human</b> or <b style="color:#f87171;">AI-synthesised</b> in under 3 seconds.
    </p>
    <div style="margin-top:1.4rem;display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
      <span style="background:rgba(139,92,246,0.2);border:1px solid rgba(139,92,246,0.4);
          border-radius:100px;padding:5px 16px;font-size:0.78rem;font-weight:700;color:#a78bfa;">
        {model_dot} Engine: {model_label}
      </span>
      <span style="background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.3);
          border-radius:100px;padding:5px 16px;font-size:0.78rem;font-weight:700;color:#22d3ee;">
        🔬 XGBoost · MFCC · 223 Acoustic Features
      </span>
      <span style="background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.3);
          border-radius:100px;padding:5px 16px;font-size:0.78rem;font-weight:700;color:#4ade80;">
        ⚡ Sub-3s Inference · Real-Time Ready
      </span>
    </div>
  </div>

  <!-- LIVE STATS -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;">
    <div style="background:rgba(17,24,39,0.85);border:1px solid rgba(139,92,246,0.25);
         border-radius:16px;padding:1.4rem;text-align:center;">
      <div style="font-size:2.4rem;font-weight:800;color:#a78bfa;">223</div>
      <div style="font-size:0.7rem;opacity:0.5;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Acoustic Biomarkers</div>
    </div>
    <div style="background:rgba(17,24,39,0.85);border:1px solid rgba(6,182,212,0.25);
         border-radius:16px;padding:1.4rem;text-align:center;">
      <div style="font-size:2.4rem;font-weight:800;color:#22d3ee;">{n_samples}</div>
      <div style="font-size:0.7rem;opacity:0.5;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Training Samples</div>
    </div>
    <div style="background:rgba(17,24,39,0.85);border:1px solid rgba(34,197,94,0.25);
         border-radius:16px;padding:1.4rem;text-align:center;">
      <div style="font-size:2.4rem;font-weight:800;color:#4ade80;">{acc_str}</div>
      <div style="font-size:0.7rem;opacity:0.5;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Model Accuracy</div>
    </div>
    <div style="background:rgba(17,24,39,0.85);border:1px solid rgba(244,63,94,0.25);
         border-radius:16px;padding:1.4rem;text-align:center;">
      <div style="font-size:2.4rem;font-weight:800;color:#f87171;">{fake_count}</div>
      <div style="font-size:0.7rem;opacity:0.5;text-transform:uppercase;letter-spacing:1px;margin-top:4px;">Synthetic Samples Flagged</div>
    </div>
  </div>

  <!-- DETECTION PIPELINE -->
  <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(255,255,255,0.07);
       border-radius:18px;padding:1.8rem;margin-bottom:1.5rem;">
    <div style="font-size:0.72rem;font-weight:800;color:#22d3ee;letter-spacing:3px;margin-bottom:1.2rem;">
      🧠 DETECTION PIPELINE
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;text-align:center;">
      <div style="padding:1.2rem;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:14px;">
        <div style="font-size:1.8rem;margin-bottom:8px;">🎤</div>
        <div style="font-weight:700;font-size:0.88rem;color:#a78bfa;margin-bottom:5px;">01 · INGEST</div>
        <div style="font-size:0.78rem;opacity:0.65;line-height:1.6;">Accepts .wav / .mp3 upload or live microphone capture</div>
      </div>
      <div style="padding:1.2rem;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);border-radius:14px;">
        <div style="font-size:1.8rem;margin-bottom:8px;">🔬</div>
        <div style="font-weight:700;font-size:0.88rem;color:#22d3ee;margin-bottom:5px;">02 · EXTRACT</div>
        <div style="font-size:0.78rem;opacity:0.65;line-height:1.6;">Derives 223 features — MFCCs, spectral centroid, ZCR, chroma, RMS</div>
      </div>
      <div style="padding:1.2rem;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:14px;">
        <div style="font-size:1.8rem;margin-bottom:8px;">🤖</div>
        <div style="font-weight:700;font-size:0.88rem;color:#a78bfa;margin-bottom:5px;">03 · CLASSIFY</div>
        <div style="font-size:0.78rem;opacity:0.65;line-height:1.6;">XGBoost ensemble scores synthetic probability against learned voice manifold</div>
      </div>
      <div style="padding:1.2rem;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:14px;">
        <div style="font-size:1.8rem;margin-bottom:8px;">📋</div>
        <div style="font-weight:700;font-size:0.88rem;color:#4ade80;margin-bottom:5px;">04 · REPORT</div>
        <div style="font-size:0.78rem;opacity:0.65;line-height:1.6;">Delivers verdict, confidence score, spectrogram, and a downloadable PDF report</div>
      </div>
    </div>
  </div>

  <!-- VERDICT GUIDE + USAGE MODES -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.5rem;">

    <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:1.6rem;">
      <div style="font-size:0.72rem;font-weight:800;color:#22d3ee;letter-spacing:3px;margin-bottom:1rem;">
        🚦 VERDICT CLASSIFICATION
      </div>
      <div style="display:flex;flex-direction:column;gap:14px;">
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="color:#4ade80;font-weight:700;">✓ GENUINE HUMAN</span>
            <span style="font-size:0.78rem;opacity:0.5;">Synthetic probability &lt; threshold</span>
          </div>
          <div style="background:rgba(0,0,0,0.3);border-radius:6px;height:8px;">
            <div style="width:25%;height:100%;background:#22c55e;border-radius:6px;"></div>
          </div>
          <div style="font-size:0.78rem;opacity:0.55;margin-top:4px;">Acoustic biomarkers consistent with natural human phonation.</div>
        </div>
        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="color:#f87171;font-weight:700;">⚠ SYNTHETIC VOICE</span>
            <span style="font-size:0.78rem;opacity:0.5;">High synthetic confidence</span>
          </div>
          <div style="background:rgba(0,0,0,0.3);border-radius:6px;height:8px;">
            <div style="width:92%;height:100%;background:#ef4444;border-radius:6px;"></div>
          </div>
          <div style="font-size:0.78rem;opacity:0.55;margin-top:4px;">Strong AI-generation signature. Voice classified as deepfake.</div>
        </div>
      </div>
    </div>

    <div style="background:rgba(17,24,39,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:1.6rem;">
      <div style="font-size:0.72rem;font-weight:800;color:#22d3ee;letter-spacing:3px;margin-bottom:1rem;">
        ⚡ ANALYSIS MODES
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div style="display:flex;align-items:center;gap:12px;padding:12px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:12px;">
          <div style="font-size:1.6rem;flex-shrink:0;">🎙️</div>
          <div>
            <div style="font-weight:700;font-size:0.88rem;color:#a78bfa;">Live Capture</div>
            <div style="font-size:0.76rem;opacity:0.6;margin-top:2px;">Forensic Lab → activate microphone → capture → analyse</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;padding:12px;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.2);border-radius:12px;">
          <div style="font-size:1.6rem;flex-shrink:0;">📁</div>
          <div>
            <div style="font-weight:700;font-size:0.88rem;color:#22d3ee;">File Upload</div>
            <div style="font-size:0.76rem;opacity:0.6;margin-top:2px;">Forensic Lab → drop .wav or .mp3 → run analysis</div>
          </div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;padding:12px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:12px;">
          <div style="font-size:1.6rem;flex-shrink:0;">🗂️</div>
          <div>
            <div style="font-weight:700;font-size:0.88rem;color:#4ade80;">Batch Audit</div>
            <div style="font-size:0.76rem;opacity:0.6;margin-top:2px;">Batch Audit tab → upload multiple files → bulk scan</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- THREAT CONTEXT -->
  <div style="background:linear-gradient(135deg,rgba(244,63,94,0.08),rgba(139,92,246,0.08));
       border:1px solid rgba(244,63,94,0.25);border-radius:18px;padding:1.6rem;text-align:center;">
    <div style="font-size:0.72rem;font-weight:800;color:#f87171;letter-spacing:3px;margin-bottom:8px;">⚠️ THREAT LANDSCAPE</div>
    <p style="opacity:0.78;max-width:660px;margin:0 auto;font-size:0.92rem;line-height:1.8;">
      Modern TTS and voice-cloning models produce output that defeats human detection at
      <b style="color:#fbbf24;">~60% accuracy</b> — barely above chance.
      Adversarial voice synthesis is actively exploited in CEO fraud, identity spoofing, and
      disinformation campaigns. DeepVoice Shield deploys
      <b style="color:#a78bfa;">ML-driven acoustic forensics</b> to close that gap —
      providing journalists, security teams, and enterprises with a verifiable, explainable verdict.
    </p>
  </div>

</div>
"""


# ─────────────────────────────────────────────
# GRADIO APP
# ─────────────────────────────────────────────

with gr.Blocks(title="DeepVoice Shield Pro", css=css) as demo:
    gr.HTML(
        "<div class='main-header'>"
        "<h1>DEEPVOICE SHIELD PRO</h1>"
        "<p style='opacity:0.6'>Advanced Forensic Audio Verification Suite</p>"
        "</div>"
    )

    with gr.Tabs():

        # ── TAB 0: COMMAND CENTER ────────────────────────────────────────
        with gr.TabItem("🏠 Command Center"):
            homepage_html = gr.HTML(build_homepage_html())
            refresh_btn = gr.Button("🔄 Refresh System Status", elem_classes=["primary-btn"])
            refresh_btn.click(fn=build_homepage_html, inputs=[], outputs=[homepage_html])

        # ── TAB 1: FORENSIC LAB ──────────────────────────────────────────
        with gr.TabItem("🔍 Forensic Lab"):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(elem_classes=["glass-card"]):
                        audio_input = gr.Audio(
                            label="Audio Evidence — Upload or Record",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        analyze_btn = gr.Button("🔍 START ANALYSIS", variant="primary",
                                                elem_classes=["primary-btn"])
                    threshold = gr.Slider(0.3, 0.8, 0.5, step=0.05, label="Sensitivity")
                    with gr.Row():
                        d_out = gr.Textbox(label="Duration",  interactive=False, elem_classes=["info-card"])
                        p_out = gr.Textbox(label="Pitch/Mag", interactive=False, elem_classes=["info-card"])
                    with gr.Row():
                        c_out = gr.Textbox(label="AI Conf %", interactive=False, elem_classes=["info-card"])
                        r_out = gr.Textbox(label="Threat",    interactive=False, elem_classes=["info-card"])

                with gr.Column(scale=5):
                    v_out      = gr.HTML("<div class='result-display' style='opacity:0.5'>AWAITING EVIDENCE...</div>")
                    reason_out = gr.HTML()
                    report_out = gr.File(label="📄 Forensic Report", elem_classes=["doc-box"])

            with gr.Row():
                spec_out = gr.Image(label="Spectrogram",       elem_classes=["viz-container"])
                feat_out = gr.Image(label="Feature Breakdown", elem_classes=["viz-container"])

            history = gr.Dataframe(
                headers=["File", "Verdict", "Conf", "Time"],
                elem_classes=["history-table"]
            )
            analyze_btn.click(
                fn=analyze_voice,
                inputs=[audio_input, threshold, history],
                outputs=[v_out, spec_out, feat_out, d_out, p_out,
                         c_out, r_out, history, report_out, reason_out]
            )

        # ── TAB 2: BATCH AUDIT ───────────────────────────────────────────
        with gr.TabItem("🗳️ Batch Audit"):
            with gr.Column(elem_classes=["glass-card"]):
                b_files     = gr.File(label="Upload Samples", file_count="multiple")
                b_threshold = gr.Slider(0.3, 0.8, 0.5, step=0.05, label="Sensitivity")
                b_btn       = gr.Button("🚀 START SCAN", variant="primary",
                                        elem_classes=["primary-btn"])
                b_out       = gr.Dataframe(headers=["File", "Result", "Conf", "Time"])
                b_btn.click(batch_analyze, [b_files, b_threshold], b_out)


if __name__ == "__main__":
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    demo.launch(share=True) 
 # updated

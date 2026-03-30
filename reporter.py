try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False
    class FPDF: pass # Dummy class to prevent crash on inheritance

import datetime
import os

class ForensicReport(FPDF):
    def __init__(self):
        if HAS_FPDF:
            super().__init__()
            self.set_margins(15, 15, 15)
            self.set_auto_page_break(True, margin=15)

    def header(self):
        if not HAS_FPDF: return
        self.set_fill_color(15, 15, 46)  
        self.rect(0, 0, 210, 40, 'F')
        self.set_font('Arial', 'B', 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, 'DeepVoice Shield - Forensic Analysis', 0, 1, 'C')
        self.ln(20)

    def footer(self):
        if not HAS_FPDF: return
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Digital Forensic Audit', 0, 0, 'C')

def generate_pdf_report(filename, prob_fake, risk_level, duration, avg_pitch, reasoning, output_path):
    if not HAS_FPDF:
        print("⚠ Skipping PDF generation: 'fpdf2' library not installed.")
        return None

    try:
        pdf = ForensicReport()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, f"Analysis Summary: {filename}", 0, 1, 'L')
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(50, 10, "Verdict:", 0, 0)
        pdf.cell(0, 10, risk_level, 0, 1)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(50, 10, "Synthetic Likelihood:", 0, 0)
        pdf.cell(0, 10, f"{prob_fake*100:.1f} %", 0, 1)
        
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "AI Forensic Reasoning", "B", 1)
        pdf.ln(2)
        for reason in reasoning:
            # Use fixed width of 180 to avoid 'not enough space' errors
            pdf.multi_cell(180, 8, f"- {reason}", 0, 'L')
        
        pdf.output(output_path)
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None
# updated

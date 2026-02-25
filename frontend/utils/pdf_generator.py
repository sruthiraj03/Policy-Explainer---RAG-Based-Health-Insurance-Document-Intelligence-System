"""
Utility for generating PDF summaries.
"""
from fpdf import FPDF  # pip install fpdf2

def generate_policy_pdf(summary_data: dict) -> bytes:
    """Generate a PDF from the provided summary dictionary."""

    def clean_text(text: str) -> str:
        replacements = {
            "\u2022": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "Policy Explainer Summary", ln=True, align="C")
    pdf.set_font("Arial", "", 10)

    for section in summary_data.get("sections", []):
        if section.get("present"):
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, clean_text(f" {section['section_name']}"), ln=True, fill=True)
            pdf.ln(2)

            pdf.set_font("Arial", "", 11)
            for bullet in section.get("bullets", []):
                pdf.multi_cell(0, 7, clean_text(f"- {bullet['text']}"))
                pdf.ln(1)
            pdf.ln(5)

    out = pdf.output(dest="S")
    return out.encode("latin-1") if isinstance(out, str) else bytes(out)
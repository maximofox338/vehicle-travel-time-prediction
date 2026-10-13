"""
Render report.md to a polished PDF (Vehicle_Travel_Time_Prediction_Report.pdf).

A small Markdown renderer built on fpdf2 (pure Python, no LaTeX needed) that supports the
subset used by the report: headings, paragraphs with **bold**, bullet lists, block quotes,
images and pipe tables.  report.md is the single source of truth.
"""
from pathlib import Path

from fpdf import FPDF

from src import config as C

REPORT_MD = C.PACKAGE_ROOT / "report.md"
OUT_PDF = C.PACKAGE_ROOT / "Vehicle_Travel_Time_Prediction_Report.pdf"
FONTS = Path(r"C:\Windows\Fonts")
PAGE_W, MARGIN = 210, 18
CONTENT_W = PAGE_W - 2 * MARGIN


def png_size(path: Path):
    head = Path(path).read_bytes()[:24]
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


class Report(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font(self.body, "I", 8)
        self.set_text_color(140)
        self.cell(0, 8, f"{self.page_no()}", align="C")
        self.set_text_color(0)


def make_pdf():
    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    # Use a Unicode TrueType font if available, otherwise fall back to a core font.
    try:
        pdf.add_font("Body", "", str(FONTS / "arial.ttf"))
        pdf.add_font("Body", "B", str(FONTS / "arialbd.ttf"))
        pdf.add_font("Body", "I", str(FONTS / "ariali.ttf"))
        pdf.body = "Body"
    except Exception:
        pdf.body = "Helvetica"
    pdf.add_page()

    # backticks are literal in fpdf2's markdown; drop them for a clean look in the PDF
    lines = REPORT_MD.read_text(encoding="utf-8").replace("`", "").splitlines()
    i, n = 0, len(lines)

    def text_block(buf):
        if not buf:
            return
        pdf.set_font(pdf.body, "", 10.5)
        pdf.multi_cell(0, 5.6, " ".join(buf), markdown=True)
        pdf.ln(1.5)

    para = []
    while i < n:
        line = lines[i].rstrip()

        if not line.strip():
            text_block(para); para = []; i += 1; continue

        if line.startswith("# "):
            text_block(para); para = []
            pdf.set_font(pdf.body, "B", 18)
            pdf.multi_cell(0, 9, line[2:].strip(), markdown=True); pdf.ln(2)
        elif line.startswith("## "):
            text_block(para); para = []
            pdf.ln(1); pdf.set_font(pdf.body, "B", 13.5)
            pdf.multi_cell(0, 7, line[3:].strip(), markdown=True); pdf.ln(1)
        elif line.startswith("### "):
            text_block(para); para = []
            pdf.set_font(pdf.body, "B", 11.5)
            pdf.multi_cell(0, 6, line[4:].strip(), markdown=True); pdf.ln(0.5)
        elif line.strip() == "---":
            text_block(para); para = []
            pdf.ln(1); pdf.set_draw_color(200); pdf.line(MARGIN, pdf.get_y(), PAGE_W - MARGIN, pdf.get_y())
            pdf.set_draw_color(0); pdf.ln(2)
        elif line.startswith("!["):                      # image: ![alt](path)
            text_block(para); para = []
            path = line[line.index("(") + 1:line.rindex(")")]
            img = C.PACKAGE_ROOT / path
            w, h = png_size(img)
            disp_w = min(CONTENT_W, 150)
            if disp_w * h / w > 92:
                disp_w = 92 * w / h
            pdf.ln(1); pdf.image(str(img), x=(PAGE_W - disp_w) / 2, w=disp_w); pdf.ln(2.5)
        elif line.startswith(">"):                       # block quote (the metric formula)
            text_block(para); para = []
            pdf.set_font(pdf.body, "I", 10.5); pdf.set_text_color(70)
            pdf.set_x(MARGIN + 6)
            pdf.multi_cell(CONTENT_W - 6, 5.6, line.lstrip("> ").strip(), markdown=True)
            pdf.set_text_color(0); pdf.ln(1.5)
        elif line.startswith("- ") or line.startswith("* "):
            text_block(para); para = []
            pdf.set_font(pdf.body, "", 10.5)
            while i < n and (lines[i].startswith("- ") or lines[i].startswith("* ")):
                item = lines[i][2:].strip()
                pdf.multi_cell(0, 5.4, "•  " + item, markdown=True)
                pdf.ln(0.8)
                i += 1
            pdf.ln(1.0); continue
        elif line.startswith("|"):                       # pipe table
            text_block(para); para = []
            rows = []
            while i < n and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):     # skip the |---| separator
                    rows.append(cells)
                i += 1
            pdf.set_font(pdf.body, "", 9.5)
            with pdf.table(markdown=True, line_height=5.2, width=CONTENT_W,
                           headings_style=__import__("fpdf").fonts.FontFace(emphasis="BOLD")) as table:
                for r in rows:
                    trow = table.row()
                    for c in r:
                        trow.cell(c)
            pdf.ln(2.5); continue
        else:
            para.append(line.strip())
        i += 1

    text_block(para)
    pdf.output(str(OUT_PDF))
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size//1024} KB, {pdf.page_no()} pages)")


if __name__ == "__main__":
    make_pdf()

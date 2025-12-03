import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os
import re
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

from PIL import Image

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase.pdfmetrics import stringWidth
except ImportError:
    canvas = None

# ============================================================
# 1) IMPORT FULL LAB KB
# ============================================================

# Make sure lab_kb_full.py is in the same folder as this file.
# It must define: LAB_KB, ALIAS_TO_ID, NAME_TO_ID, TEST_NAMES_EN
from lab_kb_full import LAB_KB, ALIAS_TO_ID, NAME_TO_ID, TEST_NAMES_EN

# ============================================================
# GLOBALS FOR LAST PARSED FILE
# ============================================================

last_labs = []
last_raw_text = ""
last_source_file = ""


# ============================================================
# 2) REPORT GENERATION FROM LOCAL KB
# ============================================================

def generate_kb_report(test_id, condition, language):
    """
    Builds a patient-friendly explanation string for a given test_id,
    test level (high/normal/low) and language (en/tr).
    Safely handles missing fields in LAB_KB.
    """
    data = LAB_KB[test_id]

    # Names – be defensive
    if language == "tr":
        name = data.get("name_tr") or data.get("name_en") or test_id
        short = data.get("short_patient_friendly_tr", "") or data.get("short_patient_friendly_en", "")
        what_is = data.get("what_is_tr", "")
        why = data.get("why_ordered_tr", "")
        prep = data.get("preparation_tr", "")
        disclaimer = data.get("disclaimer_tr", "") or "Bu bilgiler geneldir ve doktor değerlendirmesinin yerini tutmaz."
    else:
        name = data.get("name_en") or data.get("name_tr") or test_id
        short = data.get("short_patient_friendly_en", "") or data.get("short_patient_friendly_tr", "")
        what_is = data.get("what_is_en", "")
        why = data.get("why_ordered_en", "")
        prep = data.get("preparation_en", "")
        disclaimer = data.get("disclaimer_en", "") or "This information is general and does not replace your doctor's evaluation."

    interp_all = data.get("interpretation", {})
    interp = interp_all.get(condition, {}) or {}

    if language == "tr":
        summary = interp.get("summary_tr", "")
        causes = interp.get("causes_tr", []) or []
        next_steps = interp.get("next_steps_tr", []) or []
        note = interp.get("note_tr", "")
    else:
        summary = interp.get("summary_en", "")
        causes = interp.get("causes_en", []) or []
        next_steps = interp.get("next_steps_en", []) or []
        note = interp.get("note_en", "")

    lines = []

    if language == "tr":
        lines.append(f"Laboratuvar Testi: {name}")
        lines.append(f"Test Düzeyi: {condition.capitalize()}")
        lines.append("")
        if short:
            lines.append(short)
            lines.append("")
        if what_is:
            lines.append("Test nedir?")
            lines.append(what_is)
            lines.append("")
        if why:
            lines.append("Neden istenir?")
            lines.append(why)
            lines.append("")
        if prep:
            lines.append("Hazırlık:")
            lines.append(prep)
            lines.append("")
        if summary:
            lines.append("Sonuç ne anlama gelir?")
            lines.append(summary)
            lines.append("")
        if causes:
            lines.append("Sık nedenler:")
            for c in causes:
                lines.append(f"• {c}")
            lines.append("")
        if next_steps:
            lines.append("Sıradaki olası adımlar:")
            for s in next_steps:
                lines.append(f"• {s}")
            lines.append("")
        if note:
            lines.append(note)
            lines.append("")
        lines.append(disclaimer)
    else:
        lines.append(f"Lab Test: {name}")
        lines.append(f"Test Level: {condition.capitalize()}")
        lines.append("")
        if short:
            lines.append(short)
            lines.append("")
        if what_is:
            lines.append("What is this test?")
            lines.append(what_is)
            lines.append("")
        if why:
            lines.append("Why is it ordered?")
            lines.append(why)
            lines.append("")
        if prep:
            lines.append("Preparation:")
            lines.append(prep)
            lines.append("")
        if summary:
            lines.append("What does this result mean?")
            lines.append(summary)
            lines.append("")
        if causes:
            lines.append("Common causes:")
            for c in causes:
                lines.append(f"• {c}")
            lines.append("")
        if next_steps:
            lines.append("Possible next steps:")
            for s in next_steps:
                lines.append(f"• {s}")
            lines.append("")
        if note:
            lines.append(note)
            lines.append("")
        lines.append(disclaimer)

    return "\n".join(lines)


# ============================================================
# 2.5) FILE TEXT EXTRACTION & LAB PARSING
# ============================================================

def extract_text_from_file(path: str) -> str:
    """
    Returns plain text from a PDF or image file.
    """
    ext = os.path.splitext(path)[1].lower()

    # PDF
    if ext == ".pdf":
        if pdfplumber is None:
            return "ERROR: pdfplumber is not installed. Run: pip install pdfplumber"
        text_chunks = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_chunks.append(page.extract_text() or "")
        return "\n".join(text_chunks)

    # Image formats
    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        if pytesseract is None:
            return "ERROR: pytesseract is not installed. Run: pip install pytesseract"
        img = Image.open(path)
        return pytesseract.image_to_string(img)

    return f"ERROR: Unsupported file type: {ext}"


def infer_condition_from_text(line: str, value_str: str) -> str:
    """
    Decide high/normal/low using H/L flags or reference range in the same line.
    """
    try:
        value = float(value_str.replace(",", "."))
    except ValueError:
        return "normal"

    # Check H/L flags after the value (e.g. "10.5 H" or "10.5 L")
    if re.search(rf"{re.escape(value_str)}\s*[Hh]\b", line):
        return "high"
    if re.search(rf"{re.escape(value_str)}\s*[Ll]\b", line):
        return "low"

    # Look for reference range like '3.5 - 5.5'
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)", line)
    if m:
        try:
            ref_low = float(m.group(1).replace(",", "."))
            ref_high = float(m.group(2).replace(",", "."))
            if value < ref_low:
                return "low"
            if value > ref_high:
                return "high"
            return "normal"
        except ValueError:
            pass

    return "normal"


def parse_lab_text(raw_text: str):
    """
    - Splits text into lines.
    - For each known test name/alias in TEST_NAMES_EN, searches lines containing that name.
    - Extracts the first numeric value AFTER the name and interprets it.
    Returns: list of {test_id, display_name, value, condition, raw_line}, error_message_or_None
    """
    results = []
    if raw_text.startswith("ERROR:"):
        return [], raw_text

    lines = raw_text.splitlines()

    for name in TEST_NAMES_EN:
        pattern = re.compile(rf"(?i)\b{re.escape(name)}\b")

        for line in lines:
            m_name = pattern.search(line)
            if not m_name:
                continue

            # only look for the number AFTER the matched name
            tail = line[m_name.end():]
            num_match = re.search(r"([-+]?\d+(?:[.,]\d+)?)", tail)
            if not num_match:
                continue

            value_str = num_match.group(1)
            condition = infer_condition_from_text(line, value_str)

            test_id = NAME_TO_ID.get(name)
            if not test_id:
                continue

            results.append({
                "test_id": test_id,
                "display_name": name,
                "value": value_str,
                "condition": condition,
                "raw_line": line.strip()
            })
            # stop after first hit for that name
            break

    return results, None


# ============================================================
# 3) PDF EXPORT (TITLE + PATIENT INFO + RAW LINES + ONE TEST PER PAGE)
# ============================================================

def export_report_to_pdf(report_widget, patient_name_var, patient_age_var, patient_sex_var, patient_id_var, lang_var):
    """
    Exports a structured PDF:
      - Page 1+ : Title + patient info + detected RAW lines from uploaded PDF
      - Then: one test per section/page with detailed explanation
    Falls back to simple text export if no parsed labs are stored.
    """
    global last_labs, last_raw_text, last_source_file

    if canvas is None:
        messagebox.showerror(
            "ReportLab Missing",
            "ReportLab is not installed.\nInstall with:\n\npip install reportlab"
        )
        return

    language = lang_var.get().strip()

    file_path = filedialog.asksaveasfilename(
        title="Save PDF Report",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not file_path:
        return

    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    left_margin = 40
    right_margin = 40
    top_margin = height - 50
    bottom_margin = 40
    line_height = 14
    font_name = "Helvetica"
    font_size = 11

    def wrap_line(line, max_width):
        """
        Wrap a single line according to available width,
        using ReportLab's stringWidth measurement.
        """
        words = line.split(" ")
        wrapped = []
        current = ""

        for w in words:
            candidate = w if not current else current + " " + w
            w_width = stringWidth(candidate, font_name, font_size)
            if w_width <= max_width:
                current = candidate
            else:
                if current:
                    wrapped.append(current)
                current = w
        if current:
            wrapped.append(current)
        return wrapped

    max_text_width = width - left_margin - right_margin

    def new_text_object():
        t = c.beginText()
        t.setTextOrigin(left_margin, top_margin)
        t.setFont(font_name, font_size)
        return t

    def write_wrapped_lines(text_obj, lines):
        for raw_line in lines:
            if not raw_line.strip():
                if text_obj.getY() <= bottom_margin:
                    c.drawText(text_obj)
                    c.showPage()
                    text_obj = new_text_object()
                text_obj.textLine("")
                continue

            for wline in wrap_line(raw_line, max_text_width):
                if text_obj.getY() <= bottom_margin:
                    c.drawText(text_obj)
                    c.showPage()
                    text_obj = new_text_object()
                text_obj.textLine(wline)
        return text_obj

    # ============================
    # CASE 1: WE HAVE PARSED LABS
    # ============================
    if last_labs:
        # PAGE 1: Title + Patient Info + Raw lines
        text_obj = new_text_object()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        if language == "tr":
            title = "Cep Doktorum - Laboratuvar Sonuç Özeti"
            patient_info_lines = [
                title,
                "",
                f"Rapor Tarihi: {now_str}",
                f"Hasta Adı: {patient_name_var.get().strip()}",
                f"Yaş: {patient_age_var.get().strip()}",
                f"Cinsiyet: {patient_sex_var.get().strip()}",
                f"Hasta ID: {patient_id_var.get().strip()}",
                f"Kaynak Dosya: {os.path.basename(last_source_file) if last_source_file else ''}",
                "",
                "Algılanan ham laboratuvar satırları (yorum eklenmeden):",
                ""
            ]
        else:
            title = "Cep Doktorum - Lab Result Summary"
            patient_info_lines = [
                title,
                "",
                f"Report Time: {now_str}",
                f"Patient Name: {patient_name_var.get().strip()}",
                f"Age: {patient_age_var.get().strip()}",
                f"Sex: {patient_sex_var.get().strip()}",
                f"Patient ID: {patient_id_var.get().strip()}",
                f"Source File: {os.path.basename(last_source_file) if last_source_file else ''}",
                "",
                "Detected raw lab result lines (without comments):",
                ""
            ]

        text_obj = write_wrapped_lines(text_obj, patient_info_lines)

        # Raw lines: we can just use the raw_line fields, deduplicated
        seen_raw = set()
        raw_lines = []
        for item in last_labs:
            rl = item.get("raw_line", "").strip()
            if rl and rl not in seen_raw:
                seen_raw.add(rl)
                raw_lines.append(rl)

        if not raw_lines:
            if language == "tr":
                raw_lines = ["(Ham laboratuvar satırı bulunamadı.)"]
            else:
                raw_lines = ["(No raw lab lines captured.)"]

        text_obj = write_wrapped_lines(text_obj, raw_lines)
        c.drawText(text_obj)
        c.showPage()

        # NEXT: One test per section/page
        for item in last_labs:
            test_id = item["test_id"]
            cond = item["condition"]
            value = item["value"]
            display_name = item["display_name"]
            raw_line = item["raw_line"]

            text_obj = new_text_object()

            if language == "tr":
                header_lines = [
                    f"Test: {display_name}",
                    f"Değer: {value}  |  Yorum: {cond}",
                    f"Ham Satır: {raw_line}",
                    "",
                ]
            else:
                header_lines = [
                    f"Test: {display_name}",
                    f"Value: {value}  |  Interpreted: {cond}",
                    f"Source line: {raw_line}",
                    "",
                ]

            text_obj = write_wrapped_lines(text_obj, header_lines)

            # Explanation from KB
            explanation = generate_kb_report(test_id, cond, language)
            expl_lines = explanation.splitlines()
            text_obj = write_wrapped_lines(text_obj, expl_lines)

            c.drawText(text_obj)
            c.showPage()

        c.save()
        messagebox.showinfo("PDF Exported", f"PDF report saved to:\n{file_path}")
        return

    # ======================================
    # CASE 2: NO PARSED LABS (FALLBACK MODE)
    # ======================================
    # Fallback to exporting whatever is in the text widget as before
    text = report_widget.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Empty Report", "There is no text to export.")
        return

    text_obj = new_text_object()
    for raw_line in text.splitlines():
        if not raw_line.strip():
            if text_obj.getY() <= bottom_margin:
                c.drawText(text_obj)
                c.showPage()
                text_obj = new_text_object()
            text_obj.textLine("")
            continue

        for wline in wrap_line(raw_line, max_text_width):
            if text_obj.getY() <= bottom_margin:
                c.drawText(text_obj)
                c.showPage()
                text_obj = new_text_object()
            text_obj.textLine(wline)

    c.drawText(text_obj)
    c.save()
    messagebox.showinfo("PDF Exported", f"PDF report saved to:\n{file_path}")


# ============================================================
# 4) TKINTER UI
# ============================================================

root = tk.Tk()
root.title("Cep Doktorum - Lab Result Explainer")

# ---- Top: language & info ----
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

status_label = tk.Label(
    top_frame,
    text="Select a lab test and level or analyze a PDF/image to create a patient-friendly explanation."
)
status_label.pack()

lang_frame = tk.Frame(root)
lang_frame.pack(pady=5)

lang_var = tk.StringVar(value="en")
tk.Label(lang_frame, text="Language:").pack(side=tk.LEFT, padx=(0, 5))
tk.Radiobutton(lang_frame, text="English", variable=lang_var, value="en").pack(side=tk.LEFT)
tk.Radiobutton(lang_frame, text="Türkçe", variable=lang_var, value="tr").pack(side=tk.LEFT)

# ---- Patient info frame ----
patient_frame = tk.Frame(root)
patient_frame.pack(pady=5)

patient_name_var = tk.StringVar()
patient_age_var = tk.StringVar()
patient_sex_var = tk.StringVar()
patient_id_var = tk.StringVar()

tk.Label(patient_frame, text="Patient Name / Hasta Adı:").grid(row=0, column=0, sticky="e", padx=3, pady=2)
tk.Entry(patient_frame, textvariable=patient_name_var, width=25).grid(row=0, column=1, padx=3, pady=2)

tk.Label(patient_frame, text="Age / Yaş:").grid(row=0, column=2, sticky="e", padx=3, pady=2)
tk.Entry(patient_frame, textvariable=patient_age_var, width=8).grid(row=0, column=3, padx=3, pady=2)

tk.Label(patient_frame, text="Sex / Cinsiyet:").grid(row=1, column=0, sticky="e", padx=3, pady=2)
tk.Entry(patient_frame, textvariable=patient_sex_var, width=10).grid(row=1, column=1, padx=3, pady=2)

tk.Label(patient_frame, text="Patient ID / Hasta ID:").grid(row=1, column=2, sticky="e", padx=3, pady=2)
tk.Entry(patient_frame, textvariable=patient_id_var, width=15).grid(row=1, column=3, padx=3, pady=2)

# ---- Single box frame ----
boxes_frame = tk.Frame(root)
boxes_frame.pack(pady=10)


def generate_report_for_box(test_var, condition_var, report_widget):
    test_name = test_var.get().strip()
    condition = condition_var.get().strip().lower()
    language = lang_var.get().strip()

    if not test_name:
        messagebox.showwarning("Selection Missing", "Please select a lab test.")
        return

    test_id = NAME_TO_ID.get(test_name)
    if not test_id:
        messagebox.showerror("Unknown Test", f"Test not found in knowledge base: {test_name}")
        return

    if condition not in ("high", "normal", "low"):
        condition = "normal"

    text = generate_kb_report(test_id, condition, language)

    report_widget.delete(1.0, tk.END)
    report_widget.insert(tk.END, text)


def analyze_file_and_show(report_widget):
    """
    Lets user choose a PDF or image, extracts text, parses labs,
    and prints multi-test explanations into the report widget.
    Also stores last_labs and last_raw_text for PDF export.
    """
    global last_labs, last_raw_text, last_source_file

    language = lang_var.get().strip()

    path = filedialog.askopenfilename(
        title="Select Lab Report (PDF or Image)",
        filetypes=[
            ("PDF files", "*.pdf"),
            ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
            ("All files", "*.*"),
        ]
    )
    if not path:
        return

    last_source_file = path
    raw_text = extract_text_from_file(path)
    last_raw_text = raw_text

    labs, error = parse_lab_text(raw_text)
    last_labs = labs

    report_widget.delete(1.0, tk.END)

    if error:
        report_widget.insert(tk.END, error)
        return

    if not labs:
        report_widget.insert(tk.END, "No known lab tests were detected in the file.\n")
        report_widget.insert(tk.END, "\nRaw extracted text:\n\n")
        report_widget.insert(tk.END, raw_text)
        return

    for item in labs:
        test_id = item["test_id"]
        cond = item["condition"]
        value = item["value"]
        display_name = item["display_name"]
        raw_line = item["raw_line"]

        report_widget.insert(tk.END, "=" * 60 + "\n")
        if language == "tr":
            report_widget.insert(
                tk.END,
                f"{display_name} (Değer: {value}, Yorum: {cond})\n"
            )
            report_widget.insert(tk.END, f"Ham satır: {raw_line}\n\n")
        else:
            report_widget.insert(
                tk.END,
                f"{display_name} (Value: {value}, Interpreted: {cond})\n"
            )
            report_widget.insert(tk.END, f"Source line: {raw_line}\n\n")

        txt = generate_kb_report(test_id, cond, language)
        report_widget.insert(tk.END, txt + "\n\n")


# ----- ONE LAB TEST BOX -----

box_frame = tk.Frame(boxes_frame, relief=tk.RIDGE, bd=2, padx=5, pady=5)
box_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

tk.Label(box_frame, text="Lab Test", font=("Arial", 12, "bold")).pack(pady=(0, 5))

tk.Label(box_frame, text="Select Lab Test:").pack(anchor="w")
test_var = tk.StringVar()

# Combobox directly populated from all known names/aliases
dropdown = ttk.Combobox(box_frame, textvariable=test_var, width=40, values=TEST_NAMES_EN)
dropdown.pack(pady=2)

if TEST_NAMES_EN:
    test_var.set(TEST_NAMES_EN[0])

tk.Label(box_frame, text="Select Test Level:").pack(anchor="w")
condition_var = tk.StringVar(value="high")
cond_dropdown = ttk.Combobox(
    box_frame,
    textvariable=condition_var,
    values=["high", "normal", "low"],
    width=10
)
cond_dropdown.pack(pady=2)

report_widget = tk.Text(box_frame, wrap=tk.WORD, width=80, height=26)
report_widget.pack(pady=5)

gen_btn = tk.Button(
    box_frame,
    text="Generate Report (Manual Selection)",
    command=lambda: generate_report_for_box(test_var, condition_var, report_widget)
)
gen_btn.pack(pady=5)

analyze_btn = tk.Button(
    root,
    text="Analyze Lab PDF/Image",
    command=lambda: analyze_file_and_show(report_widget)
)
analyze_btn.pack(pady=5)

export_btn = tk.Button(
    root,
    text="Export Report as PDF",
    command=lambda: export_report_to_pdf(
        report_widget,
        patient_name_var,
        patient_age_var,
        patient_sex_var,
        patient_id_var,
        lang_var
    )
)
export_btn.pack(pady=5)

clear_btn = tk.Button(
    root,
    text="Clear Report",
    command=lambda: report_widget.delete(1.0, tk.END)
)
clear_btn.pack(pady=10)

root.mainloop()

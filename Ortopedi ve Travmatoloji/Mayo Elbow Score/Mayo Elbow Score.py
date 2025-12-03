import tkinter as tk
from tkinter import messagebox, filedialog
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from datetime import datetime

# Global variables to store last score for PDF
last_score = None
last_grade = None


# --- Mayo classification helper ---
def mayo_classification(score: int) -> str:
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Poor"


def compute_mayo_score():
    """Compute Mayo score from selections. Returns (score, grade).
       Returns (None, None) if required fields are not selected.
    """

    if pain_var.get() == -1 or motion_var.get() == -1 or stability_var.get() == -1:
        return None, None

    pain = pain_var.get()
    motion = motion_var.get()
    stability = stability_var.get()

    # Functional activities
    function_score = (
        hair_cb_var.get() * 5 +
        eating_cb_var.get() * 5 +
        shirt_cb_var.get() * 5 +
        shoes_cb_var.get() * 5 +
        hygiene_cb_var.get() * 5
    )

    total_score = pain + motion + stability + function_score
    grade = mayo_classification(total_score)
    return total_score, grade


def calculate_score():
    global last_score, last_grade

    score, grade = compute_mayo_score()
    if score is None:
        messagebox.showerror(
            "Missing Information",
            "Please select pain, motion arc, and stability options."
        )
        return

    last_score = score
    last_grade = grade

    result_label.config(
        text=f"Mayo Elbow Score: {score}  ({grade})",
        fg="#003366",
        font=("Helvetica", 12, "bold")
    )


def reset_form():
    global last_score, last_grade

    pain_var.set(-1)
    motion_var.set(-1)
    stability_var.set(-1)

    hair_cb_var.set(0)
    eating_cb_var.set(0)
    shirt_cb_var.set(0)
    shoes_cb_var.set(0)
    hygiene_cb_var.set(0)

    patient_name_var.set("")
    patient_id_var.set("")

    result_label.config(text="")
    last_score = None
    last_grade = None


def generate_pdf():
    global last_score, last_grade

    # If not calculated yet, try calculating
    if last_score is None or last_grade is None:
        score, grade = compute_mayo_score()
        if score is None:
            messagebox.showerror(
                "Missing Information",
                "Please calculate the score before generating the PDF."
            )
            return
        last_score, last_grade = score, grade

    patient_name = patient_name_var.get().strip()
    patient_id = patient_id_var.get().strip()

    # Save dialog
    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF File", "*.pdf")],
        title="Save Mayo Elbow Score Report"
    )
    if not file_path:
        return

    c = pdf_canvas.Canvas(file_path, pagesize=A4)
    width, height = A4
    y = height - 50

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Mayo Elbow Score Report")
    y -= 30

    # Date
    c.setFont("Helvetica", 10)
    today_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Patient Information
    if patient_name:
        c.drawString(50, y, f"Patient Name: {patient_name}")
        y -= 15
    if patient_id:
        c.drawString(50, y, f"Patient ID: {patient_id}")
        y -= 15

    c.drawString(50, y, f"Date: {today_str}")
    y -= 30

    # Score info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Mayo Elbow Score: {last_score}  ({last_grade})")
    y -= 25

    # Explanation
    c.setFont("Helvetica", 10)
    explanation_lines = [
        "The Mayo Elbow Performance Score (MEPS) evaluates the elbow joint using four criteria:",
        "pain, range of motion, stability, and functional activities of daily living.",
        "",
        "Score interpretation:",
        "  • 90–100 points : Excellent",
        "  • 75–89 points  : Good",
        "  • 60–74 points  : Fair",
        "  • <60 points    : Poor",
        "",
        "This automated report provides a standardized scoring summary.",
        "Clinical decisions should always be made by a qualified physician based on",
        "the complete clinical picture, including physical examination and imaging.",
        "",
        "This report is intended for documentation and follow-up comparison."
    ]

    for line in explanation_lines:
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 15

    c.showPage()
    c.save()

    messagebox.showinfo("PDF Saved", "The PDF report has been successfully created.")


# --- Main Window ---
root = tk.Tk()
root.title("Mayo Elbow Score Calculator")
root.configure(bg="#e6f2ff")

PADX = 10
PADY = 5

# --- Patient Information ---
patient_frame = tk.LabelFrame(root, text="Patient Information", bg="#e6f2ff", padx=PADX, pady=PADY)
patient_frame.grid(row=0, column=0, sticky="w", padx=PADX, pady=(PADY, 0))

patient_name_var = tk.StringVar()
patient_id_var = tk.StringVar()

tk.Label(patient_frame, text="Name:", bg="#e6f2ff").grid(row=0, column=0, sticky="e")
tk.Entry(patient_frame, textvariable=patient_name_var, width=30).grid(row=0, column=1, padx=5, pady=2)

tk.Label(patient_frame, text="ID:", bg="#e6f2ff").grid(row=1, column=0, sticky="e")
tk.Entry(patient_frame, textvariable=patient_id_var, width=30).grid(row=1, column=1, padx=5, pady=2)

# --- Pain ---
pain_frame = tk.LabelFrame(root, text="Pain", bg="#e6f2ff", padx=PADX, pady=PADY)
pain_frame.grid(row=1, column=0, sticky="w", padx=PADX, pady=PADY)

pain_var = tk.IntVar(value=-1)
pain_choices = [
    ("None (45 points)", 45),
    ("Mild (30 points)", 30),
    ("Moderate (15 points)", 15),
    ("Severe (0 points)", 0),
]

for i, (text, value) in enumerate(pain_choices):
    tk.Radiobutton(
        pain_frame, text=text, variable=pain_var, value=value,
        bg="#e6f2ff", anchor="w"
    ).grid(row=i, column=0, sticky="w")

# --- Motion ---
motion_frame = tk.LabelFrame(root, text="Range of Motion (Flexion–Extension Arc)",
                             bg="#e6f2ff", padx=PADX, pady=PADY)
motion_frame.grid(row=2, column=0, sticky="w", padx=PADX, pady=PADY)

motion_var = tk.IntVar(value=-1)
motion_choices = [
    (">100° (20 points)", 20),
    ("50°–100° (15 points)", 15),
    ("<50° (5 points)", 5),
]

for i, (text, value) in enumerate(motion_choices):
    tk.Radiobutton(
        motion_frame, text=text, variable=motion_var, value=value,
        bg="#e6f2ff", anchor="w"
    ).grid(row=i, column=0, sticky="w")

# --- Stability ---
stability_frame = tk.LabelFrame(root, text="Stability", bg="#e6f2ff", padx=PADX, pady=PADY)
stability_frame.grid(row=3, column=0, sticky="w", padx=PADX, pady=PADY)

stability_var = tk.IntVar(value=-1)
stability_choices = [
    ("Stable (10 points)", 10),
    ("Mild instability (5 points)", 5),
    ("Gross instability (0 points)", 0),
]

for i, (text, value) in enumerate(stability_choices):
    tk.Radiobutton(
        stability_frame, text=text, variable=stability_var, value=value,
        bg="#e6f2ff", anchor="w"
    ).grid(row=i, column=0, sticky="w")

# --- Functional Activities ---
function_frame = tk.LabelFrame(root, text="Functional Activities (5 points each)",
                               bg="#e6f2ff", padx=PADX, pady=PADY)
function_frame.grid(row=4, column=0, sticky="w", padx=PADX, pady=PADY)

hair_cb_var = tk.IntVar()
eating_cb_var = tk.IntVar()
shirt_cb_var = tk.IntVar()
shoes_cb_var = tk.IntVar()
hygiene_cb_var = tk.IntVar()

tk.Checkbutton(function_frame, text="Combs hair", variable=hair_cb_var, bg="#e6f2ff").grid(row=0, column=0, sticky="w")
tk.Checkbutton(function_frame, text="Eats comfortably", variable=eating_cb_var, bg="#e6f2ff").grid(row=1, column=0, sticky="w")
tk.Checkbutton(function_frame, text="Dresses (shirt/T-shirt)", variable=shirt_cb_var, bg="#e6f2ff").grid(row=2, column=0, sticky="w")
tk.Checkbutton(function_frame, text="Ties shoes or puts them on", variable=shoes_cb_var, bg="#e6f2ff").grid(row=3, column=0, sticky="w")
tk.Checkbutton(function_frame, text="Performs hygiene independently", variable=hygiene_cb_var, bg="#e6f2ff").grid(row=4, column=0, sticky="w")

# --- Buttons ---
button_frame = tk.Frame(root, bg="#e6f2ff")
button_frame.grid(row=5, column=0, pady=(10, 5))

tk.Button(button_frame, text="Calculate", command=calculate_score,
          bg="#003366", fg="black", width=12).grid(row=0, column=0, padx=5)

tk.Button(button_frame, text="Save PDF", command=generate_pdf,
          width=12).grid(row=0, column=1, padx=5)

tk.Button(button_frame, text="Reset", command=reset_form,
          width=10).grid(row=0, column=2, padx=5)

result_label = tk.Label(root, text="", bg="#e6f2ff")
result_label.grid(row=6, column=0, pady=(5, 15))

root.mainloop()

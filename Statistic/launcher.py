import subprocess
from pathlib import Path
import customtkinter as ctk

BASE_DIR = Path(__file__).resolve().parent

class Launcher(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PocketDoc Launcher")
        self.geometry("400x450")
        ctk.set_appearance_mode("light")
        self.frame = ctk.CTkFrame(self)
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.progress = None
        self.show_main()

    def clear(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

    def show_loading(self, message):
        """Display a loading screen with an indeterminate progress bar."""
        self.clear()
        ctk.CTkLabel(self.frame, text=message, font=("Arial", 14)).pack(pady=10)
        self.progress = ctk.CTkProgressBar(self.frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=20, pady=20)
        self.progress.start()

    def run_script(self, relative):
        path = BASE_DIR / relative
        self.show_loading(f"Opening {path.name}...")
        subprocess.Popen(["python3", str(path)])
        # Return to the main menu after a short loading delay
        self.after(6000, self.show_main)

    def show_main(self):
        self.clear()
        if self.progress is not None:
            self.progress.stop()
            self.progress = None
        ctk.CTkLabel(self.frame, text="Select Topic", text_color="#003366", font=("Arial", 16)).pack(pady=10)
        ctk.CTkButton(self.frame, text="Dermatoloji", command=self.show_derm).pack(fill="x", pady=4)
        ctk.CTkButton(self.frame, text="Medikal Estetik", command=self.show_estetik).pack(fill="x", pady=4)
        ctk.CTkButton(self.frame, text="Ortopedi ve Travmatoloji", command=self.show_ortopedi).pack(fill="x", pady=4)
        ctk.CTkButton(self.frame, text="Laboratuvar Tablosu", command=lambda: self.run_script("Statistic/Statistic.py")).pack(fill="x", pady=4)
        ctk.CTkButton(self.frame, text="PatientSum", command=lambda: self.run_script("PatientSum/PatientSum.py")).pack(fill="x", pady=4)

    def show_derm(self):
        self.clear()
        ctk.CTkLabel(self.frame, text="Dermatoloji", font=("Arial", 16)).pack(pady=10)
        ctk.CTkButton(self.frame, text="Dermatoloji Algoritma", command=lambda: self.run_script("Dermatoloji/Dermatoloji Algoritma.py")).pack(fill="x", pady=3)
        ctk.CTkButton(self.frame, text="Melanom", command=lambda: self.run_script("Dermatoloji/melanom.py")).pack(fill="x", pady=3)
        ctk.CTkButton(self.frame, text="Back", command=self.show_main).pack(pady=10)

    def show_estetik(self):
        self.clear()
        ctk.CTkLabel(self.frame, text="Medikal Estetik", font=("Arial", 16)).pack(pady=10)
        ctk.CTkButton(self.frame, text="Aesthetic Detector", command=lambda: self.run_script("Medikal Estetik/Aesthetic Detector/aesthetic_pathology_detector.py")).pack(fill="x", pady=3)
        ctk.CTkButton(self.frame, text="Golden Ratio Live", command=lambda: self.run_script("Medikal Estetik/GoldenRatioLive/GoldenRatioLive.py")).pack(fill="x", pady=3)
        ctk.CTkButton(self.frame, text="Profiloplasti", command=lambda: self.run_script("Medikal Estetik/Profiloplasti/Profiloplasti.py")).pack(fill="x", pady=3)
        ctk.CTkButton(self.frame, text="Yüz Altın Oran", command=lambda: self.run_script("Medikal Estetik/Yüz Altın Oran Ölçümü/Yüz Altın Oran.py")).pack(fill="x", pady=3)
        ctk.CTkButton(self.frame, text="Back", command=self.show_main).pack(pady=10)

    def show_ortopedi(self):
        self.clear()
        ctk.CTkLabel(self.frame, text="Ortopedi ve Travmatoloji", font=("Arial", 16)).pack(pady=10)
        ctk.CTkButton(self.frame, text="Live K-wire", command=lambda: self.run_script("Ortopedi ve Travmatoloji/LiveK-wirePredict.py")).pack(fill="x", pady=2)
        ctk.CTkButton(self.frame, text="Mayo Elbow Score", command=lambda: self.run_script("Ortopedi ve Travmatoloji/Mayo Elbow Score.py")).pack(fill="x", pady=2)
        ctk.CTkButton(self.frame, text="Elbow ROM", command=lambda: self.run_script("Ortopedi ve Travmatoloji/Elbow Rom/elbow rom.py")).pack(fill="x", pady=2)
        ctk.CTkButton(self.frame, text="Flynn", command=lambda: self.run_script("Ortopedi ve Travmatoloji/Flynn.py")).pack(fill="x", pady=2)
        ctk.CTkButton(self.frame, text="Lateral Kondil PSR-PCA", command=lambda: self.run_script("Ortopedi ve Travmatoloji/Lateral Kondil PSR-PCA.py")).pack(fill="x", pady=2)
        ctk.CTkButton(self.frame, text="Opere SKH PSR - PCA", command=lambda: self.run_script("Ortopedi ve Travmatoloji/Opere SKH PSR - PCA.py")).pack(fill="x", pady=2)
        ctk.CTkButton(self.frame, text="Kwiresuggest", command=lambda: self.run_script("Ortopedi ve Travmatoloji/kwiresuggest.py")).pack(fill="x", pady=2)
        ctk.CTkButton(self.frame, text="Back", command=self.show_main).pack(pady=10)

if __name__ == "__main__":
    app = Launcher()
    app.mainloop()

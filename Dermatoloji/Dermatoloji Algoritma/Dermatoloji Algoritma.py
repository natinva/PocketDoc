import tkinter as tk
from tkinter import ttk, messagebox

# Diagnosis criteria based on provided document
diagnoses = {
    "Akne": {
        "Ana Şikayet": ["Birden çok lezyon"],
        "Lezyon tipi": ["Kızarıklık", "Diğer"],
        "Eşlik eden semptomlar": [""],
        "Lokasyon": ["Yüz", "Kollar", "Gövde"],
        "Şikayetin başlangıcı": ["Aylar-Yıllar"]
    },
    "Saçkıran": {
        "Ana Şikayet": ["Saç lezyonu"],
        "Lezyon tipi": ["Diğer"],
        "Eşlik eden semptomlar": [""],
        "Lokasyon": ["Saçlı deri"],
        "Şikayetin başlangıcı": ["Günler-Haftalar"]
    },
    "Kontakt Dermatit": {
        "Ana Şikayet": ["Birden çok lezyon", "Tek cilt lezyonu"],
        "Lezyon tipi": ["Kızarıklık", "Kaşıntı"],
        "Eşlik eden semptomlar": [""],
        "Lokasyon": ["Yüz", "Kollar", "Eller", "Bacaklar", "Ayaklar", "Saçlı deri", "Gövde", "Genital bölge", "Boyun"],
        "Şikayetin başlangıcı": ["Günler - Haftalar"]
    },
    "Androjenik Alopesi": {
        "Ana Şikayet": ["Saç lezyonu"],
        "Lezyon tipi": ["Diğer"],
        "Eşlik eden semptomlar": [""],
        "Lokasyon": ["Saçlı deri"],
        "Şikayetin başlangıcı": ["Aylar-Yıllar"]
    },
    "Tırnak Mantarı": {
        "Ana Şikayet": ["Tırnak lezyonu"],
        "Lezyon tipi": ["Diğer"],
        "Eşlik eden semptomlar": [""],
        "Lokasyon": ["Eller", "Ayaklar"],
        "Şikayetin başlangıcı": ["Günler-Haftalar", "Aylar-Yıllar"]
    },
    "Ayak Mantarı": {
        "Ana Şikayet": ["Birden çok lezyon", "Tek cilt lezyonu"],
        "Lezyon tipi": ["Yara"],
        "Eşlik eden semptomlar": ["Kaşıntı"],
        "Lokasyon": ["Ayaklar"],
        "Şikayetin başlangıcı": ["Günler-Haftalar", "Aylar-Yıllar"]
    },
    "Seboreik Dermatit": {
        "Ana Şikayet": ["Saç lezyonu"],
        "Lezyon tipi": ["Yara"],
        "Eşlik eden semptomlar": ["Kaşıntı"],
        "Lokasyon": ["Saçlı deri", "Yüz"],
        "Şikayetin başlangıcı": ["Aylar - Yıllar"]
    },
    "Psöriyazis": {
        "Ana Şikayet": ["Birden çok lezyon"],
        "Lezyon tipi": ["Yara"],
        "Eşlik eden semptomlar": ["Kaşıntı"],
        "Lokasyon": ["Saçlı deri", "Kollar", "Eller", "Gövde", "Genital bölge", "Bacaklar", "Ayaklar"],
        "Şikayetin başlangıcı": ["Aylar-Yıllar"]
    },
    # More diagnoses can be added here following the same structure
}

# Initialize main application
root = tk.Tk()
root.title("Dermatoloji Formu")

# Store user selections
user_answers = {}

# Updated question labels and answer options
questions = {
    "Ana Şikayet": ["Birden çok lezyon", "Tek cilt lezyonu", "Saç lezyonu", "Tırnak lezyonu", "Kaşıntı", "Diğer"],
    "Lezyon tipi": ["Su Toplaması", "Kızarıklık", "Yara", "Kabarıklık", "Diğer"],
    "Eşlik eden semptomlar": ["Kaşıntı", "Ağrı", "Ateş"],  # Blank option means any answer
    "Lokasyon": ["Yüz", "Kollar", "Eller", "Bacaklar", "Ayaklar", "Saçlı deri", "Gövde", "Genital bölge", "Boyun"],
    "Şikayetin başlangıcı": ["Günler-Haftalar", "Aylar-Yıllar", "Dakikalar - Saatler",
                             "Belli aralıklarla tekrarlayan şikayetler", "Doğuştan beri"]
}

# Create GUI form for each question
for i, (question, options) in enumerate(questions.items()):
    tk.Label(root, text=question).grid(row=i, column=0, sticky="w", padx=10, pady=5)
    user_answers[question] = tk.StringVar()
    dropdown = ttk.Combobox(root, textvariable=user_answers[question], values=options)
    dropdown.grid(row=i, column=1, padx=10, pady=5)
    dropdown.set("Seçiniz")  # Default prompt


# Function to find possible diagnoses
def show_diagnoses():
    matched_diagnoses = []

    for diagnosis, criteria in diagnoses.items():
        match = True
        for question, valid_answers in criteria.items():
            user_answer = user_answers[question].get()
            if user_answer not in valid_answers and "" not in valid_answers:
                match = False
                break
        if match:
            matched_diagnoses.append(diagnosis)

    if matched_diagnoses:
        messagebox.showinfo("Olası Tanılar", "\n".join(matched_diagnoses))
    else:
        messagebox.showinfo("Olası Tanılar", "Belirtilen yanıtlara göre bir tanı bulunamadı.")


# Button to trigger diagnosis detection
submit_button = tk.Button(root, text="Olası Hastalık Tanılarını Göster", command=show_diagnoses)
submit_button.grid(row=len(questions), column=0, columnspan=2, pady=10)

# Run the Tkinter application
root.mainloop()

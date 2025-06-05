import tkinter as tk
from tkinter import messagebox


def calculate_score():
    try:
        # Ağrı Puanı
        pain = int(pain_var.get())

        # Hareket Açıklığı Puanı
        motion = int(motion_var.get())

        # Stabilite Puanı
        stability = int(stability_var.get())

        # Fonksiyonel Aktivite Puanı
        function_score = 0
        if hair_cb_var.get() == 1:
            function_score += 5
        if eating_cb_var.get() == 1:
            function_score += 5
        if shirt_cb_var.get() == 1:
            function_score += 5
        if shoes_cb_var.get() == 1:
            function_score += 5
        if hygiene_cb_var.get() == 1:
            function_score += 5

        # Toplam Puan Hesaplama
        total_score = pain + motion + stability + function_score
        result_label.config(text=f"Mayo Elbow Score: {total_score}")

    except ValueError as ve:
        messagebox.showerror("Giriş Hatası", str(ve))


# Ana Pencere
root = tk.Tk()
root.title("Mayo Elbow Score Hesaplayıcı")

# Ağrı Seçenekleri
tk.Label(root, text="Ağrı:").grid(row=0, column=0, padx=10, pady=10)
pain_var = tk.StringVar()
pain_choices = [("Yok (45 puan)", 45), ("Hafif (30 puan)", 30), ("Orta (15 puan)", 15), ("Şiddetli (0 puan)", 0)]
for i, (text, value) in enumerate(pain_choices):
    tk.Radiobutton(root, text=text, variable=pain_var, value=value).grid(row=i, column=1, sticky="W")

# Hareket Açıklığı Seçenekleri
tk.Label(root, text="Hareket Açıklığı:").grid(row=4, column=0, padx=10, pady=10)
motion_var = tk.StringVar()
motion_choices = [(">100° (20 puan)", 20), ("50°-100° (15 puan)", 15), ("<50° (5 puan)", 5)]
for i, (text, value) in enumerate(motion_choices):
    tk.Radiobutton(root, text=text, variable=motion_var, value=value).grid(row=i + 4, column=1, sticky="W")

# Stabilite Seçenekleri
tk.Label(root, text="Stabilite:").grid(row=7, column=0, padx=10, pady=10)
stability_var = tk.StringVar()
stability_choices = [("Kararlı (10 puan)", 10), ("Hafif instabilite (5 puan)", 5), ("Ciddi instabilite (0 puan)", 0)]
for i, (text, value) in enumerate(stability_choices):
    tk.Radiobutton(root, text=text, variable=stability_var, value=value).grid(row=i + 7, column=1, sticky="W")

# Fonksiyonel Aktivite Seçenekleri (Checkbox)
tk.Label(root, text="Fonksiyonel Aktivite:").grid(row=10, column=0, padx=10, pady=10)
hair_cb_var = tk.IntVar()
eating_cb_var = tk.IntVar()
shirt_cb_var = tk.IntVar()
shoes_cb_var = tk.IntVar()
hygiene_cb_var = tk.IntVar()

tk.Checkbutton(root, text="Saçlarını rahat tarayabiliyor", variable=hair_cb_var).grid(row=10, column=1, sticky="W")
tk.Checkbutton(root, text="Rahatça yemek yiyebiliyor", variable=eating_cb_var).grid(row=11, column=1, sticky="W")
tk.Checkbutton(root, text="T-shirt giyebiliyor", variable=shirt_cb_var).grid(row=12, column=1, sticky="W")
tk.Checkbutton(root, text="Ayakkabılarını rahat giyebiliyor", variable=shoes_cb_var).grid(row=13, column=1, sticky="W")
tk.Checkbutton(root, text="Hijyenini kendisi rahatça sağlayabiliyor", variable=hygiene_cb_var).grid(row=14, column=1,
                                                                                                    sticky="W")

# Hesapla Butonu
calculate_button = tk.Button(root, text="Hesapla", command=calculate_score)
calculate_button.grid(row=15, column=0, columnspan=2, pady=20)

# Sonuç Etiketi
result_label = tk.Label(root, text="")
result_label.grid(row=16, column=0, columnspan=2)

# Arayüzü Başlat
root.mainloop()

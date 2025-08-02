#!/usr/bin/env python3
import os
import tempfile
import wave
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from tkinter.scrolledtext import ScrolledText

import sounddevice as sd
import soundfile as sf
from openai import OpenAI, OpenAIError

# Ensure OPENAI_API_KEY is set
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    messagebox.showerror("API Key Error", "Please set OPENAI_API_KEY environment variable.")
    exit(1)
client = OpenAI(api_key=API_KEY)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FORMAT = 'int16'

# Language options
LANG_OPTIONS = ["tr", "en", "ar", "es", "de", "fr"]

class PatientSumApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PatientSum")
        self.geometry("900x600")
        ctk.set_appearance_mode("light")
        self.configure(fg_color="white")

        # Main grid: two columns for transcript/summary
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top control bar
        ctrl = ctk.CTkFrame(self, fg_color="#e6f2ff")
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        # Prevent stretching of our 5 columns
        for i in range(5):
            ctrl.grid_columnconfigure(i, weight=0)

        # Language selector
        ctk.CTkLabel(ctrl, text="Language:", text_color="#003366")\
            .grid(row=0, column=0, padx=(5,2))
        self.lang = tk.StringVar(value="en")
        ctk.CTkOptionMenu(
            ctrl,
            variable=self.lang,
            values=LANG_OPTIONS,
            fg_color="#003366",
            button_color="white",
            button_hover_color="#e6f2ff",
            text_color="white"
        ).grid(row=0, column=1, padx=(0,10))

        # Control buttons: Start / Stop / Summarize
        self.start_btn = ctk.CTkButton(
            ctrl, text="Start",
            command=self.start_recording,
            fg_color="#003366", hover_color="#005599"
        )
        self.stop_btn = ctk.CTkButton(
            ctrl, text="Stop",
            command=self.stop_recording,
            state="disabled",
            fg_color="#003366", hover_color="#005599"
        )
        self.sum_btn = ctk.CTkButton(
            ctrl, text="Summarize",
            command=self.generate_summary,
            state="disabled",
            fg_color="#003366", hover_color="#005599"
        )

        self.start_btn.grid(row=0, column=2, padx=5)
        self.stop_btn.grid(row=0, column=3, padx=5)
        self.sum_btn.grid(row=0, column=4, padx=5)

        # Transcript panel (left)
        lbl_trans = ctk.CTkLabel(
            self, text="Transcript:",
            font=("Helvetica", 18), text_color="#003366"
        )
        lbl_trans.grid(row=1, column=0, sticky="nw", padx=10)
        self.transcript_box = ScrolledText(
            self, wrap=tk.WORD, font=("Helvetica", 18),
            bg="white", fg="#003366"
        )
        self.transcript_box.grid(
            row=2, column=0, sticky="nsew",
            padx=10, pady=(0,5)
        )

        # Summary panel (right)
        lbl_sum = ctk.CTkLabel(
            self, text="Summary:",
            font=("Helvetica", 18), text_color="#003366"
        )
        lbl_sum.grid(row=1, column=1, sticky="nw", padx=10)
        self.summary_box = ScrolledText(
            self, wrap=tk.WORD, font=("Helvetica", 18),
            bg="white", fg="#003366"
        )
        self.summary_box.grid(
            row=2, column=1, sticky="nsew",
            padx=10, pady=(0,5)
        )

        self.audio_frames = []

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_frames.append(bytes(indata))

    def start_recording(self):
        self.transcript_box.delete('1.0', tk.END)
        self.summary_box.delete('1.0', tk.END)
        self.audio_frames.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.sum_btn.configure(state="disabled")
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            dtype=AUDIO_FORMAT,
            channels=CHANNELS,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_recording(self):
        self.stream.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.sum_btn.configure(state="normal")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.audio_frames))

        # Always use online Whisper API
        transcript = ""
        try:
            with open(wav_path, 'rb') as f:
                resp = client.audio.transcriptions.create(
                    file=f,
                    model="whisper-1",
                    response_format="text",
                    language=self.lang.get()
                )
            transcript = resp.strip()
        except OpenAIError as e:
            messagebox.showerror("Whisper API Error", str(e))

        self.transcript_box.insert(tk.END, transcript)
        os.remove(wav_path)

    def generate_summary(self):
        text = self.transcript_box.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("No Transcript", "Please record first.")
            return

        self.summary_box.delete('1.0', tk.END)
        self.summary_box.insert(tk.END, "Generating summary...\n")

        # Choose system prompt based on selected language
        lang = self.lang.get()
        if lang == "tr":
            system_msg = "Sen tıp alanında yardımcı bir asistansın. Yalnızca Türkçe cevap ver."
        else:
            system_msg = "You are a helpful medical assistant. Please respond in English only."

        prompt = (
            "Summarize the following patient note, including:\n"
            "1. Key history and exam findings.\n"
            "2. Two to three possible differential diagnoses.\n"
            "3. Recommended laboratory or imaging tests.\n\n"
            f"Patient transcription:\n{text}"
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            summary = resp.choices[0].message.content
        except OpenAIError as e:
            summary = f"API error: {e}"

        self.summary_box.delete('1.0', tk.END)
        self.summary_box.insert(tk.END, summary)


if __name__ == "__main__":
    app = PatientSumApp()
    app.mainloop()

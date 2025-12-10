import sounddevice as sd
import numpy as np
import time
import re
import unicodedata
from scipy.io import wavfile
from scipy import signal
from faster_whisper import WhisperModel
from fuzzywuzzy import fuzz
import json
import os
from pyrnnoise import RNNoise

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class ElderVoiceAssistant:

    def __init__(self):
        self.sample_rate = 16000

        print("\n=== Loading Whisper Model ===\n")
        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8"
        )
        print("Whisper Model Loaded Successfully\n")

        # Initialize RNNoise safely
        try:
            self.denoiser = RNNoise()
            print("RNNoise Denoiser Initialized\n")
        except Exception as e:
            print(f"⚠ RNNoise failed to initialize: {e}")
            self.denoiser = None

        # Load correction dictionary
        try:
            with open("corrections.json", "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.corrections = data
                else:
                    self.corrections = []
            print("Correction Dictionary Loaded\n")
        except:
            print("⚠ corrections.json not found — continuing without corrections")
            self.corrections = []

        # INTENTS
        self.intents = {
            "MEDICINE": ["dawai", "dava", "tablet", "goli"],
            "CALL_FAMILY": ["puttar", "putar", "phone karo", "call karo", "beta"],
            "FOOD": ["chai", "chay", "doodh", "garam", "khana"],
            "TIME": ["time", "baje", "kitne"]
        }

        # ACCENT NORMALIZATION
        self.accent_map = {
            r"put[ae]r": "puttar",
            r"puter": "puttar",
            r"dava[iy]": "dawai",
            r"daabaee": "dawai",
            r"chaii?": "chai",
            r"f[ao]n": "phone",
            r"kara": "karo"
        }

    # -------------------------------------------------------
    # AUDIO RECORDING
    # -------------------------------------------------------
    def record_audio(self):
        print("\n🎤 बोलो 10cm से (Punjabi supported)...")
        audio = sd.rec(
            int(5 * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        return self.preprocess_audio(audio.flatten())

    # -------------------------------------------------------
    # RNNoise NOISE REDUCTION
    # -------------------------------------------------------
    def denoise_RNNoise(self, audio):
        if self.denoiser is None:
            return audio
        audio = np.clip(audio, -1.0, 1.0)
        return self.denoiser.filter(audio)

    # -------------------------------------------------------
    # PREPROCESSING (normalize + denoise + high-pass)
    # -------------------------------------------------------
    def preprocess_audio(self, audio):
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        # Denoise
        audio = self.denoise_RNNoise(audio)
        # High-pass filter
        b, a = signal.butter(1, 100 / (self.sample_rate / 2), "highpass")
        audio = signal.filtfilt(b, a, audio)
        return audio

    # -------------------------------------------------------
    # SAVE AUDIO
    # -------------------------------------------------------
    def save_audio(self, audio, filename):
        audio = np.clip(audio, -1.0, 1.0)
        wavfile.write(filename, self.sample_rate, (audio * 32767).astype(np.int16))
        return filename

    # -------------------------------------------------------
    # TRANSCRIPTION (Punjabi only)
    # -------------------------------------------------------
    def transcribe(self, audio_file):
        segments, _ = self.model.transcribe(audio_file, language="pa", beam_size=5)
        text = " ".join([s.text for s in segments]).strip()
        return self.clean_text(text)

    # -------------------------------------------------------
    # TEXT CLEANING
    # -------------------------------------------------------
    def clean_text(self, text):
        if not text or len(text.strip()) == 0:
            return "NO_SPEECH"
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(.)\1{3,}", r"\1", text)
        text = re.sub(r"(अगर|ॐ|वाय|om|mm)+", "", text, flags=re.IGNORECASE)
        return text.lower().strip() if len(text.strip()) >= 2 else "NO_SPEECH"

    # -------------------------------------------------------
    # AUTO CORRECT
    # -------------------------------------------------------
    def auto_correct(self, sentence):
        fixed = sentence
        for corr in self.corrections:
            if isinstance(corr, dict):
                wrong = corr.get("wrong", "")
                right = corr.get("right", "")
                fixed = fixed.replace(wrong, right)
        return fixed

    # -------------------------------------------------------
    # ACCENT NORMALIZATION
    # -------------------------------------------------------
    def normalize_accent(self, text):
        s = text
        for pattern, repl in self.accent_map.items():
            s = re.sub(pattern, repl, s)
        return s

    # -------------------------------------------------------
    # INTENT DETECTION
    # -------------------------------------------------------
    def detect_intent(self, text):
        if text == "NO_SPEECH":
            return "NO_SPEECH"
        for intent, keys in self.intents.items():
            for k in keys:
                if fuzz.partial_ratio(k, text) >= 70:
                    return intent
        return "UNKNOWN"

    # -------------------------------------------------------
    # MAIN PIPELINE
    # -------------------------------------------------------
    def process_once(self):
        audio = self.record_audio()
        filename = f"elder_{time.time()}.wav"
        self.save_audio(audio, filename)
        print("\n🧠 Processing...")
        clean_text = self.transcribe(filename)
        corrected = self.auto_correct(clean_text)
        normalized = self.normalize_accent(corrected)
        intent = self.detect_intent(normalized)
        print("\n" + "=" * 60)
        print(f"📝 Raw:            {clean_text}")
        print(f"✏ Corrected:       {corrected}")
        print(f"🔧 Normalized:      {normalized}")
        print(f"🎯 Intent:          {intent}")
        print("=" * 60)


# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
if __name__ == "__main__":
    assistant = ElderVoiceAssistant()
    while True:
        assistant.process_once()
        if input("\n🔄 फिर से? (y/n): ").lower() != "y":
            break

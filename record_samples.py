import sounddevice as sd
from scipy.io.wavfile import write
import os

os.makedirs("dataset/real", exist_ok=True)

print("Recording 20 samples — say anything for 3 seconds each")
print("Press Enter before each recording...")

for i in range(20):
    input(f"Press Enter to record sample {i+1}/20...")
    print("Recording... speak now!")
    audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1)
    sd.wait()
    write(f"dataset/real/real_{i+1}.wav", 16000, audio)
    print(f"✅ Saved sample {i+1}")

print("Done! All 20 samples saved.")# updated

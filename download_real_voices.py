import os
import requests

# folder to save real voices
folder = "dataset/real"
os.makedirs(folder, exist_ok=True)

urls = [
"https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav",
"https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav",
"https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav",
"https://www2.cs.uic.edu/~i101/SoundFiles/ImperialMarch60.wav"
]

count = 0

for url in urls:
    try:
        r = requests.get(url)
        filename = os.path.join(folder, f"real_{count}.wav")

        with open(filename, "wb") as f:
            f.write(r.content)

        print("Downloaded:", filename)
        count += 1

    except:
        print("Failed:", url)

print("Finished downloading", count, "files")# updated

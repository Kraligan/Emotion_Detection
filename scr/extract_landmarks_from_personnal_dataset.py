import os
import numpy as np
from utils import get_face_landmarks

# Dossier contenant les images organisées par classe
dataset_dir = "data/my_dataset"

# Émotions (doivent correspondre aux noms de sous-dossiers)
emotions = ['fear', 'happy', 'neutral', 'sad']
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}

# Stockera les features + label
landmarks_data = []

for emotion in emotions:
    emotion_dir = os.path.join(dataset_dir, emotion)
    if not os.path.isdir(emotion_dir):
        print(f"Dossier manquant : {emotion_dir}")
        continue

    for img_name in os.listdir(emotion_dir):
        img_path = os.path.join(emotion_dir, img_name)
        if not img_path.lower().endswith((".jpg", ".png")):
            continue

        landmarks = get_face_landmarks(img_path)
        if np.sum(landmarks) == 0:
            print(f"Landmarks non détectés : {img_path}")
            continue

        sample = np.append(landmarks, emotion_to_idx[emotion])
        landmarks_data.append(sample)

        print(f"{img_path} -> {emotion}")

# Conversion et sauvegarde
landmarks_data = np.array(landmarks_data)
os.makedirs("data/processed", exist_ok=True)
np.savetxt("data/processed/dataset_personal.txt", landmarks_data)

print(f"\nExtraction terminée : {len(landmarks_data)} images traitées.")
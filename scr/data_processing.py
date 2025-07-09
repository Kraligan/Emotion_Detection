# Imports
import kagglehub
import os
import numpy as np
import cv2
from utils import get_face_landmarks

# Dowload dataset affectnet: (Uncomment to download)
# path = kagglehub.dataset_download("mstjebashazida/affectnet")
# print("Path to dataset files:", path)

# Base path (après avoir déplacé les dossiers Train/ et Test/ au bon endroit)
base_path = os.path.expanduser("~/.cache/kagglehub/datasets/mstjebashazida/affectnet/versions/1")
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")

# Configuration
max_samples_per_class = 3000
output_path = "data/dataset_Affectnet_balanced_4classes.txt"

# Liste des émotions communes aux deux dossiers
emotions = sorted(list(set(os.listdir(train_path)).intersection(os.listdir(test_path))))

# Fusionner et trier les fichiers
def get_all_image_paths(emotion_folder):
    paths = []
    for root in [train_path, test_path]:
        folder = os.path.join(root, emotion_folder)
        if os.path.isdir(folder):
            images = sorted(os.listdir(folder))[:4000]  # max pour éviter trop de scans
            for img in images:
                paths.append(os.path.join(folder, img))
    return paths

# Extraction
landmarks_dataset = []
for emotion_idx, emotion in enumerate(emotions):
    print(f"Processing emotion: {emotion}")
    count = 0
    image_paths = get_all_image_paths(emotion)

    for img_path in image_paths:
        if count >= max_samples_per_class:
            break
        try:
            img_landmarks = get_face_landmarks(img_path)
            if img_landmarks is not None and len(img_landmarks) == 1404:
                data_row = np.append(img_landmarks, int(emotion_idx))
                landmarks_dataset.append(data_row)
                count += 1
        except Exception as e:
            print(f"Failed: {img_path} — {e}")
    
    print(f"{count} samples collected for {emotion}")

# Save
np.savetxt(output_path, np.array(landmarks_dataset))
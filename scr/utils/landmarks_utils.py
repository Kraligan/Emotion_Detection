import numpy as np
import cv2
import mediapipe as mp

# Initialiser MediaPipe Face Mesh une seule fois
def init_mediapipe():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    return face_mesh

face_mesh = init_mediapipe()

# mediapipe extract landmarks function
def get_face_landmarks(image_path:str):
    """
    Loads an image, detects facial landmarks using MediaPipe, and 
    returns a 1D NumPy array of 1,404 values (468 points with x, y, z coordinates).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return np.zeros(1404)  # Image non chargée → vecteur vide

    # Convertir BGR vers RGB pour MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détection
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return np.zeros(1404)  # Aucun visage détecté → vecteur vide

    # Extraire les coordonnées des 468 points
    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

    return coords
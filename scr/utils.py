
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt

# Initialiser MediaPipe Face Mesh une seule fois
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

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

def create_mlp_model(input_dim=1404, num_classes=7):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    return model


    return model

def load_model(model_path:str):
    return tf.keras.models.load_model(model_path)


# Plot training curves
def plot_training_curves(history):
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Val acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('model/training_curves.png')
    plt.show()

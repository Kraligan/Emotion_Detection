# import
from utils import load_model
import cv2
import numpy as np
import joblib
import mediapipe as mp
import imageio
import os

# load model and init mediapipe facemesh
model = load_model('model/fine_tune2/affectnet_finetuned_on_self2.h5')

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

emotion_classes = ['fear', 'happy', 'neutral', 'sad']
scaler = joblib.load('model/fine_tune2/scaler_finetune.pkl')

# Open webcam
cap = cv2.VideoCapture(0)

frames = []
MAX_FRAMES = 60 # For recording

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for natural webcam feel
    # frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect landmarks
    results = face_mesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract 468 landmarks (x, y, z)
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            if len(landmarks) == 1404:
                input_data = scaler.transform([landmarks])

                # Predict emotion
                prediction = model.predict(input_data, verbose=0)
                class_id = np.argmax(prediction)
                emotion = emotion_classes[class_id]
                confidence = np.max(prediction)

                # Draw result
                print("Probas:", dict(zip(emotion_classes, np.round(prediction[0], 2))))
                cv2.putText(frame, f'{emotion} ({confidence:.2f})', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                # Draw landmarks on face
                for lm in face_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if len(frames) < MAX_FRAMES:
        frames.append(rgb)

    cv2.imshow('Emotion Detection (landmarks)', frame)
    # cv2.waitKey(25)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# os.makedirs("assets", exist_ok=True)
# imageio.mimsave("assets/emotion_demo.gif", frames, fps=10)
# print("GIF saved to assets/emotion_demo.gif")
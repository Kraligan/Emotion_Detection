# import
from utils import load_model, init_mediapipe
import cv2
import numpy as np
import joblib

# load model and init mediapipe facemesh
model = load_model('model/affectnet2/best_model.h5')
face_mesh = init_mediapipe()
emotion_classes = ['disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
scaler = joblib.load('model/affectnet_opti/scaler_affectnet.pkl')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for natural webcam feel
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect landmarks
    results = face_mesh.process(rgb)

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
                cv2.putText(frame, f'{emotion} ({confidence:.2f})', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks on face
                for lm in face_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    cv2.imshow('Emotion Detection (landmarks)', frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
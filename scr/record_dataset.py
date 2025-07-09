import cv2
import os

emotions = ['fear', 'happy', 'neutral', 'sad']

# Dossier de sortie
output_dir = "data/my_dataset"
os.makedirs(output_dir, exist_ok=True)

# Nombre d’images par classe
images_per_class = 50

cap = cv2.VideoCapture(0)

for emotion in emotions:
    print(f"\nPrépare-toi pour : {emotion.upper()}")
    input("Appuie sur Entrée pour commencer...")

    emotion_dir = os.path.join(output_dir, emotion)
    os.makedirs(emotion_dir, exist_ok=True)

    count = 0
    while count < images_per_class:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        cv2.putText(display, f"{emotion} ({count+1}/{images_per_class})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Capture Emotion", display)

        key = cv2.waitKey(100)
        if key == ord('s') or key == 32:  # Espace ou 's'
            img_path = os.path.join(emotion_dir, f"{emotion}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1

        if key == ord('q'):
            break

print("Capture terminée.")
cap.release()
cv2.destroyAllWindows()
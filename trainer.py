import cv2
import numpy as np
import os
from PIL import Image
import csv

def train_model():
    path = 'TrainingImage'
    if not os.path.exists(path):
        print("❌ TrainingImage folder not found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []
    label_dict = {}

    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            parts = filename.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                name = parts[0]
                id_ = int(parts[1])
                img_path = os.path.join(path, filename)

                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_np = np.array(img, 'uint8')

                faces.append(img_np)
                ids.append(id_)
                label_dict[id_] = name

    if not faces:
        print("❌ No valid images found for training.")
        return

    # Train the recognizer
    recognizer.train(faces, np.array(ids))
    os.makedirs("TrainingImageLabel", exist_ok=True)
    recognizer.save("TrainingImageLabel/Trainer.yml")
    print("✅ Trainer.yml saved.")

    # Save student details to CSV
    os.makedirs("StudentDetails", exist_ok=True)
    with open("StudentDetails/StudentDetails.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Name"])
        for id_, name in label_dict.items():
            writer.writerow([id_, name])
    print("✅ StudentDetails.csv updated with labels.")

if __name__ == "__main__":
    train_model()

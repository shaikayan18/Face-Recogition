# flask_server.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
import csv
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Paths
DATASET_DIR = "dataset"
TRAINER_PATH = "TrainingImageLabel/Trainer.yml"
CSV_PATH = "StudentDetails/StudentDetails.csv"
ATTENDANCE_DIR = "Attendance"

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Label dictionary
def load_label_dict():
    label_dict = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2 and row[0].isdigit():
                    label_dict[int(row[0])] = row[1]
    return label_dict

@app.route('/register_image', methods=['POST'])
def register_image():
    try:
        data = request.get_json()
        username = data.get('username')
        image_data = data.get('image')

        if not username or not image_data:
            return jsonify({'message': 'Invalid data'}), 400

        # Assign ID based on CSV
        next_id = 1
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row[0].isdigit():
                        next_id = max(next_id, int(row[0]) + 1)

        user_id = next_id
        user_dir = os.path.join(DATASET_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)

        # Decode image
        img_bytes = base64.b64decode(image_data.split(',')[1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Save image
        count = len(os.listdir(user_dir)) + 1
        filename = os.path.join(user_dir, f"{username}_{count}.jpg")
        cv2.imwrite(filename, frame)

        # Write to CSV
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name"])
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([user_id, username])

        return jsonify({"message": f"Image saved for {username}"})

    except Exception as e:
        print("❌ Error in register_image:", e)
        return jsonify({"message": "Server error"}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        ids = []

        for user_id in os.listdir(DATASET_DIR):
            user_path = os.path.join(DATASET_DIR, user_id)
            if not os.path.isdir(user_path):
                continue
            for image_file in os.listdir(user_path):
                img_path = os.path.join(user_path, image_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                ids.append(int(user_id))

        if not faces:
            return jsonify({"message": "No images to train."}), 400

        recognizer.train(faces, np.array(ids))
        recognizer.save(TRAINER_PATH)
        return jsonify({"message": "Model trained successfully."})

    except Exception as e:
        print("❌ Error in train_model:", e)
        return jsonify({"message": "Training failed."}), 500

@app.route('/receive_image', methods=['POST'])
def receive_image():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
        label_dict = load_label_dict()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        img_bytes = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        id_ = "Unknown"
        name = "Unknown"

        for (x, y, w, h) in faces:
            id_raw, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 70:
                id_ = id_raw
                name = label_dict.get(id_, f"ID_{id_}")
            break

        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H:%M:%S')
        filename = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Name", "Date", "Time"])
            writer.writerow([id_, name, date_str, time_str])

        return jsonify({"message": f"Attendance marked for {name}"})

    except Exception as e:
        print("❌ Error in receive_image:", e)
        return jsonify({"message": "Internal server error"}), 500

@app.route('/test', methods=['GET'])
def test():
    return "✅ Flask server is up and reachable!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

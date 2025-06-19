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

# Load the Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = "TrainingImageLabel/Trainer.yml"
if not os.path.exists(trainer_path):
    raise FileNotFoundError("Trainer.yml not found. Please run training first.")
recognizer.read(trainer_path)

# Load ID-to-name mapping from StudentDetails.csv
label_dict = {}
csv_path = "StudentDetails/StudentDetails.csv"
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2 and row[0].isdigit():
                label_dict[int(row[0])] = row[1]
else:
    raise FileNotFoundError("StudentDetails.csv not found.")

@app.route('/receive_image', methods=['POST'])
def receive_image():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode the image from base64
        image_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        id_ = "Unknown"
        name = "Unknown"
        confidence = 1000

        for (x, y, w, h) in faces:
            id_raw, conf = recognizer.predict(gray[y:y+h, x:x+w])
            print(f"Prediction: ID={id_raw}, Confidence={conf:.2f}")
            if conf < 70:
                id_ = id_raw
                name = label_dict.get(id_, f"ID_{id_}")
                confidence = conf
            else:
                name = "Unknown"
            break  # Process only one face

        # Write attendance
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H:%M:%S')
        filename = f"Attendance/Attendance_{date_str}.csv"

        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")

        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Name", "Date", "Time"])
            writer.writerow([id_, name, date_str, time_str])

        print(f"📝 Attendance marked: {id_}, {name}, {date_str}, {time_str}")
        return jsonify({"message": f"Attendance marked for {name}"}), 200

    except Exception as e:
        print("❌ Error in receive_image:", e)
        return jsonify({"message": "Internal server error"}), 500

@app.route('/test', methods=['GET'])
def test():
    return "✅ Flask server is up and reachable!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

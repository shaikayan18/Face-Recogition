# flask_server.py
from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
import csv
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition
import dlib

app = Flask(__name__)
CORS(app)

# Paths
DATASET_DIR = "dataset"
EMBEDDINGS_PATH = "TrainingImageLabel/embeddings.pkl"
CSV_PATH = "StudentDetails/StudentDetails.csv"
ATTENDANCE_DIR = "Attendance"

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# MagFace Model Architecture
class MagFace(nn.Module):
    def __init__(self, embedding_size=512):
        super(MagFace, self).__init__()
        # Simplified backbone (you can use ResNet or other backbones)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return nn.functional.normalize(x, p=2, dim=1)

# Initialize MagFace model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
magface_model = MagFace().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Face detection using dlib (more accurate than OpenCV)
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except:
    # Fallback to OpenCV if dlib model not available
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detector = None
    predictor = None

def detect_and_align_face(image):
    """Detect and align face using dlib or OpenCV"""
    if detector is not None:
        # Use dlib for better face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_img = image[y:y+h, x:x+w]
            return cv2.resize(face_img, (112, 112))
    else:
        # Fallback to OpenCV
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = image[y:y+h, x:x+w]
            return cv2.resize(face_img, (112, 112))
    
    return None

def extract_face_embedding(face_image):
    """Extract face embedding using MagFace model"""
    try:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Preprocess
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = magface_model(input_tensor)
            return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def load_label_dict():
    """Load label dictionary from CSV"""
    label_dict = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 2 and row[0].isdigit():
                    label_dict[int(row[0])] = row[1]
    return label_dict

def load_embeddings():
    """Load stored embeddings"""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(embeddings):
    """Save embeddings to file"""
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

@app.route('/register_image', methods=['POST'])
def register_image():
    try:
        data = request.get_json()
        username = data.get('username')
        image_data = data.get('image')

        if not username or not image_data:
            return jsonify({'message': 'Invalid data'}), 400

        username = username.replace("@", "_at_").replace(".", "_")

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

        # Detect and align face
        face_img = detect_and_align_face(frame)
        if face_img is None:
            return jsonify({'message': 'No face detected in the image'}), 400

        # Extract embedding
        embedding = extract_face_embedding(face_img)
        if embedding is None:
            return jsonify({'message': 'Failed to extract face embedding'}), 500

        # Save face image
        count = len(os.listdir(user_dir)) + 1
        filename = os.path.join(user_dir, f"{username}_{count}.jpg")
        cv2.imwrite(filename, face_img)

        # Save embedding
        embeddings = load_embeddings()
        if user_id not in embeddings:
            embeddings[user_id] = []
        embeddings[user_id].append(embedding)
        save_embeddings(embeddings)

        # Write to CSV
        if not os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name"])
        
        # Check if user already exists
        user_exists = False
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 2 and row[1] == username:
                        user_exists = True
                        break
        
        if not user_exists:
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_id, username])

        return jsonify({"message": f"Face registered successfully for {username}"})

    except Exception as e:
        print("❌ Error in register_image:", e)
        return jsonify({"message": "Server error"}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        embeddings = load_embeddings()
        
        if not embeddings:
            return jsonify({"message": "No embeddings found. Please register faces first."}), 400

        # Calculate average embeddings for each user
        avg_embeddings = {}
        for user_id, user_embeddings in embeddings.items():
            if user_embeddings:
                avg_embeddings[user_id] = np.mean(user_embeddings, axis=0)

        # Save average embeddings
        with open(EMBEDDINGS_PATH.replace('.pkl', '_avg.pkl'), 'wb') as f:
            pickle.dump(avg_embeddings, f)

        return jsonify({"message": f"Model trained successfully with {len(avg_embeddings)} users."})

    except Exception as e:
        print("❌ Error in train_model:", e)
        return jsonify({"message": "Training failed."}), 500

@app.route('/receive_image', methods=['POST'])
def receive_image():
    try:
        # Load average embeddings
        avg_embeddings_path = EMBEDDINGS_PATH.replace('.pkl', '_avg.pkl')
        if not os.path.exists(avg_embeddings_path):
            return jsonify({'message': 'Model not trained. Please train the model first.'}), 400

        with open(avg_embeddings_path, 'rb') as f:
            avg_embeddings = pickle.load(f)

        label_dict = load_label_dict()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode image
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Detect and align face
        face_img = detect_and_align_face(frame)
        if face_img is None:
            return jsonify({'message': 'No face detected in the image'}), 400

        # Extract embedding
        query_embedding = extract_face_embedding(face_img)
        if query_embedding is None:
            return jsonify({'message': 'Failed to extract face embedding'}), 500

        # Find best match
        best_match_id = None
        best_similarity = -1
        threshold = 0.6  # Cosine similarity threshold

        for user_id, stored_embedding in avg_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                stored_embedding.reshape(1, -1)
            )[0][0]

            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match_id = user_id

        # Determine identity
        if best_match_id is not None:
            id_ = best_match_id
            name = label_dict.get(id_, f"ID_{id_}")
            confidence = round(best_similarity * 100, 2)
        else:
            id_ = "Unknown"
            name = "Unknown"
            confidence = 0

        # Save attendance
        date_str = datetime.now().strftime('%Y-%m-%d')
        time_str = datetime.now().strftime('%H:%M:%S')
        filename = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Name", "Date", "Time", "Confidence"])
            writer.writerow([id_, name, date_str, time_str, f"{confidence}%"])

        return jsonify({
            "message": f"Attendance marked for {name} (Confidence: {confidence}%)",
            "confidence": confidence
        })

    except Exception as e:
        print("❌ Error in receive_image:", e)
        return jsonify({"message": "Internal server error"}), 500

@app.route('/dataset/<path:subpath>')
def serve_image(subpath):
    return send_from_directory(DATASET_DIR, subpath)

@app.route('/get_embeddings_info', methods=['GET'])
def get_embeddings_info():
    """Get information about stored embeddings"""
    try:
        embeddings = load_embeddings()
        label_dict = load_label_dict()
        
        info = {}
        total_embeddings = 0
        
        for user_id, user_embeddings in embeddings.items():
            name = label_dict.get(user_id, f"ID_{user_id}")
            info[name] = len(user_embeddings)
            total_embeddings += len(user_embeddings)
        
        return jsonify({
            "total_users": len(embeddings),
            "total_embeddings": total_embeddings,
            "users": info
        })
    
    except Exception as e:
        print("❌ Error in get_embeddings_info:", e)
        return jsonify({"message": "Error retrieving embeddings info"}), 500

@app.route('/test', methods=['GET'])
def test():
    return "✅ MagFace Flask server is up and reachable!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

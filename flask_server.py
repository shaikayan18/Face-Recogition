from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
import os
import csv
import pickle
from datetime import datetime
from flask_cors import CORS
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Initialize FaceNet
facenet = FaceNet()

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

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_label_dict():
    """Load user ID to name mapping from CSV"""
    label_dict = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_dict[int(row['ID'])] = row['Name']
    return label_dict

def extract_face_embedding(image):
    """Extract face embedding using FaceNet"""
    try:
        # Detect face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
            
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
        face_img = image[y:y+h, x:x+w]
        
        # Resize to 160x160 for FaceNet
        face_resized = cv2.resize(face_img, (160, 160))
        face_array = np.expand_dims(face_resized, axis=0)
        
        # Extract embedding
        embedding = facenet.embeddings(face_array)
        return embedding[0]
        
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

@app.route('/save_image', methods=['POST'])
def save_image():
    try:
        data = request.get_json()
        username = data.get('username')
        image_data = data.get('image')
        
        if not username or not image_data:
            return jsonify({"message": "Username and image required"}), 400

        # Get next user ID
        next_id = 1
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, 'r') as f:
                reader = csv.DictReader(f)
                ids = [int(row['ID']) for row in reader if row['ID'].isdigit()]
                if ids:
                    next_id = max(ids) + 1

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
        print("❌ Error in save_image:", e)
        return jsonify({"message": "Failed to save image"}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        embeddings_db = {}
        
        for user_id in os.listdir(DATASET_DIR):
            user_path = os.path.join(DATASET_DIR, user_id)
            if not os.path.isdir(user_path):
                continue
                
            user_embeddings = []
            
            for image_file in os.listdir(user_path):
                img_path = os.path.join(user_path, image_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                embedding = extract_face_embedding(img)
                if embedding is not None:
                    user_embeddings.append(embedding)
            
            if user_embeddings:
                # Average embeddings for this user
                avg_embedding = np.mean(user_embeddings, axis=0)
                embeddings_db[int(user_id)] = avg_embedding

        if not embeddings_db:
            return jsonify({"message": "No valid images found for training"}), 400

        # Save embeddings
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings_db, f)
            
        return jsonify({"message": f"Model trained successfully with {len(embeddings_db)} users"})

    except Exception as e:
        print("❌ Error in train_model:", e)
        return jsonify({"message": "Training failed"}), 500

@app.route('/receive_image', methods=['POST'])
def receive_image():
    try:
        # Load embeddings
        if not os.path.exists(EMBEDDINGS_PATH):
            return jsonify({"message": "No trained model found. Please train first."}), 400
            
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings_db = pickle.load(f)
            
        label_dict = load_label_dict()

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode image
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Extract embedding
        current_embedding = extract_face_embedding(frame)
        if current_embedding is None:
            return jsonify({"message": "No face detected"}), 400

        # Find best match
        best_match_id = None
        best_similarity = 0
        threshold = 0.6
        
        for user_id, stored_embedding in embeddings_db.items():
            similarity = cosine_similarity([current_embedding], [stored_embedding])[0][0]
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match_id = user_id

        if best_match_id is not None:
            name = label_dict.get(best_match_id, f"ID_{best_match_id}")
            
            # Save attendance
            date_str = datetime.now().strftime('%Y-%m-%d')
            time_str = datetime.now().strftime('%H:%M:%S')
            filename = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")

            file_exists = os.path.isfile(filename)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["ID", "Name", "Date", "Time"])
                writer.writerow([best_match_id, name, date_str, time_str])

            return jsonify({"message": f"Attendance marked for {name} (confidence: {best_similarity:.2f})"})
        else:
            return jsonify({"message": "Face not recognized"})

    except Exception as e:
        print("❌ Error in receive_image:", e)
        return jsonify({"message": "Internal server error"}), 500

@app.route('/download_csv')
def download_csv():
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"Attendance_{date_str}.csv"
        return send_from_directory(ATTENDANCE_DIR, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

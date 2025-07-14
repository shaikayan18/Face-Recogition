import cv2
import os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tkinter.messagebox as mess
import tkinter.simpledialog as simpledialog
import urllib.request
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class AttendanceBackend:
    def __init__(self):
        self.haarcascade_path = "haarcascade_frontalface_default.xml"
        self.training_image_path = "TrainingImage/"
        self.training_label_path = "TrainingImageLabel/"
        self.student_details_path = "StudentDetails/"
        self.attendance_path = "Attendance/"
        
        # Create directories
        self.assure_path_exists(self.training_image_path)
        self.assure_path_exists(self.training_label_path)
        self.assure_path_exists(self.student_details_path)
        self.assure_path_exists(self.attendance_path)
        
        # Download haarcascade if not present
        self.ensure_haarcascade_exists()
        
    def assure_path_exists(self, path):
        """Ensure directory exists, create if not"""
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                logging.info(f"Created directory: {path}")
        except Exception as e:
            logging.error(f"Error creating directory {path}: {e}")
    
    def ensure_haarcascade_exists(self):
        """Download haarcascade file if it doesn't exist"""
        if not os.path.isfile(self.haarcascade_path):
            try:
                logging.info("Haarcascade file not found. Downloading...")
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, self.haarcascade_path)
                logging.info("Haarcascade file downloaded successfully")
            except Exception as e:
                logging.error(f"Error downloading haarcascade file: {e}")
                return False
        return True
    
    def check_haarcascade_file(self):
        """Check if haarcascade file exists"""
        exists = os.path.isfile(self.haarcascade_path)
        logging.info(f"Haarcascade file exists: {exists}")
        return exists
    
    def test_camera(self):
        """Test camera connection"""
        try:
            logging.info("Testing camera connection...")
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                logging.error("Camera not accessible")
                cam.release()
                return False
            
            ret, frame = cam.read()
            cam.release()
            cv2.destroyAllWindows()
            
            if not ret:
                logging.error("Camera opened but couldn't read frame")
                return False
                
            logging.info("Camera test successful")
            return True
        except Exception as e:
            logging.error(f"Camera test failed: {e}")
            return False
    
    # FRONTEND COMPATIBILITY METHODS
    def get_registration_count(self):
        """Get total number of registrations - FRONTEND COMPATIBLE"""
        return self.get_total_registrations()
    
    def validate_student_id(self, student_id):
        """Validate student ID format - FRONTEND COMPATIBLE"""
        return self.is_valid_id(student_id)
    
    def student_exists(self, student_id):
        """Check if student already exists - FRONTEND COMPATIBLE"""
        student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
        if os.path.isfile(student_file):
            try:
                df = pd.read_csv(student_file)
                return student_id in df['ID'].astype(str).values
            except Exception as e:
                logging.error(f"Error checking student existence: {e}")
                return False
        return False
    
    def capture_images(self, student_id, student_name):
        """Capture face images for training - FRONTEND COMPATIBLE"""
        try:
            logging.info(f"Starting image capture for {student_name} ({student_id})")
            
            # Validate inputs
            if not self.is_valid_id(student_id):
                logging.error(f"Invalid student ID format: {student_id}")
                return False
            
            if not self.is_valid_name(student_name):
                logging.error(f"Invalid student name format: {student_name}")
                return False
            
            # Check if student already exists
            if self.student_exists(student_id):
                logging.error(f"Student {student_id} already exists")
                return False
            
            success, message = self.capture_images_internal(student_id, student_name)
            logging.info(f"Image capture result: {success}, Message: {message}")
            return success
        except Exception as e:
            logging.error(f"Error in capture_images: {e}")
            return False
    
    def train_model(self):
        """Train the face recognition model - FRONTEND COMPATIBLE"""
        try:
            logging.info("Starting model training")
            success, message = self.train_images()
            logging.info(f"Training result: {success}, Message: {message}")
            return success
        except Exception as e:
            logging.error(f"Error in train_model: {e}")
            return False
    
    def take_attendance(self):
        """Take attendance using face recognition - FRONTEND COMPATIBLE"""
        try:
            logging.info("Starting attendance taking")
            success, message, attendance_data = self.take_attendance_internal()
            logging.info(f"Attendance result: {success}, Message: {message}")
            return success
        except Exception as e:
            logging.error(f"Error in take_attendance: {e}")
            return False
    
    def get_attendance_records(self):
        """Get attendance records for display - FRONTEND COMPATIBLE"""
        try:
            records = []
            today_attendance = self.get_today_attendance()
            
            for record in today_attendance:
                records.append((
                    record['id'],
                    record['name'], 
                    record['date'],
                    record['time']
                ))
            
            return records
        except Exception as e:
            logging.error(f"Error getting attendance records: {e}")
            return []
    
    def view_attendance_details(self):
        """View detailed attendance - FRONTEND COMPATIBLE"""
        try:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
            
            if os.path.exists(attendance_file):
                if os.name == 'nt':  # Windows
                    os.startfile(attendance_file)
                else:  # Linux/Mac
                    os.system(f"xdg-open {attendance_file}")
            else:
                mess.showinfo("Info", "No attendance data available for today")
        except Exception as e:
            logging.error(f"Error viewing attendance: {e}")
    
    def export_to_csv(self):
        """Export attendance to CSV with email option - FRONTEND COMPATIBLE"""
        try:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
            
            if not os.path.exists(attendance_file):
                mess.showwarning("Warning", "No attendance data available for today")
                return False
            
            # Ask user if they want to email the report
            email_option = mess.askyesno("Export Options", 
                                       "Do you want to email the attendance report?\n\n"
                                       "Yes - Send via email\n"
                                       "No - Save to local file only")
            
            if email_option:
                # Ask for email address
                email = simpledialog.askstring("Email Address", 
                                             "Enter the receiver's email address:",
                                             parent=None)
                
                if email and email.strip():
                    email = email.strip()
                    if self.validate_email(email):
                        # Send email
                        success, message = self.export_attendance_email(email, date)
                        if success:
                            mess.showinfo("Success", f"Attendance report sent to {email}")
                        else:
                            mess.showerror("Error", f"Failed to send email: {message}")
                            # Fallback to local save
                            return self.save_to_local_file(attendance_file, date)
                    else:
                        mess.showerror("Error", "Invalid email address format")
                        return False
                else:
                    mess.showinfo("Info", "Email address not provided. Operation cancelled.")
                    return False
            else:
                # Save to local file
                return self.save_to_local_file(attendance_file, date)
                
        except Exception as e:
            logging.error(f"Error exporting CSV: {e}")
            mess.showerror("Error", f"Export failed: {str(e)}")
            return False
    
    def save_to_local_file(self, attendance_file, date):
        """Save attendance file to local directory"""
        try:
            import shutil
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            if os.path.exists(desktop_path):
                export_path = os.path.join(desktop_path, f"Attendance_Export_{date}.csv")
            else:
                export_path = f"Attendance_Export_{date}.csv"
            
            shutil.copy2(attendance_file, export_path)
            mess.showinfo("Success", f"Attendance exported to: {export_path}")
            return True
        except Exception as e:
            mess.showerror("Error", f"Failed to save file: {str(e)}")
            return False
    
    def verify_password(self, password):
        """Verify admin password - FRONTEND COMPATIBLE"""
        stored_password = self.get_password()
        if stored_password is None:
            # No password set, create default
            self.save_new_password("admin123")
            return password == "admin123"
        return password == stored_password
    
    def change_password(self, old_password, new_password):
        """Change admin password - FRONTEND COMPATIBLE"""
        success, message = self.change_password_internal(old_password, new_password)
        return success
    
    def cleanup(self):
        """Cleanup resources - FRONTEND COMPATIBLE"""
        # Close any open camera connections
        cv2.destroyAllWindows()
    
    # REFRESH FUNCTIONALITY METHODS - FRONTEND COMPATIBLE
    def refresh_attendance_display(self):
        """Refresh attendance display - FRONTEND COMPATIBLE"""
        try:
            return self.get_attendance_records()
        except Exception as e:
            logging.error(f"Error refreshing attendance: {e}")
            return []
    
    def refresh_registration_count(self):
        """Refresh and get updated registration count - FRONTEND COMPATIBLE"""
        try:
            return self.get_total_registrations()
        except Exception as e:
            logging.error(f"Error refreshing registration count: {e}")
            return 0
    
    def refresh_student_list(self):
        """Refresh and get updated student list - FRONTEND COMPATIBLE"""
        try:
            df = self.load_student_details()
            if df is not None:
                return df.to_dict('records')
            return []
        except Exception as e:
            logging.error(f"Error refreshing student list: {e}")
            return []
    
    def refresh_camera_connection(self):
        """Refresh camera connection - FRONTEND COMPATIBLE"""
        return self.test_camera()
    
    def refresh_all_data(self):
        """Refresh all system data - FRONTEND COMPATIBLE"""
        try:
            results = {
                'attendance_records': self.get_attendance_records(),
                'registration_count': self.get_total_registrations(),
                'student_list': self.refresh_student_list(),
                'camera_status': self.test_camera()
            }
            return True, results
        except Exception as e:
            logging.error(f"Error refreshing all data: {e}")
            return False, None
    
    # INTERNAL METHODS (Original functionality)
    def is_valid_id(self, student_id):
        """Validate student ID format"""
        if not student_id or not isinstance(student_id, str):
            return False
        # More flexible ID validation - accepts alphanumeric IDs
        return bool(re.match(r'^[a-zA-Z0-9]{6,15}$', student_id.strip()))
    
    def is_valid_name(self, name):
        """Validate student name format"""
        if not name or not isinstance(name, str):
            return False
        return bool(re.match(r'^[A-Za-z\s]{2,50}$', name.strip())) and name.strip() != ""
    
    def validate_email(self, email):
        """Validate email format"""
        if not email or not isinstance(email, str):
            return False
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))
    
    def get_password(self):
        """Get stored password or return None if not exists"""
        password_file = os.path.join(self.training_label_path, "psd.txt")
        if os.path.isfile(password_file):
            try:
                with open(password_file, "r") as tf:
                    return tf.read().strip()
            except Exception as e:
                logging.error(f"Error reading password file: {e}")
                return None
        return None
    
    def save_new_password(self, password):
        """Save new password"""
        try:
            password_file = os.path.join(self.training_label_path, "psd.txt")
            with open(password_file, "w") as tf:
                tf.write(password)
        except Exception as e:
            logging.error(f"Error saving password: {e}")
    
    def change_password_internal(self, old_password, new_password):
        """Change password"""
        stored_password = self.get_password()
        if stored_password is None:
            return False, "No password found"
        
        if old_password != stored_password:
            return False, "Incorrect old password"
        
        self.save_new_password(new_password)
        return True, "Password changed successfully"
    
    def get_next_serial_number(self):
        """Get next serial number for student registration"""
        student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
        
        if os.path.isfile(student_file):
            try:
                df = pd.read_csv(student_file)
                return len(df) + 1
            except Exception as e:
                logging.error(f"Error reading student file: {e}")
                return 1
        else:
            # Create new file with headers
            try:
                columns = ['SERIAL NO.', 'ID', 'NAME']
                df = pd.DataFrame(columns=columns)
                df.to_csv(student_file, index=False)
                return 1
            except Exception as e:
                logging.error(f"Error creating student file: {e}")
                return 1
    
    def capture_images_internal(self, student_id, name):
        """Capture face images for training"""
        logging.info(f"Starting internal image capture for {name} ({student_id})")
        
        # Check haarcascade
        if not self.check_haarcascade_file():
            logging.error("Haarcascade file missing")
            return False, "Haarcascade file missing. Please check your installation."
        
        # Test camera first
        if not self.test_camera():
            logging.error("Camera test failed")
            return False, "Camera not accessible. Please check your camera connection."
        
        serial = self.get_next_serial_number()
        logging.info(f"Using serial number: {serial}")
        
        cam = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        if not cam.isOpened():
            logging.error("Camera not accessible")
            return False, "Camera not accessible. Please check your camera connection."
        
        try:
            detector = cv2.CascadeClassifier(self.haarcascade_path)
            if detector.empty():
                logging.error("Failed to load cascade classifier")
                cam.release()
                return False, "Failed to load face detector. Please check haarcascade file."
            
            sample_num = 0
            no_face_count = 0
            max_no_face = 50  # Maximum frames without face before showing warning
            
            logging.info(f"Capturing images for {name} ({student_id}). Press 'q' to quit early.")
            
            while sample_num < 100:
                ret, img = cam.read()
                if not ret:
                    logging.error("Failed to read frame from camera")
                    break
                
                # Flip image for better user experience
                img = cv2.flip(img, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with multiple scale factors for better detection
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    no_face_count = 0  # Reset counter when face is found
                    
                    for (x, y, w, h) in faces:
                        # Draw rectangle around face
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Only capture if face is of good size
                        if w > 50 and h > 50:
                            sample_num += 1
                            
                            # Save image with better naming convention
                            image_path = os.path.join(
                                self.training_image_path,
                                f"{name}.{serial}.{student_id}.{sample_num}.jpg"
                            )
                            
                            # Extract and save face region
                            face_region = gray[y:y + h, x:x + w]
                            face_resized = cv2.resize(face_region, (200, 200))
                            
                            success = cv2.imwrite(image_path, face_resized)
                            if not success:
                                logging.error(f"Failed to save image: {image_path}")
                            
                            # Show progress on image
                            progress_text = f"Capturing: {sample_num}/100"
                            cv2.putText(img, progress_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Show student info
                            info_text = f"Student: {name} ({student_id})"
                            cv2.putText(img, info_text, (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    no_face_count += 1
                    # Show warning if no face detected for too long
                    if no_face_count > max_no_face:
                        warning_text = "No face detected! Please position yourself properly"
                        cv2.putText(img, warning_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Show instructions
                instruction_text = "Press 'q' to quit | Keep your face in the green rectangle"
                cv2.putText(img, instruction_text, (10, img.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Capturing Face Images', img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or sample_num >= 100:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            if sample_num >= 50:  # Minimum 50 images for good training
                # Save student details
                self.save_student_details(serial, student_id, name)
                logging.info(f"Successfully captured {sample_num} images for {name}")
                return True, f"Successfully captured {sample_num} images for {name} ({student_id})"
            else:
                logging.warning(f"Only {sample_num} images captured, minimum 50 required")
                return False, f"Insufficient images captured ({sample_num}/50). Please try again with better lighting and positioning."
            
        except Exception as e:
            logging.error(f"Error during image capture: {e}")
            cam.release()
            cv2.destroyAllWindows()
            return False, f"Error during image capture: {str(e)}"
    
    def save_student_details(self, serial, student_id, name):
        """Save student details to CSV"""
        try:
            student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
            
            # Create or append to CSV
            new_row = {'SERIAL NO.': serial, 'ID': student_id, 'NAME': name}
            
            if os.path.exists(student_file):
                df = pd.read_csv(student_file)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            
            df.to_csv(student_file, index=False)
            logging.info(f"Student details saved: {student_id} - {name}")
        except Exception as e:
            logging.error(f"Error saving student details: {e}")
    
    def get_images_and_labels(self, path):
        """Get face images and labels for training"""
        try:
            image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
            faces = []
            ids = []
            
            for image_path in image_paths:
                try:
                    # Load image
                    pil_image = Image.open(image_path).convert('L')
                    image_np = np.array(pil_image, 'uint8')
                    
                    # Extract serial number from filename
                    filename = os.path.split(image_path)[-1]
                    parts = filename.split(".")
                    if len(parts) >= 2:
                        student_id = int(parts[1])  # Serial number
                        faces.append(image_np)
                        ids.append(student_id)
                    else:
                        logging.warning(f"Invalid filename format: {filename}")
                        
                except (ValueError, IndexError) as e:
                    logging.error(f"Error processing {image_path}: {e}")
                    continue
                except Exception as e:
                    logging.error(f"Unexpected error processing {image_path}: {e}")
                    continue
            
            logging.info(f"Loaded {len(faces)} training images")
            return faces, ids
        except Exception as e:
            logging.error(f"Error loading images and labels: {e}")
            return [], []
    
    def train_images(self):
        """Train the face recognition model"""
        try:
            if not self.check_haarcascade_file():
                return False, "Haarcascade file missing"
            
            logging.info("Loading training images...")
            faces, ids = self.get_images_and_labels(self.training_image_path)
            
            if len(faces) == 0:
                return False, "No training images found. Please capture images first."
            
            if len(set(ids)) < 1:
                return False, "Insufficient training data. Please capture more images."
            
            logging.info(f"Training with {len(faces)} images for {len(set(ids))} students...")
            
            # Create and train recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            
            # Save trained model
            model_path = os.path.join(self.training_label_path, "Trainer.yml")
            recognizer.save(model_path)
            
            logging.info("Training completed successfully")
            return True, f"Training completed successfully. Total images: {len(faces)}, Students: {len(set(ids))}"
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return False, f"Training failed: {str(e)}"
    
    def load_student_details(self):
        """Load student details from CSV"""
        student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
        if os.path.isfile(student_file):
            try:
                df = pd.read_csv(student_file)
                logging.info(f"Loaded {len(df)} student records")
                return df
            except Exception as e:
                logging.error(f"Error loading student details: {e}")
                return None
        return None
    
    def take_attendance_internal(self):
        """Take attendance using face recognition"""
        try:
            if not self.check_haarcascade_file():
                return False, "Haarcascade file missing", []
            
            # Load trained model
            model_path = os.path.join(self.training_label_path, "Trainer.yml")
            if not os.path.isfile(model_path):
                return False, "No trained model found. Please train the model first.", []
            
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(model_path)
            
            # Load student details
            df = self.load_student_details()
            if df is None:
                return False, "Student details missing", []
            
            face_cascade = cv2.CascadeClassifier(self.haarcascade_path)
            
            # Test camera
            if not self.test_camera():
                return False, "Camera not accessible", []
            
            cam = cv2.VideoCapture(0)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            attendance = []
            recognized_ids = set()
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            logging.info("Taking attendance... Press 'q' to quit")
            
            start_time = time.time()
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    try:
                        serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        confidence = round(100 - conf)
                        
                        if confidence > 60:  # Confidence threshold
                            # Get student details
                            student_data = df.loc[df['SERIAL NO.'] == serial]
                            if not student_data.empty:
                                name = student_data['NAME'].iloc[0]
                                student_id = student_data['ID'].iloc[0]
                                
                                if str(student_id) not in recognized_ids:
                                    ts = time.time()
                                    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                                    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%I:%M:%S %p')
                                    
                                    attendance_record = {
                                        'id': str(student_id),
                                        'name': str(name),
                                        'date': date,
                                        'time': timestamp
                                    }
                                    attendance.append(attendance_record)
                                    recognized_ids.add(str(student_id))
                                    
                                    logging.info(f"Recognized: {name} ({student_id})")
                                
                                cv2.putText(frame, f"{name} ({confidence}%)", (x, y-10), 
                                           font, 0.8, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, f"Unknown ({confidence}%)", (x, y-10), 
                                           font, 0.8, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "Unknown", (x, y-10), font, 0.8, (0, 0, 255), 2)
                    
                    except Exception as e:
                        logging.error(f"Recognition error: {e}")
                        cv2.putText(frame, "Error", (x, y-10), font, 0.8, (0, 0, 255), 2)
                
                # Show status
                cv2.putText(frame, f"Recognized: {len(recognized_ids)} students", (10, 30), 
                           font, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 60), font, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Taking Attendance', frame)
                
                # Auto-quit after 60 seconds or manual quit
                if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > 60:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            # Save attendance
            if attendance:
                self.save_attendance(attendance)
                logging.info(f"Attendance saved for {len(attendance)} students")
                return True, f"Attendance completed. {len(attendance)} students recognized.", attendance
            else:
                logging.info("No students recognized during attendance")
                return False, "No students recognized during attendance", []
                
        except Exception as e:
            logging.error(f"Error taking attendance: {e}")
            cam.release()
            cv2.destroyAllWindows()
            return False, f"Error taking attendance: {str(e)}", []
    
    def save_attendance(self, attendance_data):
        """Save attendance data to CSV file"""
        try:
            if not attendance_data:
                return False
            
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
            
            # Create DataFrame from attendance data
            df = pd.DataFrame(attendance_data)
            
            # If file exists, append; otherwise create new
            if os.path.exists(attendance_file):
                existing_df = pd.read_csv(attendance_file)
                # Avoid duplicates based on ID
                df = df[~df['id'].isin(existing_df['id'])]
                if not df.empty:
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(attendance_file, index=False)
            else:
                df.to_csv(attendance_file, index=False)
            
            logging.info(f"Attendance saved to {attendance_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving attendance: {e}")
            return False
    
    def get_today_attendance(self):
        """Get today's attendance records"""
        try:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
            
            if os.path.exists(attendance_file):
                df = pd.read_csv(attendance_file)
                return df.to_dict('records')
            return []
            
        except Exception as e:
            logging.error(f"Error getting today's attendance: {e}")
            return []
    
    def get_total_registrations(self):
        """Get total number of registered students"""
        try:
            df = self.load_student_details()
            if df is not None:
                return len(df)
            return 0
        except Exception as e:
            logging.error(f"Error getting total registrations: {e}")
            return 0
    
    def export_attendance_email(self, email, date):
        """Export attendance via email"""
        try:
            attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
            
            if not os.path.exists(attendance_file):
                return False, "Attendance file not found"
            
            # Email configuration - you'll need to configure these with your email settings
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_email@gmail.com"  # Configure with your email
            sender_password = "your_app_password"  # Configure with your app password
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = email
            msg['Subject'] = f"Attendance Report - {date}"
            
            # Email body
            body = f"""
            Dear User,
            
            Please find attached the attendance report for {date}.
            
            Generated by Face Recognition Attendance System.
            
            Best regards,
            Attendance System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach file
            with open(attendance_file, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= Attendance_{date}.csv',
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, email, text)
            server.quit()
            
            return True, "Email sent successfully"
            
        except Exception as e:
            logging.error(f"Error sending email: {e}")
            return False, str(e)
    
    def get_attendance_summary(self, date=None):
        """Get attendance summary for a specific date"""
        try:
            if date is None:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            
            attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
            
            if os.path.exists(attendance_file):
                df = pd.read_csv(attendance_file)
                total_students = self.get_total_registrations()
                present_students = len(df)
                absent_students = total_students - present_students
                
                return {
                    'date': date,
                    'total_students': total_students,
                    'present_students': present_students,
                    'absent_students': absent_students,
                    'attendance_percentage': (present_students / total_students * 100) if total_students > 0 else 0
                }
            else:
                return {
                    'date': date,
                    'total_students': self.get_total_registrations(),
                    'present_students': 0,
                    'absent_students': self.get_total_registrations(),
                    'attendance_percentage': 0
                }
                
        except Exception as e:
            logging.error(f"Error getting attendance summary: {e}")
            return None
    
    def delete_student(self, student_id):
        """Delete a student from the system"""
        try:
            # Remove from student details
            student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
            if os.path.exists(student_file):
                df = pd.read_csv(student_file)
                initial_count = len(df)
                df = df[df['ID'] != student_id]
                
                if len(df) < initial_count:
                    df.to_csv(student_file, index=False)
                    
                    # Remove training images
                    for filename in os.listdir(self.training_image_path):
                        if student_id in filename:
                            os.remove(os.path.join(self.training_image_path, filename))
                    
                    logging.info(f"Student {student_id} deleted successfully")
                    return True, "Student deleted successfully"
                else:
                    return False, "Student not found"
            else:
                return False, "Student database not found"
                
        except Exception as e:
            logging.error(f"Error deleting student: {e}")
            return False, str(e)
    
    def get_student_info(self, student_id):
        """Get information about a specific student"""
        try:
            df = self.load_student_details()
            if df is not None:
                student_data = df[df['ID'] == student_id]
                if not student_data.empty:
                    return student_data.iloc[0].to_dict()
            return None
        except Exception as e:
            logging.error(f"Error getting student info: {e}")
            return None
    
    def backup_data(self):
        """Create backup of all data"""
        try:
            import shutil
            import zipfile
            
            backup_dir = "backup"
            self.assure_path_exists(backup_dir)
            
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(backup_dir, f"attendance_backup_{timestamp}.zip")
            
            with zipfile.ZipFile(backup_file, 'w') as zipf:
                # Backup directories
                for root, dirs, files in os.walk(self.training_image_path):
                    for file in files:
                        zipf.write(os.path.join(root, file), 
                                 os.path.relpath(os.path.join(root, file), '.'))
                
                for root, dirs, files in os.walk(self.student_details_path):
                    for file in files:
                        zipf.write(os.path.join(root, file), 
                                 os.path.relpath(os.path.join(root, file), '.'))
                
                for root, dirs, files in os.walk(self.attendance_path):
                    for file in files:
                        zipf.write(os.path.join(root, file), 
                                 os.path.relpath(os.path.join(root, file), '.'))
                
                if os.path.exists(self.training_label_path):
                    for root, dirs, files in os.walk(self.training_label_path):
                        for file in files:
                            zipf.write(os.path.join(root, file), 
                                     os.path.relpath(os.path.join(root, file), '.'))
            
            logging.info(f"Backup created: {backup_file}")
            return True, f"Backup created successfully: {backup_file}"
            
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            return False, str(e)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the backend
    backend = AttendanceBackend()
    
    # Test basic functionality
    print("Testing camera connection...")
    if backend.test_camera():
        print("Camera test passed!")
    else:
        print("Camera test failed!")
    
    print(f"Total registrations: {backend.get_total_registrations()}")
    
    # Example of how to use the backend
    # backend.capture_images("STU001", "John Doe")
    # backend.train_model()
    # backend.take_attendance()                self.save
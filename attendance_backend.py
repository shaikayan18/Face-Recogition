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
        
    def assure_path_exists(self, path):
        """Ensure directory exists, create if not"""
        if not os.path.exists(path):
            os.makedirs(path)
    
    def check_haarcascade_file(self):
        """Check if haarcascade file exists"""
        return os.path.isfile(self.haarcascade_path)
    
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
            except:
                return False
        return False
    
    def capture_images(self, student_id, student_name):
        """Capture face images for training - FRONTEND COMPATIBLE"""
        try:
            success, message = self.capture_images_internal(student_id, student_name)
            return success
        except Exception as e:
            print(f"Error in capture_images: {e}")
            return False
    
    def train_model(self):
        """Train the face recognition model - FRONTEND COMPATIBLE"""
        try:
            success, message = self.train_images()
            return success
        except Exception as e:
            print(f"Error in train_model: {e}")
            return False
    
    def take_attendance(self):
        """Take attendance using face recognition - FRONTEND COMPATIBLE"""
        try:
            success, message, attendance_data = self.take_attendance_internal()
            return success
        except Exception as e:
            print(f"Error in take_attendance: {e}")
            return False
    
    def get_attendance_records(self):
        """Get attendance records for display - FRONTEND COMPATIBLE"""
        try:
            records = []
            # Get today's attendance
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
            print(f"Error getting attendance records: {e}")
            return []
    
    def view_attendance_details(self):
        """View detailed attendance - FRONTEND COMPATIBLE"""
        try:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
            
            if os.path.exists(attendance_file):
                os.startfile(attendance_file)  # Windows
                # For Linux/Mac: os.system(f"xdg-open {attendance_file}")
            else:
                mess.showinfo("Info", "No attendance data available for today")
        except Exception as e:
            print(f"Error viewing attendance: {e}")
    
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
            print(f"Error exporting CSV: {e}")
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
            # Clear any cached data and reload fresh attendance records
            return self.get_attendance_records()
        except Exception as e:
            print(f"Error refreshing attendance: {e}")
            return []
    
    def refresh_registration_count(self):
        """Refresh and get updated registration count - FRONTEND COMPATIBLE"""
        try:
            return self.get_total_registrations()
        except Exception as e:
            print(f"Error refreshing registration count: {e}")
            return 0
    
    def refresh_student_list(self):
        """Refresh and get updated student list - FRONTEND COMPATIBLE"""
        try:
            df = self.load_student_details()
            if df is not None:
                return df.to_dict('records')
            return []
        except Exception as e:
            print(f"Error refreshing student list: {e}")
            return []
    
    def refresh_camera_connection(self):
        """Refresh camera connection - FRONTEND COMPATIBLE"""
        try:
            # Test camera connection
            cam = cv2.VideoCapture(0)
            if cam.isOpened():
                ret, frame = cam.read()
                cam.release()
                cv2.destroyAllWindows()
                return ret  # True if camera works
            return False
        except Exception as e:
            print(f"Error checking camera: {e}")
            return False
    
    def refresh_all_data(self):
        """Refresh all system data - FRONTEND COMPATIBLE"""
        try:
            results = {
                'attendance_records': self.get_attendance_records(),
                'registration_count': self.get_total_registrations(),
                'student_list': self.refresh_student_list(),
                'camera_status': self.refresh_camera_connection()
            }
            return True, results
        except Exception as e:
            print(f"Error refreshing all data: {e}")
            return False, None
    
    # INTERNAL METHODS (Original functionality)
    def is_valid_id(self, student_id):
        """Validate student ID format"""
        return bool(re.fullmatch(r'1si\d{2}mc\d{3}', student_id.strip().lower()))
    
    def is_valid_name(self, name):
        """Validate student name format"""
        return bool(re.fullmatch(r'[A-Za-z ]+', name.strip())) and name.strip() != ""
    
    def validate_email(self, email):
        """Validate email format"""
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))
    
    def get_password(self):
        """Get stored password or return None if not exists"""
        password_file = os.path.join(self.training_label_path, "psd.txt")
        if os.path.isfile(password_file):
            with open(password_file, "r") as tf:
                return tf.read().strip()
        return None
    
    def save_new_password(self, password):
        """Save new password"""
        password_file = os.path.join(self.training_label_path, "psd.txt")
        with open(password_file, "w") as tf:
            tf.write(password)
    
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
            except:
                return 1
        else:
            # Create new file with headers
            columns = ['SERIAL NO.', 'ID', 'NAME']
            df = pd.DataFrame(columns=columns)
            df.to_csv(student_file, index=False)
            return 1
    
    def capture_images_internal(self, student_id, name):
        """Capture face images for training"""
        if not self.check_haarcascade_file():
            return False, "Haarcascade file missing. Please download haarcascade_frontalface_default.xml"
        
        if not self.is_valid_id(student_id):
            return False, "Invalid ID format"
        
        if not self.is_valid_name(name):
            return False, "Invalid name format"
        
        serial = self.get_next_serial_number()
        
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return False, "Camera not accessible"
        
        detector = cv2.CascadeClassifier(self.haarcascade_path)
        sample_num = 0
        
        try:
            print(f"Capturing images for {name} ({student_id}). Press 'q' to quit early.")
            
            while sample_num < 60:  # Reduced from 100 to 60 for faster capture
                ret, img = cam.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    sample_num += 1
                    
                    # Save image
                    image_path = os.path.join(
                        self.training_image_path,
                        f"{name}.{serial}.{student_id}.{sample_num}.jpg"
                    )
                    cv2.imwrite(image_path, gray[y:y + h, x:x + w])
                    
                    # Show progress
                    cv2.putText(img, f"Capturing: {sample_num}/60", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Capturing Images', img)
                
                if cv2.waitKey(100) & 0xFF == ord('q') or sample_num >= 60:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            if sample_num > 0:
                # Save student details
                self.save_student_details(serial, student_id, name)
                return True, f"Images captured for ID: {student_id}"
            else:
                return False, "No face detected. Please try again."
            
        except Exception as e:
            cam.release()
            cv2.destroyAllWindows()
            return False, f"Error capturing images: {str(e)}"
    
    def save_student_details(self, serial, student_id, name):
        """Save student details to CSV"""
        student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
        
        # Create or append to CSV
        new_row = {'SERIAL NO.': serial, 'ID': student_id, 'NAME': name}
        
        if os.path.exists(student_file):
            df = pd.read_csv(student_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])
        
        df.to_csv(student_file, index=False)
    
    def get_images_and_labels(self, path):
        """Get face images and labels for training"""
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces = []
        ids = []
        
        for image_path in image_paths:
            try:
                pil_image = Image.open(image_path).convert('L')
                image_np = np.array(pil_image, 'uint8')
                student_id = int(os.path.split(image_path)[-1].split(".")[1])
                faces.append(image_np)
                ids.append(student_id)
            except (ValueError, IndexError) as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return faces, ids
    
    def train_images(self):
        """Train the face recognition model"""
        if not self.check_haarcascade_file():
            return False, "Haarcascade file missing"
        
        try:
            print("Loading training images...")
            faces, ids = self.get_images_and_labels(self.training_image_path)
            
            if len(faces) == 0:
                return False, "No training images found. Please capture images first."
            
            print(f"Training with {len(faces)} images...")
            
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            
            print("Saving trained model...")
            model_path = os.path.join(self.training_label_path, "Trainer.yml")
            recognizer.save(model_path)
            
            return True, f"Training completed. Total images: {len(faces)}"
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def load_student_details(self):
        """Load student details from CSV"""
        student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
        if os.path.isfile(student_file):
            try:
                return pd.read_csv(student_file)
            except Exception as e:
                print(f"Error loading student details: {e}")
                return None
        return None
    
    def take_attendance_internal(self):
        """Take attendance using face recognition"""
        if not self.check_haarcascade_file():
            return False, "Haarcascade file missing", []
        
        # Load trained model
        model_path = os.path.join(self.training_label_path, "Trainer.yml")
        if not os.path.isfile(model_path):
            return False, "No trained model found. Please train first.", []
        
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(model_path)
        except Exception as e:
            return False, f"Error loading model: {e}", []
        
        # Load student details
        df = self.load_student_details()
        if df is None:
            return False, "Student details missing", []
        
        face_cascade = cv2.CascadeClassifier(self.haarcascade_path)
        cam = cv2.VideoCapture(0)
        
        if not cam.isOpened():
            return False, "Camera not accessible", []
        
        attendance = []
        recognized_ids = set()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        print("Taking attendance... Press 'q' to quit")
        
        try:
            start_time = time.time()
            while True:
                ret, frame = cam.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    try:
                        serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
                        
                        if conf < 60:  # Increased threshold for better accuracy
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
                                    
                                    print(f"Recognized: {name} ({student_id})")
                                
                                cv2.putText(frame, f"{name} ({conf:.0f}%)", (x, y-10), font, 0.8, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, f"Unknown ({conf:.0f}%)", (x, y-10), font, 0.8, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "Unknown", (x, y-10), font, 0.8, (0, 0, 255), 2)
                    
                    except Exception as e:
                        print(f"Recognition error: {e}")
                        cv2.putText(frame, "Error", (x, y-10), font, 0.8, (0, 0, 255), 2)
                
                # Show status
                cv2.putText(frame, f"Recognized: {len(recognized_ids)} students", (10, 30), 
                           font, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 60), font, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Taking Attendance', frame)
                
                # Auto-quit after 30 seconds or manual quit
                if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time) > 30:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            # Save attendance
            if attendance:
                self.save_attendance(attendance)
                return True, f"{len(attendance)} students marked present", attendance
            else:
                return True, "No students recognized", []
                
        except Exception as e:
            cam.release()
            cv2.destroyAllWindows()
            return False, f"Error taking attendance: {str(e)}", []
    
    def save_attendance(self, attendance_data):
        """Save attendance data to CSV"""
        if not attendance_data:
            return
        
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
        
        # Create or append to CSV
        df_new = pd.DataFrame(attendance_data)
        
        if os.path.exists(attendance_file):
            df_existing = pd.read_csv(attendance_file)
            # Remove duplicates based on ID
            df_existing = df_existing[~df_existing['id'].isin(df_new['id'])]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        df_combined.to_csv(attendance_file, index=False)
        print(f"Attendance saved to: {attendance_file}")
    
    def get_today_attendance(self):
        """Get today's attendance data"""
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
        
        if os.path.isfile(attendance_file):
            try:
                df = pd.read_csv(attendance_file)
                return df.to_dict('records')
            except Exception as e:
                print(f"Error reading attendance file: {e}")
                return []
        
        return []
    
    def get_total_registrations(self):
        """Get total number of registrations"""
        student_file = os.path.join(self.student_details_path, "StudentDetails.csv")
        if os.path.isfile(student_file):
            try:
                df = pd.read_csv(student_file)
                return len(df)
            except:
                return 0
        return 0
    
    def export_attendance_email(self, email, date=None):
        """Export attendance via email"""
        if not self.validate_email(email):
            return False, "Invalid email format"
        
        if date is None:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        
        attendance_file = os.path.join(self.attendance_path, f"Attendance_{date}.csv")
        
        if not os.path.exists(attendance_file):
            return False, "No attendance data available for the specified date"
        
        try:
            # Email configuration - UPDATE THESE WITH YOUR ACTUAL EMAIL CREDENTIALS
            sender_email = "heshaikayanro@gmail.com"  # CHANGE THIS TO YOUR EMAIL
            sender_password = "ioqh kkdh jyjr pcel"   # CHANGE THIS TO YOUR APP PASSWORD
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = email
            msg['Subject'] = f"Attendance Report for {date}"
            
            body = f"Dear Recipient,\n\nPlease find attached the attendance report for {date}.\n\nThis report contains the attendance records for all students who were present on the specified date.\n\nBest regards,\nAttendance Management System"
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach file
            with open(attendance_file, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f"attachment; filename= Attendance_{date}.csv")
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email, msg.as_string())
            server.quit()
            
            return True, f"Attendance report sent successfully to {email}"
            
        except smtplib.SMTPAuthenticationError:
            return False, "Email authentication failed. Please check your email credentials."
        except smtplib.SMTPException as e:
            return False, f"SMTP error occurred: {str(e)}"
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
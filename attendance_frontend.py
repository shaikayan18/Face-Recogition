import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
from tkinter import PhotoImage
from PIL import Image, ImageTk
import tkinter.simpledialog as tsd
import cv2
import datetime
import time
import threading

from attendance_backend import AttendanceBackend

class AttendanceFrontend:
    def __init__(self):
        self.backend = AttendanceBackend()
        self.setup_window()
        self.create_widgets()
        self.setup_validation()
        self.load_initial_data()
        
    def setup_window(self):
        """Setup main window"""
        self.window = tk.Tk()
        self.window.geometry(f"{self.window.winfo_screenwidth()}x{self.window.winfo_screenheight()}")
        self.window.title("Smart Attendance System")
        self.window.resizable(True, False)
        
        # Set theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Try to load background image
        try:
            bg_image = Image.open("background_image1.png")
            bg_photo = ImageTk.PhotoImage(bg_image)
            background_label = tk.Label(self.window, image=bg_photo)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)
            background_label.image = bg_photo  # Keep a reference
        except:
            self.window.configure(background='#f0f0f0')
    
    def configure_styles(self):
        """Configure theme styles"""
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 11))
        self.style.configure('TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('TEntry', font=('Helvetica', 11))
        
        # Configure Treeview
        self.style.configure("Treeview", 
                            background="#ffffff",
                            foreground="#333333",
                            rowheight=25,
                            fieldbackground="#ffffff",
                            font=('Helvetica', 10))
                            
        self.style.configure("Treeview.Heading", 
                            font=('Helvetica', 11, 'bold'),
                            background="#4CAF50",
                            foreground="white")
        
        self.style.map('Treeview', background=[('selected', '#007BFF')])
    
    def create_widgets(self):
        """Create all GUI widgets"""
        self.create_header()
        self.create_footer()
        self.create_main_content()
        self.create_clock()
        
    def create_header(self):
        """Create header section"""
        header_frame = tk.Frame(self.window, bg="#4CAF50", height=60)
        header_frame.place(x=0, y=0, relwidth=1)
        
        app_name = tk.Label(header_frame, text="Face Recognition Based Attendance System", 
                           fg="white", bg="#4CAF50", font=('Helvetica', 24, 'bold'))
        app_name.pack(pady=10)
    
    def create_footer(self):
        """Create footer section"""
        footer_frame = tk.Frame(self.window, bg="#333333", height=30)
        footer_frame.place(relx=0, rely=0.97, relwidth=1)
        
        footer_text = tk.Label(footer_frame, text="© 2025 Smart Attendance System", 
                              fg="white", bg="#333333", font=('Helvetica', 9))
        footer_text.pack(pady=5)
    
    def create_main_content(self):
        """Create main content area"""
        main_frame = tk.Frame(self.window, bg="#f0f0f0")
        main_frame.pack(pady=70, padx=20, fill="both", expand=True)
        
        # Left panel
        self.left_panel = tk.Frame(main_frame, bg='#f0f0f0', padx=20, pady=20, width=500)
        self.left_panel.pack(side=tk.LEFT, fill="y", expand=False)
        self.left_panel.pack_propagate(False)
        
        # Right panel
        self.right_panel = tk.Frame(main_frame, bg='#f0f0f0', padx=20, pady=20)
        self.right_panel.pack(side=tk.RIGHT, fill="both", expand=True)
        
        self.create_left_panel()
        self.create_right_panel()
    
    def create_clock(self):
        """Create digital clock"""
        clock_frame = tk.Frame(self.window, bg="#f0f0f0", bd=5, relief=tk.FLAT)
        clock_frame.place(relx=0.9, rely=0.05, anchor="ne")
        
        self.clock = tk.Label(clock_frame, bg="#f0f0f0", fg="#333333", font=('Helvetica', 16, 'bold'))
        self.clock.pack(pady=5, padx=10)
        self.tick()
    
    def tick(self):
        """Update clock"""
        current_time = time.strftime('%I:%M:%S %p')
        self.clock.config(text=current_time)
        self.clock.after(1000, self.tick)
    
    def create_left_panel(self):
        """Create left panel widgets"""
        # Registration Form
        self.create_registration_form()
        
        # Action buttons
        self.create_action_buttons()
        
        # Status message
        self.message1 = tk.Label(self.left_panel, text="1) Take Images  →  2) Save Profile", 
                                bg="#f0f0f0", fg="#666666", width=50, font=('Helvetica', 10, 'italic'))
        self.message1.pack(pady=10)
        
        # Information section
        self.create_info_section()
        
        # Admin tools
        self.create_admin_tools()
    
    def create_registration_form(self):
        """Create registration form"""
        form_frame = tk.LabelFrame(self.left_panel, text="Registration", bg="#f0f0f0", fg="#333333", 
                                  font=('Helvetica', 12, 'bold'), padx=15, pady=15)
        form_frame.pack(fill="x", pady=10)
        
        # Student ID
        lbl = tk.Label(form_frame, text="Enter ID:", bg='#f0f0f0', fg="#333333", font=('Helvetica', 11))
        lbl.grid(row=0, column=0, padx=5, pady=10, sticky="w")
        
        self.txt = tk.Entry(form_frame, width=20, fg="black", font=('Helvetica', 11), relief='solid')
        self.txt.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        self.txt.focus()
        
        # Student Name
        lbl2 = tk.Label(form_frame, text="Enter Name:", bg='#f0f0f0', fg="#333333", font=('Helvetica', 11))
        lbl2.grid(row=1, column=0, padx=5, pady=10, sticky="w")
        
        self.txt2 = tk.Entry(form_frame, width=20, fg="black", font=('Helvetica', 11), relief='solid')
        self.txt2.grid(row=1, column=1, padx=5, pady=10, sticky="w")
        
        # Format notes
        id_note = tk.Label(form_frame, text="ID Format: 1SIYYMCXXX (e.g., 1SI23MC027)", 
                          bg='#f0f0f0', fg="#666666", font=('Helvetica', 8))
        id_note.grid(row=0, column=2, padx=5, pady=10, sticky="w")
        
        name_note = tk.Label(form_frame, text="Name Format: Letters only", 
                            bg='#f0f0f0', fg="#666666", font=('Helvetica', 8))
        name_note.grid(row=1, column=2, padx=5, pady=10, sticky="w")
        
        # Clear buttons
        clear1 = tk.Button(form_frame, text="Clear", command=self.clear_id, fg="white", bg="#f44336", 
                          height=1, width=8, font=('Helvetica', 9))
        clear1.grid(row=0, column=3, padx=5, pady=10)
        
        clear2 = tk.Button(form_frame, text="Clear", command=self.clear_name, fg="white", bg="#f44336", 
                          height=1, width=8, font=('Helvetica', 9))
        clear2.grid(row=1, column=3, padx=5, pady=10)
    
    def create_action_buttons(self):
        """Create action buttons"""
        btn_frame = tk.Frame(self.left_panel, bg="#f0f0f0", pady=10)
        btn_frame.pack(fill="x")
        
        # Take Image button
        self.takeImg = tk.Button(btn_frame, text="Take Images", command=self.take_images, 
                                fg="white", bg="#4CAF50", height=2, width=15, font=('Helvetica', 12, 'bold'))
        self.takeImg.grid(row=0, column=0, padx=10, pady=10)
        self.takeImg.config(state='disabled')
        
        # Save Profile button
        self.trainImg = tk.Button(btn_frame, text="Save Profile", command=self.save_profile, 
                                 fg="white", bg="#007BFF", height=2, width=15, font=('Helvetica', 12, 'bold'))
        self.trainImg.grid(row=0, column=1, padx=10, pady=10)
        self.trainImg.config(state='disabled')
        
        # Take Attendance button
        self.trackImg = tk.Button(btn_frame, text="Take Attendance", command=self.take_attendance, 
                                 fg="white", bg="#FF9800", height=2, width=31, font=('Helvetica', 12, 'bold'))
        self.trackImg.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
    
    def create_info_section(self):
        """Create information section"""
        info_frame = tk.LabelFrame(self.left_panel, text="Information", bg="#f0f0f0", fg="#333333", 
                                  font=('Helvetica', 12, 'bold'), padx=15, pady=15)
        info_frame.pack(fill="x", pady=10)
        
        self.message = tk.Label(info_frame, text="Total Registrations: 0", bg="#f0f0f0", 
                               fg="#333333", width=30, font=('Helvetica', 11))
        self.message.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Date
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        day, month, year = date.split("-")
        
        mont = {'01':'January', '02':'February', '03':'March', '04':'April',
                '05':'May', '06':'June', '07':'July', '08':'August',
                '09':'September', '10':'October', '11':'November', '12':'December'}
        
        date_label = tk.Label(info_frame, text=f"Date: {day}-{mont[month]}-{year}", 
                             bg="#f0f0f0", fg="#333333", width=30, font=('Helvetica', 11))
        date_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
    
    def create_admin_tools(self):
        """Create admin tools"""
        admin_frame = tk.Frame(self.left_panel, bg="#f0f0f0", pady=10)
        admin_frame.pack(fill="x", side="bottom")
        
        # Admin buttons
        quitWindow = tk.Button(admin_frame, text="Quit", command=self.window.destroy, 
                              fg="white", bg="#f44336", height=1, width=10, font=('Helvetica', 10, 'bold'))
        quitWindow.grid(row=0, column=0, padx=5, pady=10)
        
        # Continuing from where the code was cut off...

        change_pass_btn = tk.Button(admin_frame, text="Change Password", command=self.change_password, 
                                   fg="white", bg="#333333", height=1, width=15, font=('Helvetica', 10, 'bold'))
        change_pass_btn.grid(row=0, column=1, padx=5, pady=10)
        
        contact_btn = tk.Button(admin_frame, text="Contact Us", command=self.contact_us, 
                               fg="white", bg="#795548", height=1, width=10, font=('Helvetica', 10, 'bold'))
        contact_btn.grid(row=0, column=2, padx=5, pady=10)
    
    def create_right_panel(self):
        """Create right panel with attendance records"""
        # Title
        title_label = tk.Label(self.right_panel, text="Attendance Records", 
                              bg="#f0f0f0", fg="#333333", font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Control frame
        control_frame = tk.Frame(self.right_panel, bg="#f0f0f0")
        control_frame.pack(fill="x", pady=10)
        
        # Refresh button
        refresh_btn = tk.Button(control_frame, text="Refresh", command=self.refresh_records, 
                               fg="white", bg="#17a2b8", height=1, width=10, font=('Helvetica', 10, 'bold'))
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # View attendance button
        view_attendance_btn = tk.Button(control_frame, text="View Attendance", command=self.view_attendance, 
                                       fg="white", bg="#6f42c1", height=1, width=15, font=('Helvetica', 10, 'bold'))
        view_attendance_btn.pack(side=tk.LEFT, padx=5)
        
        # Export button
        export_btn = tk.Button(control_frame, text="Export CSV", command=self.export_csv, 
                              fg="white", bg="#28a745", height=1, width=10, font=('Helvetica', 10, 'bold'))
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # Search frame
        search_frame = tk.Frame(self.right_panel, bg="#f0f0f0")
        search_frame.pack(fill="x", pady=5)
        
        search_label = tk.Label(search_frame, text="Search:", bg="#f0f0f0", fg="#333333", font=('Helvetica', 11))
        search_label.pack(side=tk.LEFT, padx=5)
        
        self.search_entry = tk.Entry(search_frame, width=20, font=('Helvetica', 11))
        self.search_entry.pack(side=tk.LEFT, padx=5)
        self.search_entry.bind('<KeyRelease>', self.search_records)
        
        # Treeview for attendance records
        tree_frame = tk.Frame(self.right_panel, bg="#f0f0f0")
        tree_frame.pack(fill="both", expand=True, pady=10)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        # Treeview
        self.tree = ttk.Treeview(tree_frame, columns=("ID", "Name", "Date", "Time"), 
                                show="headings", yscrollcommand=v_scrollbar.set, 
                                xscrollcommand=h_scrollbar.set)
        
        # Configure scrollbars
        v_scrollbar.config(command=self.tree.yview)
        h_scrollbar.config(command=self.tree.xview)
        
        # Define headings
        self.tree.heading("ID", text="Student ID")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Date", text="Date")
        self.tree.heading("Time", text="Time")
        
        # Configure column widths
        self.tree.column("ID", width=120, anchor="center")
        self.tree.column("Name", width=200, anchor="w")
        self.tree.column("Date", width=100, anchor="center")
        self.tree.column("Time", width=100, anchor="center")
        
        # Pack treeview and scrollbars
        self.tree.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
    
    def setup_validation(self):
        """Setup input validation"""
        # Register validation functions
        vcmd_id = (self.window.register(self.validate_id), '%P')
        vcmd_name = (self.window.register(self.validate_name), '%P')
        
        # Apply validation
        self.txt.config(validate='key', validatecommand=vcmd_id)
        self.txt2.config(validate='key', validatecommand=vcmd_name)
        
        # Bind events for enabling/disabling buttons
        self.txt.bind('<KeyRelease>', self.check_inputs)
        self.txt2.bind('<KeyRelease>', self.check_inputs)
    
    def validate_id(self, value):
        """Validate student ID format"""
        if len(value) > 10:
            return False
        # Allow only alphanumeric characters
        return all(c.isalnum() for c in value)
    
    def validate_name(self, value):
        """Validate student name format"""
        if len(value) > 50:
            return False
        # Allow only letters and spaces
        return all(c.isalpha() or c.isspace() for c in value)
    
    def check_inputs(self, event=None):
        """Enable/disable buttons based on input validation"""
        id_valid = len(self.txt.get().strip()) >= 3
        name_valid = len(self.txt2.get().strip()) >= 2
        
        if id_valid and name_valid:
            self.takeImg.config(state='normal')
        else:
            self.takeImg.config(state='disabled')
    
    def load_initial_data(self):
        """Load initial data"""
        self.update_registration_count()
        self.refresh_records()
    
    def update_registration_count(self):
        """Update registration count"""
        try:
            count = self.backend.get_registration_count()
            self.message.config(text=f"Total Registrations: {count}")
        except Exception as e:
            print(f"Error updating registration count: {e}")
    
    def clear_id(self):
        """Clear ID field"""
        self.txt.delete(0, 'end')
        self.takeImg.config(state='disabled')
    
    def clear_name(self):
        """Clear name field"""
        self.txt2.delete(0, 'end')
        self.takeImg.config(state='disabled')
    
    def take_images(self):
        """Take images for face recognition"""
        student_id = self.txt.get().strip()
        student_name = self.txt2.get().strip()
        
        if not student_id or not student_name:
            mess.showerror('Error', 'Please fill in both ID and Name fields')
            return
        
        # Validate ID format
        if not self.backend.validate_student_id(student_id):
            mess.showerror('Error', 'Invalid ID format. Use format: 1SIYYMCXXX')
            return
        
        # Check if student already exists
        if self.backend.student_exists(student_id):
            mess.showerror('Error', f'Student with ID {student_id} already exists')
            return
        
        try:
            self.message1.config(text="Taking Images... Please look at the camera")
            self.window.update()
            
            # Disable button during image capture
            self.takeImg.config(state='disabled')
            
            # Start image capture in separate thread
            thread = threading.Thread(target=self.capture_images_thread, 
                                     args=(student_id, student_name))
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            mess.showerror('Error', f'Failed to start image capture: {str(e)}')
            self.takeImg.config(state='normal')
    
    def capture_images_thread(self, student_id, student_name):
        """Capture images in separate thread"""
        try:
            success = self.backend.capture_images(student_id, student_name)
            
            # Update UI in main thread
            self.window.after(0, self.on_capture_complete, success, student_id, student_name)
            
        except Exception as e:
            self.window.after(0, self.on_capture_error, str(e))
    
    def on_capture_complete(self, success, student_id, student_name):
        """Handle image capture completion"""
        if success:
            self.message1.config(text="Images captured successfully! Now click 'Save Profile'")
            self.trainImg.config(state='normal')
            mess.showinfo('Success', f'Images captured for {student_name} (ID: {student_id})')
        else:
            self.message1.config(text="Image capture failed. Please try again.")
            mess.showerror('Error', 'Failed to capture images')
        
        self.takeImg.config(state='normal')
    
    def on_capture_error(self, error_msg):
        """Handle image capture error"""
        self.message1.config(text="Error during image capture")
        mess.showerror('Error', f'Image capture failed: {error_msg}')
        self.takeImg.config(state='normal')
    
    def save_profile(self):
        """Save/train the face recognition model"""
        try:
            self.message1.config(text="Training model... Please wait")
            self.window.update()
            
            self.trainImg.config(state='disabled')
            
            # Start training in separate thread
            thread = threading.Thread(target=self.train_model_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            mess.showerror('Error', f'Failed to start training: {str(e)}')
            self.trainImg.config(state='normal')
    
    def train_model_thread(self):
        """Train model in separate thread"""
        try:
            success = self.backend.train_model()
            self.window.after(0, self.on_training_complete, success)
            
        except Exception as e:
            self.window.after(0, self.on_training_error, str(e))
    
    def on_training_complete(self, success):
        """Handle training completion"""
        if success:
            self.message1.config(text="Profile saved successfully! Registration complete.")
            mess.showinfo('Success', 'Profile saved successfully!')
            
            # Clear form and update count
            self.clear_id()
            self.clear_name()
            self.update_registration_count()
            
        else:
            self.message1.config(text="Failed to save profile. Please try again.")
            mess.showerror('Error', 'Failed to save profile')
        
        self.trainImg.config(state='disabled')
    
    def on_training_error(self, error_msg):
        """Handle training error"""
        self.message1.config(text="Error during profile saving")
        mess.showerror('Error', f'Training failed: {error_msg}')
        self.trainImg.config(state='disabled')
    
    def take_attendance(self):
        """Take attendance using face recognition"""
        try:
            # Start attendance in separate thread
            thread = threading.Thread(target=self.attendance_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            mess.showerror('Error', f'Failed to start attendance: {str(e)}')
    
    def attendance_thread(self):
        """Take attendance in separate thread"""
        try:
            success = self.backend.take_attendance()
            self.window.after(0, self.on_attendance_complete, success)
            
        except Exception as e:
            self.window.after(0, self.on_attendance_error, str(e))
    
    def on_attendance_complete(self, success):
        """Handle attendance completion"""
        if success:
            mess.showinfo('Success', 'Attendance taken successfully!')
            self.refresh_records()
        else:
            mess.showwarning('Warning', 'No faces recognized or attendance already marked')
    
    def on_attendance_error(self, error_msg):
        """Handle attendance error"""
        mess.showerror('Error', f'Attendance failed: {error_msg}')
    
    def refresh_records(self):
        """Refresh attendance records"""
        try:
            # Clear existing records
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Get records from backend
            records = self.backend.refresh_attendance_display()
            
            # Insert records into treeview
            for record in records:
                self.tree.insert('', 'end', values=record)
                
        except Exception as e:
            mess.showerror('Error', f'Failed to refresh records: {str(e)}')
    
    def search_records(self, event=None):
        """Search attendance records"""
        search_term = self.search_entry.get().lower()
        
        # Clear current display
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            # Get all records
            records = self.backend.get_attendance_records()
            
            # Filter records based on search term
            filtered_records = []
            for record in records:
                if any(search_term in str(field).lower() for field in record):
                    filtered_records.append(record)
            
            # Insert filtered records
            for record in filtered_records:
                self.tree.insert('', 'end', values=record)
                
        except Exception as e:
            mess.showerror('Error', f'Search failed: {str(e)}')
    
    def view_attendance(self):
        """View detailed attendance"""
        try:
            self.backend.view_attendance_details()
        except Exception as e:
            mess.showerror('Error', f'Failed to view attendance: {str(e)}')
    
    def export_csv(self):
        """Export attendance to CSV"""
        try:
            success = self.backend.export_to_csv()
            if success:
                mess.showinfo('Success', 'Attendance exported to CSV successfully!')
            else:
                mess.showerror('Error', 'Failed to export attendance')
        except Exception as e:
            mess.showerror('Error', f'Export failed: {str(e)}')
    
    def change_password(self):
        """Change admin password"""
        try:
            # Get current password
            current_pwd = tsd.askstring('Password', 'Enter current password:', show='*')
            if not current_pwd:
                return
            
            # Verify current password
            if not self.backend.verify_password(current_pwd):
                mess.showerror('Error', 'Incorrect current password')
                return
            
            # Get new password
            new_pwd = tsd.askstring('Password', 'Enter new password:', show='*')
            if not new_pwd:
                return
            
            # Confirm new password
            confirm_pwd = tsd.askstring('Password', 'Confirm new password:', show='*')
            if not confirm_pwd:
                return
            
            if new_pwd != confirm_pwd:
                mess.showerror('Error', 'Passwords do not match')
                return
            
            # Change password
            if self.backend.change_password(current_pwd, new_pwd):
                mess.showinfo('Success', 'Password changed successfully!')
            else:
                mess.showerror('Error', 'Failed to change password')
                
        except Exception as e:
            mess.showerror('Error', f'Password change failed: {str(e)}')
    
    def contact_us(self):
        """Show contact information"""
        contact_info = """
        Smart Attendance System
        
        For technical support:
        Email: support@smartattendance.com
        Phone: +1-234-567-8900
        
        For general inquiries:
        Email: info@smartattendance.com
        
        Version: 1.0
        """
        mess.showinfo('Contact Us', contact_info)
    
    def run(self):
        """Start the application"""
        try:
            self.window.mainloop()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            # Cleanup
            if hasattr(self.backend, 'cleanup'):
                self.backend.cleanup()

# Main execution
if __name__ == "__main__":
    try:
        app = AttendanceFrontend()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()
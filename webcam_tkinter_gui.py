#!/usr/bin/env python3
"""
YOLO Character Recognition GUI using Tkinter
Desktop application for webcam character recognition
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from PIL import Image, ImageTk
import threading

class WebcamRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 YOLO Character Recognition")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.cap = None
        self.model = None
        self.is_running = False
        self.current_frame = None
        self.roi_size = 224
        self.output_dir = "webcam_predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
        
        # Start video update loop
        self.update_video()
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            model_path = "runs/classify/train11/weights/best.pt"
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            messagebox.showerror("Model Error", f"Could not load model: {e}")
            return False
            
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="🎯 YOLO Character Recognition", 
                              font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#2196F3')
        title_label.pack()
        
        # Video frame
        self.video_frame = tk.Frame(self.root, bg='#ddd', width=640, height=480)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, text="📹 Camera Feed", 
                                   font=("Arial", 14), bg='#ddd', fg='#666')
        self.video_label.pack(expand=True)
        
        # Control buttons
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(control_frame, text="📹 Start Camera", 
                                  command=self.start_camera, font=("Arial", 12),
                                  bg='#4CAF50', fg='white', padx=20, pady=5)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="🛑 Stop Camera", 
                                 command=self.stop_camera, font=("Arial", 12),
                                 bg='#f44336', fg='white', padx=20, pady=5,
                                 state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.predict_btn = tk.Button(control_frame, text="🎯 Predict Character", 
                                    command=self.predict_character, font=("Arial", 12),
                                    bg='#2196F3', fg='white', padx=20, pady=5,
                                    state='disabled')
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to start", 
                                    font=("Arial", 10), bg='#f0f0f0', fg='#666')
        self.status_label.pack(pady=5)
        
        # Results frame
        results_frame = tk.LabelFrame(self.root, text="📊 Prediction Results", 
                                     font=("Arial", 12, "bold"), bg='#f0f0f0')
        results_frame.pack(pady=10, padx=20, fill='x')
        
        self.prediction_label = tk.Label(results_frame, text="No predictions yet", 
                                        font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2196F3')
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = tk.Label(results_frame, text="", 
                                        font=("Arial", 12), bg='#f0f0f0', fg='#4CAF50')
        self.confidence_label.pack()
        
        # Top 3 predictions frame
        self.top3_frame = tk.Frame(results_frame, bg='#f0f0f0')
        self.top3_frame.pack(pady=5)
        
        # Instructions
        instructions_frame = tk.LabelFrame(self.root, text="📝 Instructions", 
                                          font=("Arial", 10, "bold"), bg='#f0f0f0')
        instructions_frame.pack(pady=10, padx=20, fill='x')
        
        instructions = [
            "1. Click 'Start Camera' to begin",
            "2. Place a character or digit in the green box",
            "3. Click 'Predict Character' to analyze",
            "4. Results are saved in the webcam_predictions folder"
        ]
        
        for instruction in instructions:
            inst_label = tk.Label(instructions_frame, text=instruction, 
                                 font=("Arial", 9), bg='#f0f0f0', anchor='w')
            inst_label.pack(fill='x', padx=10, pady=1)
            
    def start_camera(self):
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            
            # Update button states
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.predict_btn.config(state='normal')
            
            self.status_label.config(text="📹 Camera started", fg='#4CAF50')
            print("📹 Camera started")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Error starting camera: {e}")
            
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
        # Update button states
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.predict_btn.config(state='disabled')
        
        # Clear video display
        self.video_label.config(image='', text="📹 Camera Feed")
        
        self.status_label.config(text="🛑 Camera stopped", fg='#f44336')
        print("🛑 Camera stopped")
        
    def update_video(self):
        """Update video feed continuously"""
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Keep original orientation (no flip)
                self.current_frame = frame.copy()
                
                # Draw ROI rectangle
                height, width = frame.shape[:2]
                roi_x = (width - self.roi_size) // 2
                roi_y = (height - self.roi_size) // 2
                
                cv2.rectangle(frame, (roi_x, roi_y), 
                             (roi_x + self.roi_size, roi_y + self.roi_size), 
                             (0, 255, 0), 2)
                
                # Add instruction text
                cv2.putText(frame, "Place character in green box", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Convert frame for Tkinter display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil)
                
                # Update video label
                self.video_label.config(image=frame_tk, text='')
                self.video_label.image = frame_tk  # Keep a reference
        
        # Schedule next update
        self.root.after(33, self.update_video)  # ~30 FPS
        
    def predict_character(self):
        """Make a prediction on the current frame"""
        if not self.current_frame is None and self.model:
            try:
                self.status_label.config(text="🎯 Making prediction...", fg='#2196F3')
                self.root.update()
                
                height, width = self.current_frame.shape[:2]
                roi_x = (width - self.roi_size) // 2
                roi_y = (height - self.roi_size) // 2
                
                roi = self.current_frame[roi_y:roi_y + self.roi_size, 
                                        roi_x:roi_x + self.roi_size]
                
                # Convert to RGB for YOLO
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                
                # Run prediction
                results = self.model(roi_rgb, verbose=False)
                
                if results and len(results) > 0:
                    probs = results[0].probs
                    if probs is not None:
                        top_class_id = probs.top1
                        confidence = probs.top1conf.item() * 100
                        class_name = self.model.names[top_class_id]
                        
                        # Update display
                        self.prediction_label.config(text=f"🎯 Prediction: {class_name}")
                        self.confidence_label.config(text=f"📊 Confidence: {confidence:.1f}%")
                        
                        # Clear previous top3 display
                        for widget in self.top3_frame.winfo_children():
                            widget.destroy()
                        
                        # Show top 3 predictions
                        top3_indices = probs.top5[:3]
                        top3_label = tk.Label(self.top3_frame, text="🏆 Top 3 Predictions:", 
                                             font=("Arial", 10, "bold"), bg='#f0f0f0')
                        top3_label.pack()
                        
                        for i, idx in enumerate(top3_indices):
                            pred_class_name = self.model.names[idx]
                            conf = probs.data[idx].item() * 100
                            pred_text = f"{i+1}. {pred_class_name}: {conf:.1f}%"
                            pred_label = tk.Label(self.top3_frame, text=pred_text, 
                                                 font=("Arial", 9), bg='#f0f0f0')
                            pred_label.pack()
                        
                        # Save prediction
                        timestamp = int(time.time())
                        frame_with_roi = self.current_frame.copy()
                        cv2.rectangle(frame_with_roi, (roi_x, roi_y), 
                                    (roi_x + self.roi_size, roi_y + self.roi_size), 
                                    (0, 255, 0), 3)
                        cv2.putText(frame_with_roi, f"Prediction: {class_name} ({confidence:.1f}%)", 
                                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        
                        filename = f"{self.output_dir}/prediction_{timestamp}_{class_name}.jpg"
                        cv2.imwrite(filename, frame_with_roi)
                        
                        self.status_label.config(text=f"✅ Predicted: {class_name} ({confidence:.1f}%) - Saved!", fg='#4CAF50')
                        
                        print(f"\n🎯 Prediction: {class_name}")
                        print(f"📊 Confidence: {confidence:.2f}%")
                        print(f"💾 Saved: {filename}")
                        
                    else:
                        self.status_label.config(text="❌ No predictions found", fg='#f44336')
                else:
                    self.status_label.config(text="❌ Prediction failed", fg='#f44336')
                    
            except Exception as e:
                error_msg = f"Error during prediction: {e}"
                self.status_label.config(text="❌ Prediction error", fg='#f44336')
                messagebox.showerror("Prediction Error", error_msg)
                print(f"❌ {error_msg}")
        else:
            self.status_label.config(text="❌ No frame available", fg='#f44336')
            
    def on_closing(self):
        """Handle application closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    # Create main window
    root = tk.Tk()
    
    # Create application
    app = WebcamRecognitionApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("🚀 Starting YOLO Character Recognition Desktop App...")
    print("🛑 Close the window to stop the application")
    
    # Start the GUI
    root.mainloop()

if __name__ == '__main__':
    main()

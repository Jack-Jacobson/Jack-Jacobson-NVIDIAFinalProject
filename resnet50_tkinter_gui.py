#!/usr/bin/env python3
"""
ResNet50 Character Recognition GUI using Tkinter
Desktop application for webcam character recognition with ResNet50
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import time
import os

class ResNet50Predictor:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.class_names = checkpoint['class_names']
        self.num_classes = len(self.class_names)
        
        # Initialize model
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ ResNet50 model loaded from {model_path}")
        print(f"🎯 Classes: {self.class_names}")
        
    def predict_image(self, image):
        """Predict single image"""
        # Convert image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top predictions
            top5_prob, top5_indices = torch.topk(probabilities, min(5, len(self.class_names)))
            
            results = []
            for i in range(len(top5_indices)):
                class_idx = top5_indices[i].item()
                confidence = top5_prob[i].item()
                class_name = self.class_names[class_idx]
                results.append({
                    'class': class_name,
                    'confidence': confidence * 100,
                    'class_idx': class_idx
                })
        
        return results

class ResNet50RecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 ResNet50 Character Recognition")
        self.root.geometry("800x750")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.cap = None
        self.predictor = None
        self.is_running = False
        self.current_frame = None
        self.roi_size = 224
        self.output_dir = "resnet50_predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
        
        # Start video update loop
        self.update_video()
        
    def load_model(self):
        """Load the ResNet50 model"""
        try:
            model_path = "resnet50_best.pth"
            if not os.path.exists(model_path):
                # Try alternative paths
                alt_paths = ["resnet50_final.pth", "runs/classify/train11/weights/best.pt"]
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        if alt_path.endswith('.pt'):
                            messagebox.showinfo("Model Info", 
                                               "ResNet50 model not found. Using YOLO model as fallback.")
                            return False
                        model_path = alt_path
                        break
                else:
                    messagebox.showerror("Model Error", 
                                       "ResNet50 model not found. Please train the model first by running 'python train_resnet50.py'")
                    return False
            
            self.predictor = ResNet50Predictor(model_path)
            return True
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load ResNet50 model: {e}")
            return False
            
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(title_frame, text="🤖 ResNet50 Character Recognition", 
                              font=("Arial", 18, "bold"), bg='#f0f0f0', fg='#2196F3')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Fine-tuned ResNet50 for Character Classification", 
                                 font=("Arial", 10), bg='#f0f0f0', fg='#666')
        subtitle_label.pack()
        
        # Model info
        info_frame = tk.Frame(self.root, bg='#e8f4fd', relief='raised', bd=1)
        info_frame.pack(pady=5, padx=20, fill='x')
        
        if self.predictor:
            device_info = f"Device: {self.predictor.device}"
            classes_info = f"Classes: {len(self.predictor.class_names)}"
        else:
            device_info = "Device: Not loaded"
            classes_info = "Classes: Not available"
            
        tk.Label(info_frame, text=f"🔧 {device_info} | 📊 {classes_info}", 
                font=("Arial", 9), bg='#e8f4fd', fg='#333').pack(pady=3)
        
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
        
        self.predict_btn = tk.Button(control_frame, text="🤖 Predict Character", 
                                    command=self.predict_character, font=("Arial", 12),
                                    bg='#2196F3', fg='white', padx=20, pady=5,
                                    state='disabled' if not self.predictor else 'disabled')
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready to start", 
                                    font=("Arial", 10), bg='#f0f0f0', fg='#666')
        self.status_label.pack(pady=5)
        
        # Results frame
        results_frame = tk.LabelFrame(self.root, text="🤖 ResNet50 Prediction Results", 
                                     font=("Arial", 12, "bold"), bg='#f0f0f0')
        results_frame.pack(pady=10, padx=20, fill='x')
        
        self.prediction_label = tk.Label(results_frame, text="No predictions yet", 
                                        font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2196F3')
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = tk.Label(results_frame, text="", 
                                        font=("Arial", 12), bg='#f0f0f0', fg='#4CAF50')
        self.confidence_label.pack()
        
        self.inference_time_label = tk.Label(results_frame, text="", 
                                           font=("Arial", 10), bg='#f0f0f0', fg='#666')
        self.inference_time_label.pack()
        
        # Top 5 predictions frame
        self.top5_frame = tk.Frame(results_frame, bg='#f0f0f0')
        self.top5_frame.pack(pady=5)
        
        # Instructions
        instructions_frame = tk.LabelFrame(self.root, text="📝 Instructions", 
                                          font=("Arial", 10, "bold"), bg='#f0f0f0')
        instructions_frame.pack(pady=10, padx=20, fill='x')
        
        instructions = [
            "1. Click 'Start Camera' to begin",
            "2. Place a character or digit in the green box",
            "3. Click 'Predict Character' to analyze with ResNet50",
            "4. Results are saved in the resnet50_predictions folder"
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
            if self.predictor:
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
        """Make a prediction on the current frame using ResNet50"""
        if not self.current_frame is None and self.predictor:
            try:
                self.status_label.config(text="🤖 Making ResNet50 prediction...", fg='#2196F3')
                self.root.update()
                
                height, width = self.current_frame.shape[:2]
                roi_x = (width - self.roi_size) // 2
                roi_y = (height - self.roi_size) // 2
                
                roi = self.current_frame[roi_y:roi_y + self.roi_size, 
                                        roi_x:roi_x + self.roi_size]
                
                # Make prediction with timing
                start_time = time.time()
                results = self.predictor.predict_image(roi)
                inference_time = time.time() - start_time
                
                if results:
                    best_pred = results[0]
                    
                    # Update display
                    self.prediction_label.config(text=f"🤖 Prediction: {best_pred['class']}")
                    self.confidence_label.config(text=f"📊 Confidence: {best_pred['confidence']:.2f}%")
                    self.inference_time_label.config(text=f"⏱️ Inference time: {inference_time:.3f}s")
                    
                    # Clear previous top5 display
                    for widget in self.top5_frame.winfo_children():
                        widget.destroy()
                    
                    # Show top 5 predictions
                    top5_label = tk.Label(self.top5_frame, text="🏆 Top 5 Predictions:", 
                                         font=("Arial", 10, "bold"), bg='#f0f0f0')
                    top5_label.pack()
                    
                    for i, result in enumerate(results):
                        pred_text = f"{i+1}. {result['class']}: {result['confidence']:.2f}%"
                        pred_label = tk.Label(self.top5_frame, text=pred_text, 
                                             font=("Arial", 9), bg='#f0f0f0')
                        pred_label.pack()
                    
                    # Save prediction
                    timestamp = int(time.time())
                    frame_with_roi = self.current_frame.copy()
                    cv2.rectangle(frame_with_roi, (roi_x, roi_y), 
                                (roi_x + self.roi_size, roi_y + self.roi_size), 
                                (0, 255, 0), 3)
                    cv2.putText(frame_with_roi, f"ResNet50: {best_pred['class']} ({best_pred['confidence']:.1f}%)", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame_with_roi, f"Time: {inference_time:.3f}s", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    filename = f"{self.output_dir}/resnet50_prediction_{timestamp}_{best_pred['class']}.jpg"
                    cv2.imwrite(filename, frame_with_roi)
                    
                    self.status_label.config(text=f"✅ Predicted: {best_pred['class']} ({best_pred['confidence']:.1f}%) - Saved!", fg='#4CAF50')
                    
                    print(f"\n🤖 ResNet50 Prediction: {best_pred['class']}")
                    print(f"📊 Confidence: {best_pred['confidence']:.2f}%")
                    print(f"⏱️  Inference time: {inference_time:.3f}s")
                    print(f"💾 Saved: {filename}")
                        
                else:
                    self.status_label.config(text="❌ No predictions found", fg='#f44336')
                    
            except Exception as e:
                error_msg = f"Error during ResNet50 prediction: {e}"
                self.status_label.config(text="❌ Prediction error", fg='#f44336')
                messagebox.showerror("Prediction Error", error_msg)
                print(f"❌ {error_msg}")
        else:
            if not self.predictor:
                self.status_label.config(text="❌ ResNet50 model not loaded", fg='#f44336')
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
    app = ResNet50RecognitionApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("🚀 Starting ResNet50 Character Recognition Desktop App...")
    print("🛑 Close the window to stop the application")
    
    # Start the GUI
    root.mainloop()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
PyTorch YOLO Character Recognition Web GUI
Flask web interface for webcam character recognition using PyTorch model
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
import torch

app = Flask(__name__)

class WebcamPredictor:
    def __init__(self):
        self.cap = None
        self.model = None
        self.class_names = None
        self.is_running = False
        self.current_frame = None
        self.latest_prediction = {"class": "None", "confidence": 0.0, "timestamp": ""}
        self.roi_size = 224
        self.output_dir = "pytorch_predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_model()
        
    def load_model(self):
        """Load the PyTorch YOLO model"""
        try:
            model_path = "/home/{USERNAME}/FinalProject/classify/train3/weights/best.pt"
            
            if not os.path.exists(model_path):
                print(f"❌ PyTorch model not found at {model_path}")
                return False
            
            # Load PyTorch YOLO model
            self.model = YOLO(model_path)
            
            # Load class names from lettersdatabase2
            labels_file = "/home/{USERNAME}/FinalProject/lettersdatabase2/labels.txt"
            if os.path.exists(labels_file):
                with open(labels_file, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
                print(f"📂 Loaded {len(self.class_names)} classes from labels.txt: {self.class_names}")
            else:
                # Fallback to A-Z if labels file not found
                self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                  'U', 'V', 'W', 'X', 'Y', 'Z']
                print(f"⚠️ Using fallback class names (A-Z): {len(self.class_names)} classes")
            
            # Verify model and classes match
            model_classes = len(self.model.names)
            if model_classes != len(self.class_names):
                print(f"⚠️ Model has {model_classes} classes but dataset has {len(self.class_names)} classes")
            
            print(f"✅ PyTorch YOLO model loaded successfully from {model_path}")
            print(f"� Model classes: {model_classes}")
            print(f"� Dataset classes: {len(self.class_names)}")
            print(f"�️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
            return False
            
    def start_camera(self):
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(0)  # Using camera index 1 as specified
            if not self.cap.isOpened():
                print("❌ Failed to open camera at index 1")
                return False
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            print("📹 Camera started successfully")
            return True
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
            
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("🛑 Camera stopped")
        
    def get_frame(self):
        """Get current frame with ROI overlay"""
        if not self.is_running or not self.cap:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
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
        
        # Add latest prediction
        if self.latest_prediction["class"] != "None":
            pred_text = f"Last: {self.latest_prediction['class']} ({self.latest_prediction['confidence']:.1f}%)"
            cv2.putText(frame, pred_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
        
    def predict_character(self):
        """Make a prediction on the current frame using PyTorch YOLO model"""
        if not self.current_frame is None and self.model:
            try:
                height, width = self.current_frame.shape[:2]
                roi_x = (width - self.roi_size) // 2
                roi_y = (height - self.roi_size) // 2
                
                roi = self.current_frame[roi_y:roi_y + self.roi_size, 
                                        roi_x:roi_x + self.roi_size]
                
                # Save original ROI for debugging
                timestamp = int(time.time())
                cv2.imwrite(f"{self.output_dir}/debug_roi_{timestamp}.jpg", roi)
                
                # Run YOLO prediction
                start_time = time.time()
                results = self.model(roi, verbose=False)
                inference_time = time.time() - start_time
                
                # Process results
                if len(results) > 0 and results[0].probs is not None:
                    probs = results[0].probs
                    
                    # Get top predictions
                    top_indices = probs.top5  # Top 5 predictions
                    confidences = probs.top5conf.cpu().numpy()  # Convert to numpy
                    
                    if len(top_indices) > 0:
                        top_class_id = top_indices[0]  # Already an int, no need for .item()
                        confidence = confidences[0] * 100
                        
                        # Use our class names from lettersdatabase2
                        if top_class_id < len(self.class_names):
                            class_name = self.class_names[top_class_id]
                        else:
                            class_name = f"Class_{top_class_id}"
                        
                        # Update latest prediction
                        self.latest_prediction = {
                            "class": class_name,
                            "confidence": confidence,
                            "timestamp": time.strftime("%H:%M:%S")
                        }
                        
                        # Get top 3 predictions
                        top3_predictions = []
                        for i in range(min(3, len(top_indices))):
                            pred_class_id = top_indices[i]  # Already an int
                            pred_conf = confidences[i] * 100
                            pred_class_name = self.class_names[pred_class_id] if pred_class_id < len(self.class_names) else f"Class_{pred_class_id}"
                            top3_predictions.append({"class": pred_class_name, "confidence": pred_conf})
                        
                        # Save prediction with debug info
                        frame_with_roi = self.current_frame.copy()
                        cv2.rectangle(frame_with_roi, (roi_x, roi_y), 
                                    (roi_x + self.roi_size, roi_y + self.roi_size), 
                                    (0, 255, 0), 3)
                        cv2.putText(frame_with_roi, f"PyTorch: {class_name} ({confidence:.1f}%)", 
                                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(frame_with_roi, f"Time: {inference_time:.3f}s", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        filename = f"{self.output_dir}/pytorch_prediction_{timestamp}_{class_name}.jpg"
                        cv2.imwrite(filename, frame_with_roi)
                        
                        return {
                            "success": True,
                            "prediction": class_name,
                            "confidence": confidence,
                            "inference_time": inference_time,
                            "top3": top3_predictions,
                            "filename": filename
                        }
                        
                return {"success": False, "error": "No predictions found"}
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Detailed error: {error_details}")
                return {"success": False, "error": str(e), "details": error_details}
                
        return {"success": False, "error": "No frame available or PyTorch model not loaded"}

# Initialize the predictor
predictor = WebcamPredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    """Start the camera"""
    success = predictor.start_camera()
    return jsonify({"success": success})

@app.route('/stop_camera')
def stop_camera():
    """Stop the camera"""
    predictor.stop_camera()
    return jsonify({"success": True})

@app.route('/predict')
def predict():
    """Make a prediction"""
    result = predictor.predict_character()
    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = predictor.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Create templates directory and HTML file
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>PyTorch YOLO Character Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .video-container { text-align: center; margin: 20px 0; }
        .controls { text-align: center; margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
        .start-btn { background-color: #4CAF50; color: white; }
        .stop-btn { background-color: #f44336; color: white; }
        .predict-btn { background-color: #2196F3; color: white; }
        .results { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .prediction { font-size: 18px; font-weight: bold; color: #2196F3; }
        .confidence { color: #4CAF50; }
        .top3 { margin-top: 10px; }
        .top3-item { margin: 5px 0; }
        #video { border: 2px solid #ddd; border-radius: 10px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.success { background-color: #d4edda; color: #155724; }
        .status.error { background-color: #f8d7da; color: #721c24; }
        .pytorch-badge { background: linear-gradient(45deg, #ee4c2c, #ff6b35); color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 PyTorch YOLO Character Recognition <span class="pytorch-badge">PyTorch</span></h1>
        <p><strong>Model:</strong> /home/nvidia10/FinalProject/classify/train3/weights/best.pt</p>
        <p><strong>Dataset:</strong> lettersdatabase2 (26 letter classes A-Z)</p>
        
        <div class="video-container">
            <img id="video" src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>
        
        <div class="controls">
            <button class="start-btn" onclick="startCamera()">📹 Start Camera</button>
            <button class="stop-btn" onclick="stopCamera()">🛑 Stop Camera</button>
            <button class="predict-btn" onclick="predict()">🎯 Predict Character</button>
        </div>
        
        <div id="status" class="status"></div>
        
        <div class="results">
            <h3>📊 Latest Prediction Results</h3>
            <div id="prediction-result">No predictions yet. Start camera and click predict!</div>
        </div>
        
        <div class="results">
            <h3>📝 Instructions</h3>
            <ul>
                <li>Click "Start Camera" to begin</li>
                <li>Place a letter (A-Z) in the green box</li>
                <li>Click "Predict Character" to analyze</li>
                <li>Results are saved in the pytorch_predictions folder</li>
                <li>This GUI uses the PyTorch model trained on lettersdatabase2</li>
            </ul>
        </div>
    </div>

    <script>
        function showStatus(message, isError = false) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + (isError ? 'error' : 'success');
        }

        function startCamera() {
            fetch('/start_camera')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showStatus('Camera started successfully!');
                    } else {
                        showStatus('Failed to start camera', true);
                    }
                });
        }

        function stopCamera() {
            fetch('/stop_camera')
                .then(response => response.json())
                .then(data => {
                    showStatus('Camera stopped');
                });
        }

        function predict() {
            showStatus('Making prediction...');
            fetch('/predict')
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('prediction-result');
                    
                    if (data.success) {
                        let html = `
                            <div class="prediction">🎯 Prediction: ${data.prediction}</div>
                            <div class="confidence">📊 Confidence: ${data.confidence.toFixed(1)}%</div>
                            <div class="top3">
                                <strong>🏆 Top 3 Predictions:</strong>
                        `;
                        
                        data.top3.forEach((pred, index) => {
                            html += `<div class="top3-item">${index + 1}. ${pred.class}: ${pred.confidence.toFixed(1)}%</div>`;
                        });
                        
                        html += `</div><div style="margin-top: 10px; color: #666;">💾 Saved: ${data.filename}</div>`;
                        resultDiv.innerHTML = html;
                        
                        showStatus(`Predicted: ${data.prediction} (${data.confidence.toFixed(1)}%)`);
                    } else {
                        resultDiv.innerHTML = `<div style="color: red;">❌ Error: ${data.error}</div>`;
                        showStatus('Prediction failed', true);
                    }
                });
        }
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    print("🚀 Starting PyTorch YOLO Character Recognition Web GUI...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("🔥 Using PyTorch model: /home/nvidia10/FinalProject/classify/train3/weights/best.pt")
    print("📂 Dataset: lettersdatabase2 (26 letter classes A-Z)")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)

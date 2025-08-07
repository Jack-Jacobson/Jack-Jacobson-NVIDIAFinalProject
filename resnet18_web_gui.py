#!/usr/bin/env python3
"""
ResNet18 Character Recognition Web GUI
Flask web interface for webcam character recognition using trained ResNet18
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import base64
import time
import os
import threading
from io import BytesIO
import onnxruntime as ort

app = Flask(__name__)

class ResNet18Predictor:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.session = None
        self.class_names = None
        self.num_classes = 42  # Default for character dataset
        
        # Default class names for character recognition
        self.default_classes = [
            "0","1","2","3","4","5","6","7","8","9",
            "A","B","C","D","E","F","G","H","I","J",
            "K","L","M","N","O","P","Q","R","S","T",
            "U","V","W","X","Y","Z",
            "Apostraphe","Comma","Period","Slash","Question Mark","Exclamation Mark"
        ]
        
        # Try to load ONNX model
        if self.load_onnx_model(model_path):
            print(f"✅ ONNX Model loaded successfully")
        else:
            print("❌ No ONNX model found. Using untrained PyTorch ResNet18.")
            self.load_default_model()
        
        # Define transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_onnx_model(self, model_path=None):
        """Try to load ONNX model from various possible locations"""
        possible_paths = []
        
        if model_path:
            possible_paths.append(model_path)
        
        # Add the specific path you provided
        possible_paths.extend([
            "/home/nvidia10/jetson-inference/python/training/classification/models/database/resnet18.onnx",
            "/home/nvidia10/jetson-inference/python/training/classification/models/resnet18.onnx",
            "resnet18.onnx",
            "model.onnx"
        ])
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Create ONNX Runtime session
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                    self.session = ort.InferenceSession(path, providers=providers)
                    
                    # Get input and output details
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_name = self.session.get_outputs()[0].name
                    
                    # Try to load class names from labels.txt in the same directory
                    labels_path = os.path.join(os.path.dirname(path), "labels.txt")
                    if os.path.exists(labels_path):
                        with open(labels_path, 'r') as f:
                            self.class_names = [line.strip() for line in f.readlines() if line.strip()]
                        print(f"✅ Loaded {len(self.class_names)} class labels from {labels_path}")
                    else:
                        # Use default class names as fallback
                        self.class_names = self.default_classes
                        print("⚠️ No labels.txt found, using default class names")
                    
                    self.num_classes = len(self.class_names)
                    
                    print(f"✅ Loaded ONNX model from: {path}")
                    print(f"📊 Input name: {self.input_name}")
                    print(f"📊 Output name: {self.output_name}")
                    print(f"📊 Classes: {len(self.class_names)}")
                    print(f"📊 Class names: {self.class_names}")
                    print(f"📊 Providers: {self.session.get_providers()}")
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed to load ONNX model {path}: {e}")
                    continue
        
        return False
    
    def load_default_model(self):
        """Load default untrained PyTorch ResNet18 as fallback"""
        self.class_names = self.default_classes
        self.num_classes = len(self.class_names)
        
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"⚠️ Using untrained PyTorch ResNet18 with {self.num_classes} classes")
        
    def predict_image(self, image):
        """Predict single image using ONNX or PyTorch model"""
        # Convert image if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Make prediction based on model type
        if self.session is not None:
            # ONNX inference
            input_array = input_tensor.numpy()
            outputs = self.session.run([self.output_name], {self.input_name: input_array})[0]
            outputs = torch.from_numpy(outputs)
        elif hasattr(self, 'model') and self.model is not None:
            # PyTorch inference
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
        else:
            return None
        
        # Apply softmax and get top predictions
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

class WebcamPredictor:
    def __init__(self):
        self.cap = None
        self.predictor = ResNet18Predictor()
        self.is_running = False
        self.current_frame = None
        self.latest_prediction = {"class": "None", "confidence": 0.0, "timestamp": ""}
        self.roi_size = 224
        self.output_dir = "resnet18_predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def start_camera(self):
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            print("📹 Camera started")
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
        """Make a prediction on the current frame"""
        # Check if we have either ONNX session or PyTorch model
        has_model = (self.predictor.session is not None) or (hasattr(self.predictor, 'model') and self.predictor.model is not None)
        
        if not self.current_frame is None and has_model:
            try:
                height, width = self.current_frame.shape[:2]
                roi_x = (width - self.roi_size) // 2
                roi_y = (height - self.roi_size) // 2
                
                roi = self.current_frame[roi_y:roi_y + self.roi_size, 
                                        roi_x:roi_x + self.roi_size]
                
                # Make prediction with timing
                start_time = time.time()
                results = self.predictor.predict_image(roi)
                inference_time = time.time() - start_time
                
                if results and len(results) > 0:
                    best_pred = results[0]
                    
                    # Update latest prediction
                    self.latest_prediction = {
                        "class": best_pred['class'],
                        "confidence": best_pred['confidence'],
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    
                    # Save prediction
                    timestamp = int(time.time())
                    frame_with_roi = self.current_frame.copy()
                    cv2.rectangle(frame_with_roi, (roi_x, roi_y), 
                                (roi_x + self.roi_size, roi_y + self.roi_size), 
                                (0, 255, 0), 3)
                    cv2.putText(frame_with_roi, f"ResNet18: {best_pred['class']} ({best_pred['confidence']:.1f}%)", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame_with_roi, f"Time: {inference_time:.3f}s", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    filename = f"{self.output_dir}/resnet18_prediction_{timestamp}_{best_pred['class']}.jpg"
                    cv2.imwrite(filename, frame_with_roi)
                    
                    return {
                        "success": True,
                        "prediction": best_pred['class'],
                        "confidence": best_pred['confidence'],
                        "inference_time": inference_time,
                        "top5": results,
                        "filename": filename
                    }
                        
                return {"success": False, "error": "No predictions found"}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
                
        return {"success": False, "error": "No frame available or model not loaded"}

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

@app.route('/model_info')
def model_info():
    """Get model information"""
    model_type = "ONNX" if predictor.predictor.session is not None else "PyTorch"
    model_loaded = predictor.predictor.session is not None or (hasattr(predictor.predictor, 'model') and predictor.predictor.model is not None)
    
    return jsonify({
        "model_loaded": model_loaded,
        "model_type": model_type,
        "num_classes": predictor.predictor.num_classes,
        "class_names": predictor.predictor.class_names,
        "device": predictor.predictor.device
    })

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
    <title>ResNet18 Character Recognition</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .video-container { text-align: center; margin: 20px 0; }
        .controls { text-align: center; margin: 20px 0; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; border: none; border-radius: 5px; cursor: pointer; }
        .start-btn { background-color: #4CAF50; color: white; }
        .stop-btn { background-color: #f44336; color: white; }
        .predict-btn { background-color: #2196F3; color: white; }
        .info-btn { background-color: #ff9800; color: white; }
        .results { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .prediction { font-size: 18px; font-weight: bold; color: #2196F3; }
        .confidence { color: #4CAF50; }
        .inference-time { color: #666; font-size: 14px; }
        .top5 { margin-top: 10px; }
        .top5-item { margin: 5px 0; }
        #video { border: 2px solid #ddd; border-radius: 10px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.success { background-color: #d4edda; color: #155724; }
        .status.error { background-color: #f8d7da; color: #721c24; }
        .model-info { background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 ResNet18 Character Recognition</h1>
        <p>Webcam-based character recognition using trained ResNet18 model</p>
        
        <div class="model-info" id="model-info">
            <strong>Loading model information...</strong>
        </div>
        
        <div class="video-container">
            <img id="video" src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>
        
        <div class="controls">
            <button class="start-btn" onclick="startCamera()">📹 Start Camera</button>
            <button class="stop-btn" onclick="stopCamera()">🛑 Stop Camera</button>
            <button class="predict-btn" onclick="predict()">🤖 Predict Character</button>
            <button class="info-btn" onclick="loadModelInfo()">ℹ️ Model Info</button>
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
                <li>Place a character or digit in the green box</li>
                <li>Click "Predict Character" to analyze with ResNet18</li>
                <li>Results are saved in the resnet18_predictions folder</li>
                <li>Click "Model Info" to see loaded model details</li>
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
            showStatus('Making ResNet18 prediction...');
            fetch('/predict')
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('prediction-result');
                    
                    if (data.success) {
                        let html = `
                            <div class="prediction">🤖 Prediction: ${data.prediction}</div>
                            <div class="confidence">📊 Confidence: ${data.confidence.toFixed(2)}%</div>
                            <div class="inference-time">⏱️ Inference time: ${data.inference_time.toFixed(3)}s</div>
                            <div class="top5">
                                <strong>🏆 Top 5 Predictions:</strong>
                        `;
                        
                        data.top5.forEach((pred, index) => {
                            html += `<div class="top5-item">${index + 1}. ${pred.class}: ${pred.confidence.toFixed(2)}%</div>`;
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

        function loadModelInfo() {
            fetch('/model_info')
                .then(response => response.json())
                .then(data => {
                    const infoDiv = document.getElementById('model-info');
                    infoDiv.innerHTML = `
                        <strong>Model Status:</strong> ${data.model_loaded ? '✅ Loaded' : '❌ Not Loaded'}<br>
                        <strong>Model Type:</strong> ${data.model_type || 'Unknown'}<br>
                        <strong>Classes:</strong> ${data.num_classes}<br>
                        <strong>Device:</strong> ${data.device}<br>
                        <strong>Architecture:</strong> ResNet18
                    `;
                });
        }

        // Load model info on page load
        window.onload = function() {
            loadModelInfo();
        };
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    print("🚀 Starting ResNet18 Character Recognition Web GUI...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

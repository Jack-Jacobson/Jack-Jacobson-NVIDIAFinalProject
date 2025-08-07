#!/usr/bin/env python3
"""
YOLO Character Recognition Web GUI
Flask web interface for webcam character recognition
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import onnxruntime as ort
import base64
import time
import os
import threading
from io import BytesIO

app = Flask(__name__)

class WebcamPredictor:
    def __init__(self):
        self.cap = None
        self.session = None
        self.class_names = None
        self.is_running = False
        self.current_frame = None
        self.latest_prediction = {"class": "None", "confidence": 0.0, "timestamp": ""}
        self.roi_size = 224
        self.output_dir = "webcam_predictions"
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_model()
        
    def load_model(self):
        """Load the ONNX model"""
        try:
            # Look for ONNX models in various locations
            possible_paths = [
                "/home/nvidia10/runs/classify/train2/weights/best.onnx"
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
                    
            if not model_path:
                print("❌ No ONNX model found. Please export your YOLO model to ONNX format first.")
                return False
            
            # Load ONNX model
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in str(ort.get_available_providers()).lower() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input and output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Load class names based on actual training data structure
            # Get classes from the training directory structure
            train_dir = "/home/nvidia10/datasets/lettersdatabase/train"
            if os.path.exists(train_dir):
                self.class_names = sorted([d for d in os.listdir(train_dir) 
                                         if os.path.isdir(os.path.join(train_dir, d))])
                print(f"📂 Loaded {len(self.class_names)} classes from training data: {self.class_names}")
            else:
                # Fallback to manual list if training dir not found
                self.class_names = [
                    "0","1","2","3","4","5","6","7","8","9",
                    "A","B","C","D","E","F","G","H","I","J",
                    "K","L","M","N","O","P","Q","R","S","T",
                    "U","V","W","X","Y","Z",
                    "Apostraphe","Comma","Period","Slash","Question Mark","Exclamation Mark"
                ]
                print(f"⚠️ Using fallback class names: {len(self.class_names)} classes")
            
            print(f"✅ ONNX model loaded successfully from {model_path}")
            print(f"📊 Input shape: {self.input_shape}")
            print(f"📊 Providers: {self.session.get_providers()}")
            print(f"📊 Classes: {len(self.class_names)}")
            return True
        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            return False
            
    def start_camera(self):
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(1)  # Changed from 0 to 1
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
        """Make a prediction on the current frame using ONNX model"""
        if not self.current_frame is None and self.session:
            try:
                height, width = self.current_frame.shape[:2]
                roi_x = (width - self.roi_size) // 2
                roi_y = (height - self.roi_size) // 2
                
                roi = self.current_frame[roi_y:roi_y + self.roi_size, 
                                        roi_x:roi_x + self.roi_size]
                
                # Save original ROI for debugging
                timestamp = int(time.time())
                cv2.imwrite(f"{self.output_dir}/debug_roi_original_{timestamp}.jpg", roi)
                
                # Preprocess the image for ONNX model
                # Apply some image enhancement to better match training data
                roi_enhanced = roi.copy()
                
                # Convert to grayscale and back to RGB for better contrast (optional)
                gray = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2GRAY)
                # Apply some contrast enhancement
                gray = cv2.equalizeHist(gray)
                # Convert back to BGR
                roi_enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # Resize to model input size (224x224)
                roi_resized = cv2.resize(roi_enhanced, (224, 224))
                
                # Convert BGR to RGB
                roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                
                # Save enhanced and preprocessed image for debugging
                cv2.imwrite(f"{self.output_dir}/debug_roi_enhanced_{timestamp}.jpg", 
                           cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
                
                # Try different normalization approaches
                # Option 1: Simple [0,1] normalization (common for YOLO models)
                roi_normalized_simple = roi_rgb.astype(np.float32) / 255.0
                
                # Option 2: ImageNet normalization (for models pretrained on ImageNet)
                roi_normalized_imagenet = roi_rgb.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                roi_normalized_imagenet = (roi_normalized_imagenet - mean) / std
                
                # Try both normalizations and see which works better
                predictions_results = []
                
                for norm_type, roi_normalized in [("simple", roi_normalized_simple), ("imagenet", roi_normalized_imagenet)]:
                    # Transpose to CHW format and add batch dimension
                    input_data = np.transpose(roi_normalized, (2, 0, 1)).astype(np.float32)
                    input_data = np.expand_dims(input_data, axis=0)
                    
                    # Run inference
                    start_time = time.time()
                    outputs = self.session.run([self.output_name], {self.input_name: input_data})
                    inference_time = time.time() - start_time
                    
                    # Process outputs
                    logits = outputs[0][0]  # Remove batch dimension
                    
                    # Apply softmax to get probabilities
                    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
                    probabilities = exp_logits / np.sum(exp_logits)
                    
                    # Get top predictions
                    top_indices = np.argsort(probabilities)[::-1][:3]  # Top 3
                    
                    if len(top_indices) > 0:
                        top_class_id = top_indices[0]
                        confidence = probabilities[top_class_id] * 100
                        class_name = self.class_names[top_class_id] if top_class_id < len(self.class_names) else f"Class_{top_class_id}"
                        
                        predictions_results.append({
                            "norm_type": norm_type,
                            "class": class_name,
                            "confidence": confidence,
                            "inference_time": inference_time,
                            "top_indices": top_indices,
                            "probabilities": probabilities
                        })
                
                # Choose the best prediction (highest confidence)
                if predictions_results:
                    best_prediction = max(predictions_results, key=lambda x: x["confidence"])
                    
                    # Update latest prediction
                    self.latest_prediction = {
                        "class": best_prediction["class"],
                        "confidence": best_prediction["confidence"],
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    
                    # Get top 3 predictions from best result
                    top3_predictions = []
                    for idx in best_prediction["top_indices"]:
                        pred_class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class_{idx}"
                        conf = best_prediction["probabilities"][idx] * 100
                        top3_predictions.append({"class": pred_class_name, "confidence": conf})
                    
                    # Save prediction with debug info
                    frame_with_roi = self.current_frame.copy()
                    cv2.rectangle(frame_with_roi, (roi_x, roi_y), 
                                (roi_x + self.roi_size, roi_y + self.roi_size), 
                                (0, 255, 0), 3)
                    cv2.putText(frame_with_roi, f"ONNX ({best_prediction['norm_type']}): {best_prediction['class']} ({best_prediction['confidence']:.1f}%)", 
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame_with_roi, f"Time: {best_prediction['inference_time']:.3f}s", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Add debug info about both predictions
                    y_offset = 100
                    for result in predictions_results:
                        cv2.putText(frame_with_roi, f"{result['norm_type']}: {result['class']} ({result['confidence']:.1f}%)", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                        y_offset += 25
                    
                    filename = f"{self.output_dir}/onnx_prediction_{timestamp}_{best_prediction['class']}.jpg"
                    cv2.imwrite(filename, frame_with_roi)
                    
                    return {
                        "success": True,
                        "prediction": best_prediction["class"],
                        "confidence": best_prediction["confidence"],
                        "inference_time": best_prediction["inference_time"],
                        "normalization": best_prediction["norm_type"],
                        "top3": top3_predictions,
                        "filename": filename,
                        "debug": {
                            "all_predictions": [{"norm": r["norm_type"], "class": r["class"], "conf": r["confidence"]} for r in predictions_results]
                        }
                    }
                        
                return {"success": False, "error": "No predictions found"}
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Detailed error: {error_details}")
                return {"success": False, "error": str(e), "details": error_details}
                
        return {"success": False, "error": "No frame available or ONNX model not loaded"}

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
    <title>YOLO Character Recognition</title>
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 ONNX Character Recognition</h1>
        
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
                <li>Place a character or digit in the green box</li>
                <li>Click "Predict Character" to analyze</li>
                <li>Results are saved in the webcam_predictions folder</li>
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
    
    print("🚀 Starting ONNX Character Recognition Web GUI...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

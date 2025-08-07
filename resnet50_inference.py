#!/usr/bin/env python3
"""
ResNet50 Character Recognition Inference
Load trained ResNet50 model and make predictions
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os
import time

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
    
    def predict_webcam_roi(self, frame, roi_size=224):
        """Extract ROI from webcam frame and predict"""
        height, width = frame.shape[:2]
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        # Extract ROI
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        
        # Make prediction
        results = self.predict_image(roi)
        
        return results, (roi_x, roi_y, roi_size)

def test_resnet50_predictor():
    """Test the ResNet50 predictor"""
    model_path = "resnet50_best.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found. Please train the model first.")
        return
    
    # Initialize predictor
    predictor = ResNet50Predictor(model_path)
    
    # Test with webcam
    print("\n📹 Testing with webcam...")
    print("Press 'p' to predict, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    roi_size = 224
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw ROI rectangle
        height, width = frame.shape[:2]
        roi_x = (width - roi_size) // 2
        roi_y = (height - roi_size) // 2
        
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x + roi_size, roi_y + roi_size), 
                     (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Place character in green box", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'p' to predict, 'q' to quit", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("ResNet50 Character Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("\n🎯 Making prediction...")
            
            # Make prediction
            start_time = time.time()
            results, roi_info = predictor.predict_webcam_roi(frame, roi_size)
            inference_time = time.time() - start_time
            
            # Display results
            print(f"⏱️  Inference time: {inference_time:.3f}s")
            print("📊 Top 5 predictions:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['class']}: {result['confidence']:.2f}%")
            
            # Save prediction
            timestamp = int(time.time())
            pred_frame = frame.copy()
            
            # Add prediction text to frame
            best_pred = results[0]
            cv2.putText(pred_frame, f"Prediction: {best_pred['class']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(pred_frame, f"Confidence: {best_pred['confidence']:.1f}%", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            filename = f"resnet50_prediction_{timestamp}_{best_pred['class']}.jpg"
            cv2.imwrite(filename, pred_frame)
            print(f"💾 Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_resnet50_predictor()

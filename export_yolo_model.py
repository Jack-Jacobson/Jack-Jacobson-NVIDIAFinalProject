#!/usr/bin/env python3
"""
YOLO Model Export Script
Export trained YOLO classification model to various formats including ONNX
"""

from ultralytics import YOLO
import os

def export_yolo_model():
    # Path to your best trained model
    model_path = "/home/nvidia10/FinalProject/runs/classify/train12/weights/best.pt"
    
    print(f"🔍 Loading YOLO model from: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Load the trained model
    model = YOLO(model_path)
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model info: {model.info()}")
    
    # Create export directory
    export_dir = "/home/nvidia10/FinalProject/exported_models"
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"\n🚀 Starting export process...")
    
    try:
        # Export to ONNX format
        print("📦 Exporting to ONNX format...")
        onnx_path = model.export(
            format='onnx',
            imgsz=224,  # Input image size for classification
            optimize=True,
            half=False,  # Use FP32 for better compatibility
            int8=False,
            dynamic=False,
            simplify=True,
            opset=11  # ONNX opset version
        )
        print(f"✅ ONNX export complete: {onnx_path}")
        
        # Move to export directory
        import shutil
        onnx_filename = f"yolo_classifier_best.onnx"
        final_onnx_path = os.path.join(export_dir, onnx_filename)
        shutil.move(onnx_path, final_onnx_path)
        print(f"📁 Moved to: {final_onnx_path}")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
    
    try:
        # Export to TensorRT format (for NVIDIA devices)
        print("\n📦 Exporting to TensorRT format...")
        engine_path = model.export(
            format='engine',
            imgsz=224,
            half=True,  # Use FP16 for TensorRT
            device=0 if os.system("nvidia-smi > /dev/null 2>&1") == 0 else 'cpu'
        )
        print(f"✅ TensorRT export complete: {engine_path}")
        
        # Move to export directory
        engine_filename = f"yolo_classifier_best.engine"
        final_engine_path = os.path.join(export_dir, engine_filename)
        shutil.move(engine_path, final_engine_path)
        print(f"📁 Moved to: {final_engine_path}")
        
    except Exception as e:
        print(f"❌ TensorRT export failed: {e}")
    
    try:
        # Export to TorchScript format
        print("\n📦 Exporting to TorchScript format...")
        torchscript_path = model.export(
            format='torchscript',
            imgsz=224,
            optimize=True
        )
        print(f"✅ TorchScript export complete: {torchscript_path}")
        
        # Move to export directory
        torchscript_filename = f"yolo_classifier_best.torchscript"
        final_torchscript_path = os.path.join(export_dir, torchscript_filename)
        shutil.move(torchscript_path, final_torchscript_path)
        print(f"📁 Moved to: {final_torchscript_path}")
        
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
    
    try:
        # Export to OpenVINO format (Intel optimization)
        print("\n📦 Exporting to OpenVINO format...")
        openvino_path = model.export(
            format='openvino',
            imgsz=224,
            half=False
        )
        print(f"✅ OpenVINO export complete: {openvino_path}")
        
    except Exception as e:
        print(f"❌ OpenVINO export failed: {e}")
    
    print(f"\n🎉 Export process completed!")
    print(f"📁 Exported models saved to: {export_dir}")
    
    # List all exported files
    print(f"\n📋 Exported files:")
    if os.path.exists(export_dir):
        for file in os.listdir(export_dir):
            file_path = os.path.join(export_dir, file)
            file_size = os.path.getsize(file_path) / (1024*1024)  # Size in MB
            print(f"   📄 {file} ({file_size:.2f} MB)")

if __name__ == "__main__":
    export_yolo_model()

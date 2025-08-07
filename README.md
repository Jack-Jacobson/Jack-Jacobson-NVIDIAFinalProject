
# Handwritten Letter Recognization

My project allows the user to draw a letter and it will recognize what letter is drawn in order to turn handwritten notes into virtual ones. My goal with this is to help people convert notes from handwritten to virtual which is much more usable and helpful. 

## Training, running, and testing the model

Make sure to replace {USERNAME} in all the paths below. This also assumes you have uploaded this folder as FinalProject to the /home/{USERNAME} on your Jetson computer. 

Install YOLO

```bash
  pip install ultralytics
```

Make sure you are in the correct directory
```bash
cd FinalProject
```

Start the training of the AI

```bash
yolo task=classify mode=train model=yolov8n-cls.pt data=lettersdatabase2
```

Test the Accuracy Based off of Validation Images

```bash
yolo task=classify mode=val model=/home/{USERNAME}/FinalProject/classify/train3/weights/best.pt  data=/home/{USERNAME}/FinalProject/lettersdatabase2
```

Test the Accuracy of New Images Using the Webcam Plugged into Jetson
```bash
yolo task=classify mode=predict model=/home/{USERNAME}/FinalProject/classify/train3/weights/best.pt source=0
```
NOTE: If you get an error, you may have to change source=1.  

Activate the web-based GUI to see Live Camera Output and Prediction

```bash
python3 webcam_pytorch_gui.py
```

This will output a link to the webgui.
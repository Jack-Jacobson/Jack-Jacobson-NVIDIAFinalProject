
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

Activate the web-based GUI to see Live Camera Output and Prediction In order for this to work, go to this file and go to line 33 AND 43 to include your username instead of {USERNAME}

```bash
python3 webcam_pytorch_gui.py
```

This will output a link to the webgui. Note that if you get error "Failed to start camera", go to line 73 and change the 0 to the 1, this should resolve it.

## Images

An example of the WebUI:
<img width="1468" height="1136" alt="image" src="https://github.com/user-attachments/assets/daf4e295-668f-4b9c-8ad2-6ba12d807f7d" />

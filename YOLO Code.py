#!/usr/bin/env python
# coding: utf-8

# # Image Object Detection

# In[6]:


get_ipython().run_line_magic('pip', 'install cvzone')


# In[7]:


from ultralytics import YOLO
import cv2
import cvzone


# In[8]:


model = YOLO("yolov8l.pt")
results = model("traffic.jpg", show=True)
cv2.waitKey(0)


# # Yolo With Webcam

# In[9]:


# Import all the important liabilities
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

"""if you want to capture video from webcam so we can use CV2.video() capture either we can use CV2.video(path the file) capture 
and give them path of the our video"""

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 720)
cap.set(4, 640)
# cap = cv2.VideoCapture("bikes.mp4")  # For Video
# Instantiate the model Yolo V8 large model
model = YOLO("yolov8l.pt")

# then input all the classes from Yolu V8 large mode
classNames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
              'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
              'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
              'teddy bear', 'hair drier', 'toothbrush']

prev_frame_time = 0
new_frame_time = 0

# Frame processing loop
while True:
    new_frame_time = time.time()
    success, img = cap.read()
    
    # Frame processing
    if not success:  # Break the loop if there's no frame
        break

    results = model(img, stream=True)
    # Object drawing and text display
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    # Frame rate calculation
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    
    
    # Display and key handling
    
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)  # Change the argument from 0 to 1
    if key == ord('q'):  # Exit the loop if 'q' is pressed
        break
# clean up
cap.release()
cv2.destroyAllWindows()


# In[ ]:





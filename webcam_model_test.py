import cv2
import torch
from ultralytics import YOLO
from PIL import Image as PILImage
from IPython.display import display, Image
import time
import numpy as np
import os
from filters import preprocess_img

# Load the YOLOv8 model for classification (replace 'path_to_model.pt' with your model path)
model = YOLO('training-results/train/weights/best.pt')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply a filter (grayscale)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # opencv to PIL
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = PILImage.fromarray(frame)

    myimg = preprocess_img(frame, 550, 550)

    # PIL to opencv
    numpy_image=np.array(myimg)
    myimg_cv2=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # run against model
    
    results = model(myimg_cv2)
    # print()
    # print(results)

    
    # Extract the predicted class and confidence
     # Extract the predicted class and confidence using Probs attributes
    predicted_class_index = results[0].probs.top1  # Get the index of the top class
    confidence = results[0].probs.top1conf.item()  # Get the confidence score
    label = model.names[predicted_class_index]  # Get the class name using the index# Display the label and confidence on the image
    
    if confidence > 0.9:
        cv2.putText(myimg_cv2, f'{label}: {confidence:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('YOLOv8 Classification', myimg_cv2)

    # Break the loop on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
os.remove("ignore_this.jpg")
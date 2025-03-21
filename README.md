# drowiness_detection

import cv2
import cvzone
import math
from ultralytics import YOLO
import pygame  # Importing pygame for sound playback

# Initialize pygame mixer for sound
pygame.mixer.init()

# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set the width
cap.set(4, 720)   # Set the height

# Load the model
model = YOLO("drowsinessDemo4.pt")

# Class names for the model's outputs
classNames = ['awake', 'close_eye', 'close_mouth', 'drowsy', 'glasses', 'mask', 'open_eye', 'open_mouth', 'sunglasses']

while True:
    success, img = cap.read()
    if not success:
        break  # Break if frame capture fails
    
    # Get the results from the YOLO model
    results = model(img, device="mps", stream=True)  # Adjust device accordingly
    
    # Iterate over results
    for r in results:
        boxes = r.boxes  # Corrected syntax here. boxes is now accessible.

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1
            
            # Draw the corner rectangle
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            # Get confidence and class index
            conf = box.conf[0].item() * 100  # Convert the tensor to a scalar and multiply by 100 for percentage
            conf = round(conf, 2)  # Round to two decimal places
            cls = int(box.cls[0])  # Get the class index
            
            # Add the class label and confidence to the image
            label = f'{classNames[cls]} {conf}%'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=2)
            
            # Check if the detected class is 'drowsy' and if confidence is above 50%
            if classNames[cls] == 'drowsy' and conf > 50:
                # Play a beep sound if the conditions are met
                try:
                    pygame.mixer.music.load("beep.mp3")  # Load the sound file
                    pygame.mixer.music.play()  # Play the sound
                except Exception as e:
                    print(f"Error playing sound: {e}")  # Print error if sound file not found or other issues
                    
    # Show the image with detection boxes
    cv2.imshow("Image", img)
    
    # Wait for 1 ms before capturing the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()

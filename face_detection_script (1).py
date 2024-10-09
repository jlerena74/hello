
import cv2
import numpy as np
import os
import datetime

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the USB camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a directory to save detected faces
try:
    os.makedirs('detected_faces', exist_ok=True)
except OSError as e:
    print(f"Error creating directory: {e}")
    exit()

face_id = 0

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale image to speed up detection (50% of the original size)
        small_frame = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        # Detect faces in the resized frame
        faces = face_cascade.detectMultiScale(small_frame, scaleFactor=1.1, minNeighbors=5)

        # If faces are detected, process them
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Adjust face coordinates to match the original frame size
                x *= 2
                y *= 2
                w *= 2
                h *= 2
                
                # Draw rectangles around detected faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Increment face ID and save the detected face image with a timestamp
                face_id += 1
                face_image = frame[y:y + h, x:x + w]
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f'detected_faces/face_{face_id}_{timestamp}.jpg', face_image)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Add controls: press 'q' to quit, 'p' to pause
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)  # Wait indefinitely until any key is pressed

finally:
    # Ensure resources are released properly
    cap.release()
    cv2.destroyAllWindows()

# Final Working Code

import time
import cv2
import csv
from alert_system import play_alarm, send_email
from emotion_recognition import analyze_emotion
from face_detection import detect_faces

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# CSV file to log emotions
log_file = '/home/harshad/Documents/Computer Vision/passenger_safety/emotion_log.csv'

# Open the CSV file once before the loop to append emotions
with open(log_file, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write the header if the file is empty (optional)
    if file.tell() == 0:
        writer.writerow(['Timestamp', 'Emotion'])

    # Emotion count threshold for email notifications
    emotion_count = {'fear': 0, 'sad': 0}
    threshold = 10

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect faces
        faces = detect_faces(frame)

        # Analyze emotions for each detected face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # Perform emotion analysis
            dominant_emotion = analyze_emotion(face_roi)

            if dominant_emotion:
                # Highlight emotions of concern
                if dominant_emotion in ['fear', 'sad']:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red for concerning emotions
                    cv2.putText(frame, f"ALERT: {dominant_emotion.upper()}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Log the emotion
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), dominant_emotion])

                    # Play alarm sound
                    play_alarm()

                    # Count the concerning emotion
                    emotion_count[dominant_emotion] += 1

                    # Send email if threshold is reached
                    if emotion_count[dominant_emotion] == threshold:
                        send_email(dominant_emotion, threshold)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green for neutral/safe emotions
                    cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Passenger Safety - Emotion Recognition', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

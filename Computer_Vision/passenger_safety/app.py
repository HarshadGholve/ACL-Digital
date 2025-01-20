import streamlit as st
import cv2
from PIL import Image
from passenger_safety.main import cap, detect_faces, analyze_emotion, play_alarm, send_email

st.title("Real-Time Passenger Safety System")
st.write("This application detects emotional states of passengers in real-time through the webcam.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Create a Streamlit button to start the detection
if st.button('Start Detection'):
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.write("Error: Failed to capture frame.")
            break

        # Detect faces
        faces = detect_faces(frame)

        # Analyze emotions for each detected face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            dominant_emotion = analyze_emotion(face_roi)

            if dominant_emotion:
                if dominant_emotion in ['fear', 'sad']:
                    play_alarm()
                    send_email(dominant_emotion, 10)
                
                # Convert frame to Image for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # Display image in Streamlit
                st.image(image, caption='Real-Time Passenger Safety', use_column_width=True)

# Release the video capture when finished
video_capture.release()

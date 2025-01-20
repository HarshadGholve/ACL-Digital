---

# **Real-Time Passenger Safety System Using Facial Emotion Recognition** 🚨😊

This project focuses on enhancing passenger safety by recognizing emotions such as fear and sadness in real-time using a webcam. The system uses **DeepFace** for emotion recognition and **OpenCV** for face detection. Alerts are generated through sound and email notifications if concerning emotions are detected.

## **📂 Project Structure**  
```
Real-Time-Passenger-Safety/
│
├── src/                            # Source code
│   ├── alert_system.py             # Email and alarm alert system
│   ├── emotion_recognition.py      # Emotion detection logic using DeepFace
│   ├── face_detection.py          # Face detection using OpenCV
│   ├── main.py                    # Main script to run the real-time system
│   └── requirements.txt           # List of required dependencies
└── .gitignore                      # Ignored files (e.g., logs, CSV files)
```

## **🚀 How to Run**  
### **1. Clone the Repository**  
```bash
git clone https://github.com/HarshadGholve/Real-Time-Passenger-Safety.git
cd Real-Time-Passenger-Safety
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Run the Main System**  
```bash
python src/main.py
```

The system will start detecting faces and emotions in real-time. If it detects **fear** or **sadness**, it will play an alarm and send an email notification.

### **4. Optional Streamlit UI**  
For an optional web-based UI, run the following command:
```bash
streamlit run app.py
```

## **⚙️ Requirements**  
- Python 3.10+  
- OpenCV  
- DeepFace  
- smtplib (for email notifications)  
- Streamlit (optional)

## **📦 Datasets**  
This project does not require an external dataset as it uses real-time webcam input. However, ensure that your webcam is functional and accessible for the system to run.

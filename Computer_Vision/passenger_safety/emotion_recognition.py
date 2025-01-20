from deepface import DeepFace

def analyze_emotion(face_roi):
    try:
        # Perform emotion analysis
        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        
        # Access the dominant emotion
        if isinstance(analysis, list):
            return analysis[0]['dominant_emotion']
        else:
            return analysis['dominant_emotion']
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return None

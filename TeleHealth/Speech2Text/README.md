# 🎙️ **Real-Time Speech-to-Text App Using Groq API and Streamlit**

This project is a real-time speech-to-text application that leverages **Groq API** for high-accuracy transcription. Built using **Streamlit**, the app captures audio input from a microphone and converts it into text using various Groq-supported transcription models.

## 🔧 **Features**
- Real-time speech recognition using **SpeechRecognition** and **Streamlit**.
- Integration with **Groq API** for advanced transcription models.
- Supports multiple transcription models with different accuracy and speed:
  - `distil-whisper-large-v3-en`: English-only transcription.
  - `whisper-large-v3-turbo`: Fast, multi-language transcription.
  - `whisper-large-v3`: High-accuracy, general-purpose transcription.

## 🛠️ **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/HarshadGholve/ACL-Digital.git
   cd ACL-Digital/Speech2Text
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📚 **Usage**
1. Select a transcription model from the dropdown.
2. Click **Start Recording** to begin real-time speech transcription.
3. Click **Stop Recording** to end the session.
4. View the transcribed text in the **Final Output** section.

## 🧰 **Tech Stack**
- **Python**  
- **Streamlit**  
- **SpeechRecognition**  
- **Groq API**  

## 🔑 **API Key Setup**
Replace the placeholder API key with your **Groq API key** in the `api_key` parameter:
```python
client = Groq(api_key="your_api_key_here")
```

## 🚀 **Future Enhancements**
- Add language detection.
- Improve UI/UX.
- Implement error handling for better user experience.

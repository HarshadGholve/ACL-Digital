import time
import speech_recognition as sr
import streamlit as st
from groq import Groq
import io

# Initialize the Groq client
client = Groq(api_key="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")  

# Initialize recognizer
r = sr.Recognizer()

# Streamlit session state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "outputs" not in st.session_state:
    st.session_state.outputs = []

# Model options and descriptions
models = {
    "distil-whisper-large-v3-en": "Only for English transcription.",
    "whisper-large-v3-turbo": "Faster transcription, supports multiple languages. Ideal for real-time applications.",
    "whisper-large-v3": "General-purpose transcription model supporting multiple languages with high accuracy."
}

# Function to record and process audio
def record_and_process_audio(selected_model):
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        sentence = ""
        while st.session_state.recording:
            try:
                st.write("Listening...")
                audio = r.listen(source, timeout=5)
                
                # Convert the audio to byte format
                audio_data = io.BytesIO(audio.get_wav_data())

                # Send audio data to Groq API for transcription based on selected model
                transcription = client.audio.transcriptions.create(
                    file=("audio.wav", audio_data.read()),  # Send the audio bytes
                    model=selected_model,  # Use the selected model
                    prompt="Specify context or spelling along with proper punctuation",  # Optional
                    response_format="json",  # Return format
                    language="en",  # Language setting
                    temperature=0.0  # Optional temperature for model
                )

                # Concatenate the detected text to the sentence
                detected_text = transcription.text
                sentence += " " + detected_text

                # Wait for 1 second of silence before displaying the full sentence
                time.sleep(1)

                # After silence, display the full sentence
                st.session_state.outputs.append(f"Text: {sentence.strip()}")
                st.write(f"Text: {sentence.strip()}")
                sentence = ""  # Reset sentence for the next one

            except Exception as e:
                st.write(f"Error: {e}")
                break

# UI
st.title("Real-time Speech-to-Text (Select Transcription Model)")

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a transcription model:",
    options=list(models.keys()),
    format_func=lambda model: f"{model} - {models[model]}"
)

# Display the selected model’s description
st.write(f"**Model Description:** {models[model_choice]}")

# Start/Stop buttons
if not st.session_state.recording:
    if st.button("Start Recording"):
        st.session_state.recording = True
        record_and_process_audio(model_choice)
else:
    if st.button("Stop Recording"):
        st.session_state.recording = False
        st.write("Recording stopped.")

# Display outputs
st.text_area("Final Output: ", "\n".join([output.replace("Text: ", "") for output in st.session_state.outputs]), height=200)

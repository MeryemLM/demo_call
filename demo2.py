import streamlit as st
import speech_recognition as sr
import tempfile
import scipy.io.wavfile as wavfile
import os

import warnings
warnings.filterwarnings("ignore")
import pdfplumber
import assemblyai as aai

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text

# Function to display an image in the sidebar
def display_image(image_path, width=5):
    st.image(image_path, use_column_width="auto", width=width)

# Function to transcribe audio
def transcribe_audio(audio_path):
    aai.settings.api_key = "146c7980fa5a4b6c872033d97234500b"
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(language_code="fr", speaker_labels=True, speakers_expected=2)
    transcript = transcriber.transcribe(audio_path, config)
    return transcript

# Main function
def main():
    st.markdown("<h1 style='text-align:center; color: black;'>Aide à l'agent</h1>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("<h2 style='text-align:center; color: black;'>Sidebar Title</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color: black;'>This is a white sidebar.</p>", unsafe_allow_html=True)

    option = st.sidebar.selectbox("Current option", ["Upload audio file"])
    
    if option == "Upload audio file":
        uploaded_file = st.file_uploader("Uploader un fichier audio", type=["mp3", "wav"])
      
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name
      
            transcript = transcribe_audio(audio_path)
      
            for utterance in transcript.utterances:
                st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                prompt_message="Please understand the essence of the question, considering synonyms and different ways the question might be phrased. Provide the answer exactly as it appears in the provided documents. If the exact information is not available, or you're not confident in the accuracy of the match, reply with 'None'."
                combined_query = f"{prompt_message}\n\nUser's query: {utterance.text}"
                response = qa_interface(combined_query)
                response_text = response["result"]
                if response_text.strip() != "None":
                    st.markdown(f'<span style="color:green">Suggestion : </span> {response_text}', unsafe_allow_html=True)
                else: 
                    response_text = None

if __name__ == "__main__":
    main()

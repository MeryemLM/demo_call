import streamlit as st
import tempfile
import pyaudio
import wave
import speech_recognition as sr
from gtts import gTTS
import os
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")
import pdfplumber

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text

# Assuming your PDF extraction happens here
detected_text = extract_text_from_pdf("./objections.pdf")

# Function to initialize retrieval QA interface
def initialize_qa_interface(openai_api_key):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([detected_text])

    directory = "index_store"
    vector_index = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai_api_key))
    vector_index.save_local(directory)

    vector_index = FAISS.load_local("index_store", OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    qa_interface = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_interface

# Function to record audio and return transcribed text
def record_continuous_audio(sample_rate, speaker, channels=1):
    st.write(f"Recording for {speaker}...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        duration = 5  # Record for 5 seconds

        # Set up audio stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=1024)

        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the audio as a WAV file
        wf = wave.open(tmpfile.name, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Perform speech recognition on the recorded audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmpfile.name) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language="fr-FR")
                return text
            except sr.UnknownValueError:
                pass

# Function to display an image in the sidebar
def display_image(image_path, width=5):
    st.image(image_path, use_column_width="auto", width=width)

# Function to convert text to speech
def text_to_speech(text, language='fr'):
    tts = gTTS(text=text, lang=language, tld='fr')
    tts.save("response.mp3")  # Save the speech as an MP3 file

# Main function
def main():
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
    if not openai_api_key:
        st.error("Please provide the OpenAI API Key.")

    qa_interface = initialize_qa_interface(openai_api_key)

    option = st.sidebar.selectbox("Choose an option", ["Upload audio file", "Agent Help"])
    display_image("./Logo-Les-Echos.png", width=10)

    if option == "Agent Help":
        sample_rate = 16000

        start_button_col, stop_button_col = st.columns(2)
        start_conversation = start_button_col.button("Start Conversation")
        stop_conversation = stop_button_col.button("Stop Conversation")

        if start_conversation:
            col1, col2 = st.columns(2)
            agent_history = []
            client_history = []

            continue_conversation = True

            while continue_conversation:
                with col1:
                    st.markdown("<h2 style='color: green;'>Client:</h2>", unsafe_allow_html=True)
                    client_text = record_continuous_audio(sample_rate, "Client", channels=2)
                    client_history.append(client_text)
                    st.write("Client Transcription:", client_text)

                    if client_text:
                        prompt_message = "Please understand the essence of the question, considering synonyms and different ways the question might be phrased. Provide the answer exactly as it appears in the provided documents. If the exact information is not available, or you're not confident in the accuracy of the match, reply with 'None'."
                        combined_query = f"{prompt_message}\n\nUser's query: {client_text}"
                        response = qa_interface(combined_query)
                        response_text = response["result"]
                        if response_text.strip() != "None":
                            text_to_speech(response_text)
                        else:
                            response_text = None
                            pass

                    with col2:
                        st.markdown("<h2 style='color: blue;'>Agent:</h2>", unsafe_allow_html=True)

                        if response_text:
                            st.markdown(f'<span style="color:green">Suggestion : </span> {response_text}', unsafe_allow_html=True)

                    if stop_conversation:
                        continue_conversation = False

    elif option == "Upload audio file":
        st.warning("Audio recording is not supported in Streamlit Cloud. Please use 'Agent Help' option instead.")

if __name__ == "__main__":
    main()

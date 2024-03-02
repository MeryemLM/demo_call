import streamlit as st
import tempfile
import speech_recognition as sr
import scipy.io.wavfile as wavfile
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
import audio_recorder_streamlit as ars

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text

# Assuming your PDF extraction happens here
detected_text = extract_text_from_pdf("./objections.pdf")

os.environ["OPENAI_API_KEY"] = st.sidebar.text_input('Demo key', type='password')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([detected_text])

directory = "index_store"
vector_index = FAISS.from_documents(texts, OpenAIEmbeddings())
vector_index.save_local(directory)

vector_index = FAISS.load_local("index_store", OpenAIEmbeddings())
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})
qa_interface = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# Function to record audio and return transcribed text
def record_continuous_audio(sample_rate, speaker, channels=1):
    st.write(f"Recording for {speaker}...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        ars.record(tmpfile.name, duration=5)
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

with st.sidebar:
    option = st.sidebar.selectbox("Choose an option", ["Upload audio file", "Agent Help"])
    for _ in range(25):
        st.sidebar.text("")  # Add spacing
    display_image("./Logo-Les-Echos.png", width=10)

# Function to convert text to speech
def text_to_speech(text, language='fr'):
    tts = gTTS(text=text, lang=language, tld='fr')
    tts.save("response.mp3")  # Save the speech as an MP3 file

def transcribe_audio(audio_path):
    transcript = transcriber.transcribe(audio_path, config)
    return transcript

# Main function
def main():

    st.markdown("<h1 style='text-align:center; color: white;'>Aide à l'Agent</h1>", unsafe_allow_html=True)
    for _ in range(3):
       st.text("")
   
    if option == "Agent Help":

        # Set up recording variables
        sample_rate = 16000

        # Create two columns for buttons
        start_button_col, stop_button_col = st.columns(2)
   
        # Create a button to start the conversation
        start_conversation = start_button_col.button("Start Conversation")
   
        # Create a button to stop the conversation
        stop_conversation = stop_button_col.button("Stop Conversation")
   
        # Your existing setup for streamlit interface
   
        if start_conversation:
            col1, col2 = st.columns(2)
            # Lists to store conversation history
            agent_history = []
            client_history = []
   
            continue_conversation = True
   
            while continue_conversation:
                with col1:
                    # Client's turn
                    st.markdown("<h2 style='color: green;'>Client:</h2>", unsafe_allow_html=True)
                    client_text = record_continuous_audio(sample_rate, "Client", channels=2)
                    client_history.append(client_text)  # Add client transcription to history
                    st.write("Client Transcription:", client_text)
   
                    if client_text:
                        # Define your prompt message
                        prompt_message="Please understand the essence of the question, considering synonyms and different ways the question might be phrased. Provide the answer exactly as it appears in the provided documents. If the exact information is not available, or you're not confident in the accuracy of the match, reply with 'None'."
 
                        combined_query = f"{prompt_message}\n\nUser's query: {client_text}"
                        # Use the combined query with the qa_interface
                        response = qa_interface(combined_query)  # Adjusted to use combined query
                        response_text = response["result"]
                        if response_text.strip() != "None":
                            text_to_speech(response_text)
                            #st.audio("response.mp3")
                        else:
                            # Skip playing audio if the response is 'I'm not sure'
                            response_text = None
                            pass
   
                    with col2:
                        # Agent's turn
                        st.markdown("<h2 style='color: blue;'>Agent:</h2>", unsafe_allow_html=True)
       
                        if response_text:
                            # Display the response
                            st.markdown(f'<span style="color:green">Suggestion : </span> {response_text}', unsafe_allow_html=True)
       
                    # Check if "Stop Conversation" button is clicked
                    if stop_conversation:
                        continue_conversation = False  # Stop the conversation loop
 
                # Your existing check to stop conversation
                       
    elif option == "Upload audio file":
        # Add a component to upload an audio file
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
 
        # Create a row for the "Transcription" and "Emotion" buttons
        button_col1, button_col2  = st.columns(2)
 
        # Check if a file has been uploaded
        if uploaded_file is not None:
            # Create a temporary file to save the uploaded audio
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name
 
            # Buttons for transcription and emotion analysis
            if button_col1.button("Start Processing") :
 
                # Transcription of the audio
                transcript = transcribe_audio(audio_path)
 
                for utterance in transcript.utterances:
                   
                    st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                    prompt_message="Please understand the essence of the question, considering synonyms and different ways the question might be phrased. Provide the answer exactly as it appears in

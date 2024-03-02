import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import speech_recognition as sr
from gtts import gTTS
import pdfplumber
import assemblyai
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")
 
# Streamlit page configuration
st.set_page_config(layout="wide")
 
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text
 
# Assuming your PDF extraction
detected_text = extract_text_from_pdf("./objections.pdf")
 
# Sidebar inputs for API keys
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
assemblyai_api_key = st.sidebar.text_input('AssemblyAI API Key', type='password')
 
# Initialize LangChain components
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents([detected_text])
 
directory = "index_store"
vector_index = FAISS.from_documents(texts, OpenAIEmbeddings(api_key=openai_api_key))
vector_index.save_local(directory)
 
vector_index = FAISS.load_local("index_store", OpenAIEmbeddings(api_key=openai_api_key))
retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})
qa_interface = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=openai_api_key),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
 
# Function to transcribe audio with AssemblyAI
def transcribe_audio(audio_path):
    client = assemblyai.Client(assemblyai_api_key)
    transcript = client.transcribe(filename=audio_path)
    return transcript['text']
 
# Function to convert text to speech
def text_to_speech(text, language='fr'):
    tts = gTTS(text=text, lang=language, tld='fr')
    tts.save("response.mp3")  # Save the speech as an MP3 file
 
# Main function
def main():
    st.markdown("<h1 style='text-align:center; color: white;'>Aide à l'Agent</h1>", unsafe_allow_html=True)
    option = st.sidebar.selectbox("Choose an option", ["Record audio", "Upload audio file"])
 
    if option == "Record audio":
        # Use the audio_recorder component for recording
        audio_data = audio_recorder()
        if audio_data is not None:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
                tmpfile.write(audio_data.getbuffer())
                tmpfile.seek(0)
                # Transcribe the recorded audio
                text = transcribe_audio(tmpfile.name)
                st.write("Transcription:", text)
                # Here you can add processing of the transcription
    elif option == "Upload audio file":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
                tmpfile.write(uploaded_file.getbuffer())
                tmpfile.seek(0)
                # Transcribe the uploaded audio file
                text = transcribe_audio(tmpfile.name)
                st.write("Transcription:", text)
                # Here you can add processing of the transcription
 
if __name__ == "__main__":
    main()

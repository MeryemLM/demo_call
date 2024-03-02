import streamlit as st
# Import the audio_recorder_streamlit component
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import tempfile
from gtts import gTTS
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")
import pdfplumber
import assemblyai as aai
st.set_page_config(layout="wide")
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text
# Assuming your PDF extraction happens here
detected_text = extract_text_from_pdf("./objections.pdf")
openai_api_key = st.sidebar.text_input('Demo Key', type='password')
#os.environ["OPENAI_API_KEY"] = st.sidebar.text_input('Demo key', type='password')
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
def text_to_speech(text, language='fr'):
    tts = gTTS(text=text, lang=language, tld='fr')
    tts.save("response.mp3")
def transcribe_audio(audio_path):
    aai.settings.api_key = "146c7980fa5a4b6c872033d97234500b"
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(language_code="fr", speaker_labels=True, speakers_expected=2)
    transcript = transcriber.transcribe(audio_path, config)
    return transcript
def main():
    st.markdown("<h1 style='text-align:center; color: white;'>Aide à l'Agent</h1>", unsafe_allow_html=True)
    option = st.sidebar.selectbox("Choose an option", ["Record audio", "Upload audio file"])
    if option == "Record audio":
        # Use the audio_recorder component for recording
        audio_data = audio_recorder()
        if audio_data is not None:
            # Save the recorded audio to a temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(audio_data)
                audio_path = tmpfile.name
                # Now you can use the audio_path with your transcribe_audio function or any other processing
    elif option == "Upload audio file":
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name
                # Process the uploaded file as before
if __name__ == "__main__":
    main()

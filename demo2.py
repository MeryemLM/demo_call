#WITHOUT NONE

import streamlit as st
import speech_recognition as sr
import tempfile
import scipy.io.wavfile as wavfile
import os

import warnings
warnings.filterwarnings("ignore")
import pdfplumber
import assemblyai as aai

#st.set_page_config(page_icon="🎤", page_title="Airbnb", layout="wide")
st.set_page_config(layout="wide")

 
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text
 
# Assuming your PDF extraction happens here

 
# Function to display an image in the sidebar

def display_image(image_path, width=5):
    st.image(image_path, use_column_width="auto", width=width)

st.markdown("<h1 style='text-align:center; color: white;'>Aide à l'agent</h1>", unsafe_allow_html=True)

with st.sidebar:
    #display_image("./majorel-500x300.jpg", width=250)

    option = st.sidebar.selectbox("Choose an option", ["Upload audio file"])
 
    for _ in range(25):
        st.sidebar.text("")  # Ajouter un espacement
    
 
    display_image("./Logo-Les-Echos.png", width=10)
 


def transcribe_audio(audio_path):
    # Configuration de l'API AssemblyAI
    aai.settings.api_key = "146c7980fa5a4b6c872033d97234500b"

    # Création d'un transcriber
    transcriber = aai.Transcriber()
    # Configuration de la transcription
    config = aai.TranscriptionConfig(language_code="fr", speaker_labels=True, speakers_expected=2)
    # Transcription de l'audio
    transcript = transcriber.transcribe(audio_path, config)
    return transcript





# Main function
def main():

    st.markdown("<h1 style='text-align:center; color: black;'>Aide à l'Agent</h1>", unsafe_allow_html=True)
    for _ in range(3):
       st.text("")
    openai_api_key = st.text_input('Demo key', type='password')
    os.environ["OPENAI_API_KEY"] = openai_api_key
    #os.environ["OPENAI_API_KEY"] = st.text_input('Demo key', type='password')
    if openai_api_key:
        
        from langchain_community.vectorstores import FAISS
        from langchain_community.chat_models import ChatOpenAI
        from langchain.embeddings.openai import OpenAIEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA

        # Assuming your PDF extraction happens here
        detected_text = extract_text_from_pdf("./objections.pdf")
        
        
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

                        
        if option == "Upload audio file":
              # Ajouter un composant pour uploader un fichier audio
            uploaded_file = st.file_uploader("Uploader un fichier audio", type=["mp3", "wav"])
      
              # Créer une rangée pour les boutons "Transcription" et "Emotion"
            button_col1, button_col2  = st.columns(2)
      
              # Vérifier si un fichier a été uploadé
            if uploaded_file is not None:
                  # Créer un fichier temporaire pour enregistrer l'audio uploadé
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    audio_path = tmp_file.name
      
                  # Boutons pour la transcription et l'analyse de l'émotion
                if button_col1.button("Lancez la recherche de suggestion") :
      
                      # Transcription de l'audio
                    transcript = transcribe_audio(audio_path)
      
                    for utterance in transcript.utterances:
                          
                        st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                        prompt_message="Please understand the essence of the question, considering synonyms and different ways the question might be phrased. Provide the answer exactly as it appears in the provided documents. If the exact information is not available, or you're not confident in the accuracy of the match, reply with 'None'."
                        combined_query = f"{prompt_message}\n\nUser's query: {utterance.text}"
                          # Use the combined query with the qa_interface
                        response = qa_interface(combined_query)
                        response_text = response["result"]
                        if response_text.strip() != "None":
                            #st.write("Suggestion :", response_text)
                            st.markdown(f'<span style="color:green">Suggestion : </span> {response_text}', unsafe_allow_html=True)
                        else : 
                            response_text = None
                            pass
      
        else:
                  # Message indiquant à l'utilisateur d'uploader un fichier
            st.write("Veuillez uploader un fichier audio pour commencer la transcription.") 
    else:
        st.warnings("Please enter you demo Key to proceed.")
 
if __name__ == "__main__":
    main()
    

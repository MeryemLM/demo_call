import streamlit as st
import assemblyai as aai
from transformers import pipeline
import tempfile
import openai

st.set_page_config(layout="wide")


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

sentiment_analysis = pipeline(
  "sentiment-analysis",
  framework="pt",
  model="lxyuan/distilbert-base-multilingual-cased-sentiments-student" #SamLowe/roberta-base-go_emotions  #lxyuan/distilbert-base-multilingual-cased-sentiments-student
)
def analyze_sentiment_voice(text):
    results = sentiment_analysis(text)
    sentiment_label = results[0]['label']
    return sentiment_label

# Set up OpenAI API key

openai.api_key = st.sidebar.text_input('Demo key', type='password')


def analyze_emotion(text):
   try:
       content = f"peux tu me donner seulement une émotion exacte sans commentaire, et si tu ne détecte pas une émotion met 'neutre' sans commentaires : par exemple 'Neutre ou frustration ou colère, ...'?\n{text}"
       messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo-16k",
           messages=messages,
           max_tokens=100
       )
       emotion_result = response['choices'][0]['message']['content']
       # Extraire l'émotion de la phrase complète
       emotion = emotion_result.split(":")[-1].strip()
       return emotion
   except Exception as e:
       print(f"Erreur lors de l'analyse de l'émotion : {e}")
       return None
 

# Streamlit app
def main():
     
    # Titre de l'application
    st.markdown("<h1 style='text-align:center;font-size:medium;color: #B01817;'>Transcription audio et analyse émotionnelle</h1>", unsafe_allow_html=True)
    #st.title("Analyse de la transcription audio")
    st.sidebar.image("logo2.jpg", use_column_width=True)

  

    option = st.sidebar.selectbox("Choose an option", ["Upload audio file", "Audio file path"])

    if option == "Upload audio file":
        
        # Ajouter un composant pour uploader un fichier audio
        uploaded_file = st.file_uploader("Uploader un fichier audio", type=["mp3", "wav"])

        # Créer une rangée pour les boutons "Transcription" et "Emotion"
        button_col1, button_col2, button_col3 = st.columns(3)

        # Vérifier si un fichier a été uploadé
        if uploaded_file is not None:
            # Créer un fichier temporaire pour enregistrer l'audio uploadé
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name

            # Boutons pour la transcription et l'analyse de l'émotion
            if button_col1.button("Transcription") :

                # Transcription de l'audio
                transcript = transcribe_audio(audio_path)

                for utterance in transcript.utterances:
                            st.write(f"<span style='color: #922B21;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                            
            # Bouton "Emotion"
            if button_col2.button("Text-based emotion") and uploaded_file:

                # Transcription de l'audio
                transcript = transcribe_audio(audio_path)

                for utterance in transcript.utterances:
                    st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                    
                    sentiment = analyze_emotion(utterance.text)
                    # Affichage du résultat de l'analyse de sentiment
                    st.write("Emotion détectée : ", sentiment)

            if button_col3.button("Voice-based emotion") and uploaded_file:
                # Transcription de l'audio
                transcript = transcribe_audio(audio_path)

                for utterance in transcript.utterances:
                    st.write(f"<span style='color: blue;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
            
                    # Transcription de l'audio
                    sentiment0 = analyze_sentiment_voice(utterance.text)
                    st.write("Sentiment : ", sentiment0)
                 

        else:
            # Message indiquant à l'utilisateur d'uploader un fichier
            st.write("Veuillez uploader un fichier audio pour commencer la transcription.")     
            
    elif option == "Audio file path":
            
        audio_url = st.text_input("Enter the path of the audio file")

        # Boutons pour la transcription et l'analyse de l'émotion
        button_col1, button_col2, button_col3 = st.columns(3)

        if audio_url is not None : 
        
            if button_col1.button("Transcription Texte") :

                # Transcription de l'audio
                transcript = transcribe_audio(audio_url)

                for utterance in transcript.utterances:
                            st.write(f"<span style='color: #922B21;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                            
            # Bouton "Emotion"
            if button_col2.button("Text-based emotion") and audio_url:

                # Transcription de l'audio
                transcript = transcribe_audio(audio_url)

                for utterance in transcript.utterances:
                    st.write(f"<span style='color: #922B21;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
                    sentiment = analyze_emotion(utterance.text)

                    # Affichage du résultat de l'analyse de sentiment
                    st.write("Emotion détectée : ", sentiment) 
                
            if button_col3.button("Voice-based emotion") and audio_url:
                # Transcription de l'audio
                transcript = transcribe_audio(audio_url)

                for utterance in transcript.utterances:
                    st.write(f"<span style='color: #922B21;'>Speaker {utterance.speaker}:</span> {utterance.text}", unsafe_allow_html=True)
            
                    # Transcription de l'audio
                    sentiment0 = analyze_sentiment_voice(utterance.text)
                    st.write("Sentiment : ", sentiment0)
        else:
            # Message indiquant à l'utilisateur d'uploader un fichier
            st.write("Please provide the path.") 

if __name__ == "__main__":
    main()

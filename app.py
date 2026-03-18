import streamlit as st
from dotenv import load_dotenv
from agent import build_agent
from ingest import create_or_update_vectorstore
import os
import io
import speech_recognition as sr
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder


load_dotenv()


st.set_page_config(page_title="Interview Assistant")

# # Initialize / ingest on first run (you can disable auto-ingest in production)
# if st.sidebar.button("(Re)Ingest data / Scrape default link"):
#     with st.spinner("Ingesting data and building vector store..."):
#         create_or_update_vectorstore()
#         st.success("Ingestion finished.")

# Centered page title
st.markdown(
    """
    <h1 style='text-align: center; color: #056162; font-family: Arial, sans-serif;'>
        Talk to Neha
    </h1>
    """,
    unsafe_allow_html=True
)

# Build agent (reads persisted Chroma)
agent, retriever = build_agent()



if "history" not in st.session_state:
    st.session_state.history = []


# Custom CSS for dark theme + chat bubbles + input/button
st.markdown(
    """
    <style>
        /* Full page background */
    body, .css-18e3th9 { 
        background-color: #ffffff;
        color: #000000;
    }


    /* App background */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }

    /* Chat bubbles */
    .assistant-bubble {
        background-color: #2c2c2c !important;
        color: #ffffff !important;
        padding: 8px;
        border-radius: 10px;
        margin: 5px;
        width: 60%;
    }

    .user-bubble {
        background-color: #056162 !important;
        color: #ffffff !important;
        padding: 8px;
        border-radius: 10px;
        margin: 5px;
        width: 60%;
        margin-left: auto;
    }

    /* Input box */
    div.stTextInput>div>input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #444444 !important;
    }

    /* Submit button */
    div.stButton>button {
        background-color: #056162 !important;
        color: #ffffff !important;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# # Chat form
# with st.form(key="chat_form", clear_on_submit=True):
#     user_input = st.text_input(
#         label="I’m ready — whenever you are",
#         key="user_question"
#     )
#     submit = st.form_submit_button()

    # if submit and user_input:
    #     # Example assistant reply
    #     assistant_reply = "This is a sample reply from Assistant."
    #     st.session_state.history.append((user_input, assistant_reply))

voice_mode = st.sidebar.toggle("Enable Voice Mode", value=False)

db = create_or_update_vectorstore()

voice_text = None
if voice_mode:
    st.markdown("**🎙️ Click the mic to record your question:**")
    audio_data = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        key="voice_recorder",
    )
    if audio_data and audio_data["bytes"]:
        recognizer = sr.Recognizer()
        audio_bytes = io.BytesIO(audio_data["bytes"])
        with sr.AudioFile(audio_bytes) as source:
            audio = recognizer.record(source)
        try:
            voice_text = recognizer.recognize_google(audio)
            st.success(f"You said: {voice_text}")
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")

user_input = st.text_input(
    "I'm ready — whenever you are",
    key="user_question",
    placeholder="Type your message…",
)

query = voice_text or user_input
submit = bool(query)

if submit and query:
    docs = db.similarity_search_with_score(query, k=4)
    context = "\n\n".join([d[0].page_content for d in docs])
    query_with_context = f"{query}\n\nContext:\n{context}"
    response = agent.invoke({"query": query_with_context})
    st.session_state.history.append((query, response))

    if voice_mode:
        tts = gTTS(text=str(response), lang="en")
        audio_buf = io.BytesIO()
        tts.write_to_fp(audio_buf)
        audio_buf.seek(0)
        st.audio(audio_buf, format="audio/mp3")





# Chat UI (simple)
for user_msg, bot_msg in st.session_state.history:
     # User message
    st.markdown(f"""
    <div class='user-bubble'>
        <strong>You:</strong> {user_msg}
    </div>
    """, unsafe_allow_html=True)

    # Assistant message
    st.markdown(f"""
    <div class='assistant-bubble'>
        <strong>Assistant:</strong> {bot_msg}
    </div>
    """, unsafe_allow_html=True)

    # st.sidebar.markdown("---")
    # st.sidebar.markdown("## Controls")
    # st.sidebar.markdown("- Use the Reingest button to (re)build the vector DB from the markdown files + default link.")
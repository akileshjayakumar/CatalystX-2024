import os
import streamlit as st
import logging
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from llama_index.llms.nvidia import NVIDIA
from gtts import gTTS
import tempfile
from load_images import load_multimodal_data
from stt.stt import speech_to_text
from audio_recorder_streamlit import audio_recorder


# Set up logging for the terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set the Streamlit page configuration as the first Streamlit command
st.set_page_config(
    layout="wide", page_title="EduX | AI Tutor", page_icon="🧠"
)

# Initialize NVIDIA client


def initialize_nvidia_client():
    try:
        logging.info("Initializing NVIDIA client...")
        api_key = os.getenv("NVIDIA_API_KEY")
        assert api_key.startswith(
            "nvapi-"), "Invalid API Key. Please set NVIDIA_API_KEY correctly."

        return NVIDIA(
            model="meta/llama-3.2-3b-instruct",
            api_key=api_key,
            temperature=0.2,
            max_tokens=1024,
            top_p=0.7
        )
    except Exception as e:
        logging.error(f"Error initializing NVIDIA client: {e}")
        st.error(f"Error initializing NVIDIA client: {e}")

# Initialize settings


def initialize_settings():
    try:
        logging.info("Initializing settings...")
        Settings.embed_model = NVIDIAEmbedding(
            model="NV-Embed-QA", truncate="END"
        )
        Settings.llm = initialize_nvidia_client()  # Use NVIDIA client here
        Settings.text_splitter = SentenceSplitter(chunk_size=600)
    except Exception as e:
        logging.error(f"Error initializing settings: {e}")
        st.error(f"Error initializing settings: {e}")

# Create an index from documents


def create_index(documents):
    try:
        logging.info(f"Creating index for {len(documents)} documents...")
        vector_store = MilvusVectorStore(
            host="127.0.0.1",
            port=19530,
            dim=1024
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    except Exception as e:
        logging.error(f"Error creating index: {e}")
        st.error(f"Error creating index: {e}")

# Text-to-speech function


def text_to_speech(text):
    try:
        logging.info("Running text-to-speech...")
        if text:
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3", autoplay=True)
        else:
            logging.warning("No text available for text-to-speech.")
            st.warning("No text available for text-to-speech.")
    except Exception as e:
        logging.error(f"Error in text-to-speech: {e}")
        st.error(f"Error in text-to-speech: {e}")

# Clear session state


def clear_session_state():
    try:
        logging.info("Clearing session state...")
        for key in ['index', 'history', 'first_load']:
            if key in st.session_state:
                del st.session_state[key]
    except Exception as e:
        logging.error(f"Error clearing session state: {e}")
        st.error(f"Error clearing session state: {e}")

# Initialize session state


def initialize_session():
    try:
        logging.info("Initializing session...")
        if 'first_load' not in st.session_state:
            clear_session_state()
            st.session_state['first_load'] = True

        if 'index' not in st.session_state:
            st.session_state['index'] = None
        if 'history' not in st.session_state:
            st.session_state['history'] = []
    except Exception as e:
        logging.error(f"Error initializing session: {e}")
        st.error(f"Error initializing session: {e}")

# Apply dyslexia-friendly UI styles


def apply_custom_styles():
    try:
        st.markdown("""
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                }
                .dyslexia-friendly {
                    font-size: 20px;
                    background-color: #ffffff;
                    color: #000000;
                    padding: 10px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                }
            </style>
        """, unsafe_allow_html=True)
        logging.info("Applied custom styles.")
    except Exception as e:
        logging.error(f"Error applying custom styles: {e}")
        st.error(f"Error applying custom styles: {e}")

# Layout columns


def layout_columns():
    try:
        logging.info("Setting up UI columns...")
        return st.columns([1, 2], gap="medium")
    except Exception as e:
        logging.error(f"Error setting up UI columns: {e}")
        st.error(f"Error setting up UI columns: {e}")

# Handle file upload


def handle_file_upload():
    try:
        logging.info("Handling file upload...")
        uploaded_files = st.file_uploader(
            "Drag and drop files here", accept_multiple_files=True)
        if uploaded_files and st.button("Process Files", key="process_files"):
            with st.spinner("Processing files..."):
                documents = load_multimodal_data(uploaded_files)
                st.session_state['index'] = create_index(documents)
                st.session_state['history'] = []
                st.success("Files processed and index created!")
    except Exception as e:
        logging.error(f"Error handling file upload: {e}")
        st.error(f"Error handling file upload: {e}")

# Handle camera input


def handle_camera_input():
    try:
        logging.info("Handling camera input...")
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            with st.spinner("Processing captured image..."):
                documents = load_multimodal_data([camera_image])
                st.session_state['index'] = create_index(documents)
                st.session_state['history'] = []
                st.success("Image captured and processed!")
    except Exception as e:
        logging.error(f"Error handling camera input: {e}")
        st.error(f"Error handling camera input: {e}")

# Handle direct question input


def handle_direct_question():
    try:
        logging.info("Handling direct question input...")
        # Reset index when asking questions directly
        st.session_state['index'] = None
        st.session_state['history'] = []
    except Exception as e:
        logging.error(f"Error handling direct question input: {e}")
        st.error(f"Error handling direct question input: {e}")

# Display chat


def display_chat(input_method):
    logging.info(f"Display chat triggered. Input method: {input_method}")

    llm2 = ChatNVIDIA(
        model="meta/llama-3.2-3b-instruct",
        api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    try:
        if input_method == "Ask Question Directly":

            user_input = None
            
            query_method = st.radio("Choose input method:", 
                                    ("Type", "Speak"))
            
            if query_method == "Type":
                user_input = st.chat_input("Enter your query:")
            
            elif query_method == "Speak":
                audio_bytes = audio_recorder()
                if audio_bytes:
                    # Write the audio bytes to a file
                    with st.spinner("Transcribing..."):
                        webm_file_path = "temp_audio.mp3"
                        with open(webm_file_path, "wb") as f:
                            f.write(audio_bytes)

                        user_input = speech_to_text(webm_file_path)
                        os.remove(webm_file_path)

            if user_input:
                logging.info(f"User input: {user_input}")
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append(
                    {"role": "user", "content": user_input})

                # Use NVIDIA LLM's query method for direct questions
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    try:
                        # Send the user input to the LLM and collect response
                        logging.info(
                            f"Sending query to NVIDIA LLM: {user_input}")
                        response_stream = llm2.stream(user_input)

                        # Collect and display the response in chunks
                        for chunk in response_stream:
                            logging.info(f"Received chunk: {chunk}")
                            full_response += chunk.content
                            message_placeholder.markdown(full_response + "▌")

                        message_placeholder.markdown(full_response)

                    except Exception as e:
                        logging.error(f"Error during NVIDIA LLM query: {e}")
                        st.error(f"Error during NVIDIA LLM query: {e}")

                st.session_state['history'].append(
                    {"role": "assistant", "content": full_response})

                # Speak the response if it exists
                if full_response.strip():
                    text_to_speech(full_response)
                else:
                    logging.warning(
                        "No response generated for text-to-speech.")
                    st.warning("No response generated for text-to-speech.")

        elif 'index' in st.session_state and st.session_state['index'] is not None:
            logging.info("Using document index for query.")
            query_engine = st.session_state['index'].as_query_engine(
                similarity_top_k=20, streaming=True)
            
            query_method = st.radio("Choose input method:", 
                                    ("Type", "Speak"))
            
            user_input = None

            if query_method == "Type":
                user_input = st.chat_input("Enter your query:")
            
            elif query_method == "Speak":
                audio_bytes = audio_recorder()
                if audio_bytes:
                    # Write the audio bytes to a file
                    with st.spinner("Transcribing..."):
                        webm_file_path = "temp_audio.mp3"
                        with open(webm_file_path, "wb") as f:
                            f.write(audio_bytes)

                        user_input = speech_to_text(webm_file_path)
                        os.remove(webm_file_path)


            logging.info(f"User input: {user_input}")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append(
                    {"role": "user", "content": user_input})

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    response = query_engine.query(user_input)
                    for token in response.response_gen:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state['history'].append(
                    {"role": "assistant", "content": full_response})

                if full_response.strip():
                    text_to_speech(full_response)
                else:
                    logging.warning(
                        "No response generated for text-to-speech.")
                    st.warning("No response generated for text-to-speech.")

        else:
            logging.warning(
                "No index found. Please upload files or take a picture to create an index first.")
            st.warning(
                "No index found. Please upload files or take a picture to create an index first.")

    except Exception as e:
        logging.error(f"Error in display_chat: {e}")
        st.error(f"Error in display_chat: {e}")

    # Clear chat
    if st.button("Clear Chat"):
        logging.info("Clearing chat...")
        st.session_state['history'] = []
        st.rerun()

# Main application


def main():
    try:
        initialize_session()
        initialize_settings()
        apply_custom_styles()
        col1, col2 = layout_columns()

        with col1:
            st.title("🧠 EduX")
            st.write(
                "This AI tutor chatbot can help you understand and explain homework questions, graphs, plots, charts, and tables. "
                "You can upload files or take a picture to process them, and then the chatbot will provide easy-to-understand answers."
            )

            input_method = st.radio(
                "Choose input method:",
                ("Upload Files", "Take Picture", "Ask Question Directly"),
                help="Choose how you want to provide your study materials or ask questions directly."
            )

            if input_method == "Upload Files":
                handle_file_upload()
            elif input_method == "Take Picture":
                handle_camera_input()
            elif input_method == "Ask Question Directly":
                handle_direct_question()

        with col2:
            st.title("💬 Chat with AI Tutor")
            display_chat(input_method)

    except Exception as e:
        logging.error(f"Error in main function: {e}")
        st.error(f"Error in main function: {e}")


# Run the application
if __name__ == "__main__":
    main()

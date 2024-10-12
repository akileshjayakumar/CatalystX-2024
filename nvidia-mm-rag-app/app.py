import os
import streamlit as st
import logging
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from gtts import gTTS
import tempfile
from load_images import load_multimodal_data

# Set up logging for the terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set the Streamlit page configuration as the first Streamlit command
st.set_page_config(
    layout="wide", page_title="EduX | AI Tutor", page_icon="🧠"
)

# Initialize the ChatNVIDIA client


def initialize_nvidia_client():
    logging.info("Initializing NVIDIA client...")
    return ChatNVIDIA(
        model="meta/llama-3.2-3b-instruct",  # Use your model here
        api_key=os.getenv("NVIDIA_API_KEY"),  # Set your API key here
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )

# Function to initialize LLM and embedding settings


def initialize_settings():
    logging.info("Initializing settings...")
    Settings.embed_model = NVIDIAEmbedding(
        model="nvidia/nv-embedqa-e5-v5", truncate="END"
    )
    Settings.llm = initialize_nvidia_client()
    Settings.text_splitter = SentenceSplitter(chunk_size=600)

# Function to create an index from documents


def create_index(documents):
    logging.info(f"Creating index for {len(documents)} documents...")
    vector_store = MilvusVectorStore(
        host="127.0.0.1",
        port=19530,
        dim=1024
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Function for text-to-speech functionality


def text_to_speech(text):
    logging.info("Running text-to-speech...")
    if text:  # Check if there is any text to convert to speech
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3", autoplay=True)
    else:
        logging.warning("No text available for text-to-speech.")

# Function to clear session state for fresh start


def clear_session_state():
    logging.info("Clearing session state...")
    for key in ['index', 'history', 'first_load']:
        if key in st.session_state:
            del st.session_state[key]

# Function to initialize session state


def initialize_session():
    logging.info("Initializing session...")
    if 'first_load' not in st.session_state:
        clear_session_state()
        st.session_state['first_load'] = True

    if 'index' not in st.session_state:
        st.session_state['index'] = None
    if 'history' not in st.session_state:
        st.session_state['history'] = []

# Function to apply custom dyslexia-friendly UI styles


def apply_custom_styles():
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

# Function to layout UI columns


def layout_columns():
    logging.info("Setting up UI columns...")
    return st.columns([1, 2], gap="medium")

# Function to handle file upload input


def handle_file_upload():
    logging.info("Handling file upload...")
    uploaded_files = st.file_uploader(
        "Drag and drop files here", accept_multiple_files=True)
    if uploaded_files and st.button("Process Files", key="process_files"):
        with st.spinner("Processing files..."):
            documents = load_multimodal_data(uploaded_files)
            st.session_state['index'] = create_index(documents)
            st.session_state['history'] = []
            st.success("Files processed and index created!")

# Function to handle camera input


def handle_camera_input():
    logging.info("Handling camera input...")
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        with st.spinner("Processing captured image..."):
            documents = load_multimodal_data([camera_image])
            st.session_state['index'] = create_index(documents)
            st.session_state['history'] = []
            st.success("Image captured and processed!")

# Function to handle direct question input


def handle_direct_question():
    logging.info("Handling direct question input...")
    # Reset index when asking questions directly
    st.session_state['index'] = None
    st.session_state['history'] = []

# Function to display chat and handle conversation flow


def display_chat(input_method):
    logging.info(f"Display chat triggered. Input method: {input_method}")
    if input_method == "Ask Question Directly":
        llm_client = Settings.llm

        user_input = st.chat_input("Enter your query:")
        logging.info(f"User input: {user_input}")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state['history'].append(
                {"role": "user", "content": user_input})

            # Use ChatNVIDIA's stream method for "Ask Question Directly"
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # Streaming response from the model with error handling
                try:
                    # Send the user input to the LLM directly
                    logging.info(
                        f"Sending user input directly to NVIDIA client: {user_input}")

                    # Send the user input directly, no list wrapping
                    response_stream = llm_client.stream(
                        [{"role": "user", "content": user_input}])

                    # Collect and display the response as it streams
                    for chunk in response_stream:
                        logging.info(f"Received chunk: {chunk}")
                        full_response += chunk["content"]
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                except Exception as e:
                    logging.error(f"An error occurred: {str(e)}")

            st.session_state['history'].append(
                {"role": "assistant", "content": full_response})

            # Automatically speak the response only if there's actual content
            if full_response.strip():
                text_to_speech(full_response)
            else:
                logging.warning("No response generated for text-to-speech.")

    elif 'index' in st.session_state and st.session_state['index'] is not None:
        logging.info("Using document index for query.")
        query_engine = st.session_state['index'].as_query_engine(
            similarity_top_k=20, streaming=True)

        user_input = st.chat_input("Enter your query:")
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

            # Automatically speak the response only if there's actual content
            if full_response.strip():
                text_to_speech(full_response)
            else:
                logging.warning("No response generated for text-to-speech.")

    else:
        logging.warning(
            "No index found. Please upload files or take a picture to create an index first.")

    # Add a clear button
    if st.button("Clear Chat"):
        logging.info("Clearing chat...")
        st.session_state['history'] = []
        st.rerun()

# Main application logic


def main():
    # Initialize session state
    initialize_session()

    # Initialize LLM and embedding settings
    initialize_settings()

    # Apply custom styles
    apply_custom_styles()

    # Define the UI layout
    col1, col2 = layout_columns()

    # Left panel for input method selection
    with col1:
        st.title("🧠 EduX")
        st.write(
            "This AI tutor chatbot can help you understand and explain homework questions, graphs, plots, charts, and tables. "
            "You can upload files or take a picture to process them, and then the chatbot will provide easy-to-understand answers."
        )

        # Input method options
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

    # Right panel for chat
    with col2:
        st.title("💬 Chat with AI Tutor")
        display_chat(input_method)


# Run the application
if __name__ == "__main__":
    main()

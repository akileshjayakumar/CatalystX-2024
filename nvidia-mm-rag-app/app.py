from load_images import load_multimodal_data
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from gtts import gTTS
import tempfile
import streamlit as st

# Set the Streamlit page configuration
st.set_page_config(
    layout="wide", page_title="EduX | AI Tutor", page_icon="🧠")

# Initialize LLM and other settings


def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(
        model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
    Settings.text_splitter = SentenceSplitter(chunk_size=600)

# Create index for documents


def create_index(documents):
    vector_store = MilvusVectorStore(
        host="127.0.0.1",
        port=19530,
        dim=1024
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Text-to-Speech Functionality


def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(f"{fp.name}.mp3")
        st.audio(f"{fp.name}.mp3", format="audio/mp3", autoplay=True)

# Clear session state for a fresh start


def clear_session_state():
    if 'index' in st.session_state:
        del st.session_state['index']
    if 'history' in st.session_state:
        del st.session_state['history']

# Main application logic


def main():
    # Clear session state on every startup or refresh
    if 'first_load' not in st.session_state:
        clear_session_state()
        st.session_state['first_load'] = True

    initialize_settings()

    # Set font and color customization for dyslexia-friendly UI
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

    # Define the UI layout
    col1, col2 = st.columns([1, 2], gap="medium")

    # Left panel for input method selection
    with col1:
        st.title("🧠 EduX")
        st.write("This AI tutor chatbot can help you understand and explain homework questions, graphs, plots, charts, and tables. You can upload files or take a picture to process them, and then the chatbot will provide easy-to-understand answers.")

        # Input method options
        input_method = st.radio("Choose input method:", ("Upload Files", "Take Picture"),
                                help="Choose how you want to provide your study materials.")

        if input_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Drag and drop files here", accept_multiple_files=True)
            if uploaded_files and st.button("Process Files", key="process_files"):
                with st.spinner("Processing files..."):
                    documents = load_multimodal_data(uploaded_files)
                    st.session_state['index'] = create_index(documents)
                    st.session_state['history'] = []
                    st.success("Files processed and index created!")

        elif input_method == "Take Picture":
            # Use camera input to allow users to take a picture directly
            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                with st.spinner("Processing captured image..."):
                    # Convert the image to a suitable format
                    documents = load_multimodal_data([camera_image])
                    st.session_state['index'] = create_index(documents)
                    st.session_state['history'] = []
                    st.success("Image captured and processed!")

    with col2:
        if 'index' in st.session_state:
            st.title("💬 Chat with AI Tutor")
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            query_engine = st.session_state['index'].as_query_engine(
                similarity_top_k=20, streaming=True)

            user_input = st.text_input(
                "Enter your question:", key="user_input", help="Ask a question related to the uploaded content.")

            # Handle user query
            if user_input:
                # Append user input to history
                st.session_state['history'].append(
                    {"role": "user", "content": user_input})

                # Generate assistant response
                with st.spinner("Generating response..."):
                    response = query_engine.query(user_input)
                    full_response = "".join(response.response_gen)

                st.session_state['history'].append(
                    {"role": "assistant", "content": full_response})

                # Automatically speak the response
                text_to_speech(full_response)

            # Display chat history
            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    if message["role"] == "user":
                        st.markdown(f"<div class='dyslexia-friendly' style='background-color: #e6f7ff;'><strong>User:</strong> {
                                    message['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='dyslexia-friendly' style='background-color: #fff3e6;'><strong>AI Tutor:</strong> {
                                    message['content']}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

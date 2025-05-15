# CatalystX 2024 Hackathon

## EduX - AI-Powered Educational Assistant

EduX is an interactive AI tutor application built with Streamlit that helps students learn from their educational materials. The app features:

- **Document Processing**: Upload PDFs, PowerPoint files, and images
- **Multimodal Learning**: Processes text, tables, images, and graphs from educational materials
- **AI-Powered Tutoring**: Uses NVIDIA AI models to answer questions about uploaded content
- **Accessibility Features**: Includes speech-to-text and text-to-speech capabilities
- **Dyslexia-Friendly UI**: Custom interface designed for better readability

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up NVIDIA API key in environment variables
3. Run the app: `streamlit run app.py`

## Technology Stack

- Streamlit for the web interface
- NVIDIA AI models for embeddings and language processing
- LlamaIndex for document indexing and retrieval
- PyMuPDF for PDF processing
- Python-pptx for PowerPoint file handling

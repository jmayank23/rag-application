import streamlit as st
from sidebar import display_sidebar
from chat_interface import display_chat_interface
from landing_page import display_landing_page
import uuid
import os
import sys
from pathlib import Path

# Database initialization
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(BACKEND_DIR)

# Initialize database tables
try:
    from db_utils import create_application_logs, create_document_store, create_sessions_table
    print("Initializing database tables...")
    create_application_logs()
    create_document_store()
    create_sessions_table()
    print("Database tables initialized successfully!")
except Exception as e:
    print(f"Error initializing database tables: {str(e)}")

st.set_page_config(
    page_title="Langchain RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # Dictionary to store all sessions
    
# Flag to determine if we're in a chat session or on the landing page
if "in_chat_session" not in st.session_state:
    st.session_state.in_chat_session = False

# Set default values for model, vector_db, and embedding_model if not already set
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"

if "vector_db" not in st.session_state:
    st.session_state.vector_db = "chromadb"

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = "openai"

if "previous_vector_db" not in st.session_state:
    st.session_state.previous_vector_db = st.session_state.vector_db

if "previous_embedding_model" not in st.session_state:
    st.session_state.previous_embedding_model = st.session_state.embedding_model

# Helper function to get file path for a document
def get_document_path(filename, vector_db=None):
    """
    Get the full path to a document file
    
    Args:
        filename: Name of the document file
        vector_db: Vector database where the file is stored
        
    Returns:
        Path object for the document
    """
    if vector_db is None:
        vector_db = st.session_state.get("vector_db", "chromadb")
        
    # Build path to the file
    upload_dir = Path("uploads") / vector_db
    file_path = upload_dir / filename
    
    # Check if file exists
    if file_path.exists():
        return file_path
    else:
        return None

# Conditional rendering based on whether we're in a chat session
if st.session_state.in_chat_session:
    # Display the chat interface and sidebar for chat session
    display_sidebar()
    display_chat_interface()
else:
    # Display the landing page
    display_landing_page()

# Add footer with information
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ddd;">
</div>
""", unsafe_allow_html=True)
import streamlit as st
from api_utils import upload_document, list_documents, delete_document, generate_chat_title
import datetime
import os
import sys
import time

# Add the backend directory to the path using absolute path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(BACKEND_DIR)
from db_utils import save_session, get_session, delete_session, get_all_sessions

def display_sidebar():
    """
    Display the sidebar with chat history and document management
    """
    with st.sidebar:
        st.title("RAG Assistant")
        
        # SECTION 1: Chat History at the top
        st.markdown("### Chat History")
        
        # New Chat button
        if st.button("New Chat", key="new_chat_sidebar"):
            # Save current chat if it exists
            if "messages" in st.session_state and st.session_state.messages:
                save_current_session()
            
            # Exit chat session (go back to landing page)
            st.session_state.in_chat_session = False
            st.rerun()
        
        # Display chat history
        display_chat_history()
        
        # SECTION 2: Document Management below chat history
        st.markdown("### Document Management")
        display_document_management()

def display_chat_history():
    """
    Display chat history in the sidebar
    """
    try:
        # Get current session ID
        current_session_id = st.session_state.get("session_id")
        
        # Load saved sessions from the database
        sessions = get_all_sessions()
        
        # Debug: print raw session data to console
        debug_session_data()
        
        if not sessions:
            st.info("No saved chat sessions.")
            return
        
        # Sort sessions by timestamp (most recent first)
        sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Current session indicator
        if current_session_id:
            current_session = next((s for s in sessions if s.get('id') == current_session_id), None)
            if current_session:
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #1e2130; border-left: 5px solid #4e5d95; color: #ffffff;">
                    <strong>Current Session:</strong> {current_session.get('name', 'Untitled')}
                    <br><small>Model: {current_session.get('model', 'Unknown')}</small>
                    <br><small>Messages: {current_session.get('message_count', 0)}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Past sessions
        st.markdown("#### Past Sessions")
        
        for session in sessions:
            session_id = session.get('id')
            
            # Skip current session as it's already displayed above
            if session_id == current_session_id:
                continue
                
            # Format timestamp 
            timestamp = session.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    formatted_time = timestamp
            else:
                formatted_time = "Unknown"
            
            # Display compact session info
            st.markdown(f"""
            <div style="padding: 5px 0; margin-bottom: 3px;">
                <strong>{session.get('name', 'Untitled')}</strong><br>
                <small style="color: #ffffff;">{formatted_time} • {session.get('message_count', 0)} msgs</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a horizontal button layout
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load", key=f"load_{session_id}", use_container_width=True):
                    load_session(session_id)
                    st.success(f"Loaded session: {session.get('name', 'Untitled')}")
                    st.rerun()
                    
            with col2:
                if st.button("Delete", key=f"delete_{session_id}", use_container_width=True):
                    delete_session(session_id)
                    st.success(f"Deleted session: {session.get('name', 'Untitled')}")
                    st.rerun()
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        print(f"Error in display_chat_history: {str(e)}")

def display_document_management():
    """
    Display document management section in the sidebar
    """
    # Document upload - fixed section instead of dropdown
    st.subheader("Upload New Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv", "docx", "html"], key="doc_uploader")
    
    if uploaded_file:
        # Get the current vector DB and embedding model from session state
        vector_db = st.session_state.get("vector_db", "chromadb")
        embedding_model = st.session_state.get("embedding_model", "openai")
        
        # Create directories if they don't exist
        upload_dir = os.path.join("..", "backend", "uploads", vector_db)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file locally
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner(f"Processing document with {embedding_model} embeddings..."):
            # Call the backend API to process the document
            response = upload_document(
                file=uploaded_file,
                vector_db=vector_db,
                embedding_model=embedding_model
            )
            
            if response:
                st.success(f"File uploaded and indexed successfully to {vector_db}!")
            else:
                st.error("Failed to process the document. Check the backend logs for details.")
        
        # Auto-refresh document list
        st.session_state.last_upload_time = time.time()
    
    # Show document list for current vector DB
    st.subheader(f"Documents in {st.session_state.get('vector_db', 'chromadb').capitalize()}")
    
    if st.button("Refresh Document List"):
        st.session_state.last_upload_time = time.time()
    
    # Get the current vector DB from session state
    current_vector_db = st.session_state.get("vector_db", "chromadb")
    
    # List files for the current vector DB
    document_dir = os.path.join("..", "backend", "uploads", current_vector_db)
    if os.path.exists(document_dir):
        files = [f for f in os.listdir(document_dir) if os.path.isfile(os.path.join(document_dir, f))]
        if files:
            for file in files:
                # Display each file with a delete button
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(file)
                with col2:
                    if st.button("🗑️", key=f"del_doc_{file}", help="Delete document"):
                        try:
                            # Delete the file
                            file_path = os.path.join(document_dir, file)
                            os.remove(file_path)
                            st.success(f"Deleted '{file}'")
                            # Refresh the display
                            st.session_state.last_upload_time = time.time()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
        else:
            st.info(f"No documents found in {current_vector_db}.")
    else:
        st.info(f"No documents found in {current_vector_db}.")

def save_current_session():
    """Save the current chat session to the database."""
    if "messages" not in st.session_state or not st.session_state.messages:
        print("No messages to save")
        return
    
    # Get current session ID
    session_id = st.session_state.get("session_id")
    if not session_id:
        # Generate a new session ID if none exists
        import uuid
        session_id = str(uuid.uuid4())
        st.session_state.session_id = session_id
        print(f"Generated new session ID: {session_id}")
    else:
        print(f"Using existing session ID: {session_id}")
    
    # Print message count for debugging
    print(f"Saving session with {len(st.session_state.messages)} messages")
    print(f"First message: {st.session_state.messages[0] if st.session_state.messages else 'None'}")
    
    # Check if we already have a title for this session
    existing_session = None
    try:
        existing_session = get_session(session_id)
    except Exception as e:
        print(f"Error checking existing session: {str(e)}")
    
    # Only generate a title if:
    # 1. We have at least one message from both user and assistant (enough context for a meaningful title)
    # 2. This session doesn't already have a custom title (not the default)
    has_user_message = any(m["role"] == "user" for m in st.session_state.messages)
    has_assistant_message = any(m["role"] == "assistant" for m in st.session_state.messages)
    needs_title_generation = has_user_message and has_assistant_message
    
    if existing_session and existing_session.get("name") and existing_session.get("name") != "New Chat Session":
        # We already have a custom title, use it
        session_name = existing_session.get("name")
        print(f"Using existing session name: {session_name}")
    elif needs_title_generation:
        # Generate a title using the LLM
        with st.spinner("Generating session title..."):
            # Get current model for title generation
            model = st.session_state.get("model", "gpt-4o-mini")
            
            # Generate title using LLM
            session_name = generate_chat_title(st.session_state.messages, model)
            
            # If title generation failed, fall back to the default method
            if not session_name or session_name == "New Chat Session":
                first_user_msg = next((m for m in st.session_state.messages if m["role"] == "user"), None)
                if first_user_msg:
                    # Truncate long messages
                    session_name = first_user_msg["content"][:30] + "..." if len(first_user_msg["content"]) > 30 else first_user_msg["content"]
                else:
                    session_name = f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            print(f"Generated new session name: {session_name}")
    else:
        # Not enough messages for title generation or no specific reason to generate a title
        session_name = existing_session.get("name") if existing_session and existing_session.get("name") else f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        print(f"Using default session name: {session_name}")
    
    print(f"Session name: {session_name}")
    
    # Get current model, vector_db, and embedding_model
    model = st.session_state.get("model", "gpt-4o-mini")
    vector_db = st.session_state.get("vector_db", "chromadb")
    embedding_model = st.session_state.get("embedding_model", "openai")
    
    # Save the session using the shared db_utils function
    save_session(
        session_id=session_id,
        session_name=session_name,
        messages=st.session_state.messages,
        model=model,
        vector_db=vector_db,
        embedding_model=embedding_model
    )
    
    print(f"Session saved with ID: {session_id}")

def load_session(session_id):
    """
    Load a chat session from the database.
    
    Args:
        session_id (str): ID of the session to load
    """
    # Check if we need to save the current session first
    if "session_id" in st.session_state and st.session_state.session_id and st.session_state.session_id != session_id:
        save_current_session()
    
    # Get the session data using the shared db_utils function
    session_data = get_session(session_id)
    
    if not session_data:
        st.error(f"Failed to load session {session_id}")
        return
    
    # Update session state with loaded data
    st.session_state.session_id = session_id
    st.session_state.messages = session_data.get('messages', [])
    st.session_state.model = session_data.get('model', "gpt-4o-mini")
    st.session_state.vector_db = session_data.get('vector_db', "chromadb")
    st.session_state.embedding_model = session_data.get('embedding_model', "openai")
    
    # Set the flag to show we're in a chat session
    st.session_state.in_chat_session = True

def format_timestamp(timestamp_str):
    """Format ISO timestamp string to human-readable format"""
    try:
        if isinstance(timestamp_str, str):
            dt = datetime.datetime.fromisoformat(timestamp_str)
            return dt.strftime("%Y-%m-%d %H:%M")
    except:
        pass
    return "Unknown time"

def debug_session_data():
    """Debug function to print raw session data to the console"""
    try:
        # Import the debug function
        sys.path.append(BACKEND_DIR)
        from db_utils import debug_sessions_table
        
        # Call the debug function
        debug_sessions_table()
    except Exception as e:
        print(f"Error debugging sessions: {str(e)}")

def test_db_connection():
    """Test database connection and table creation"""
    try:
        # Test if we can connect to the database
        from db_utils import get_db_connection, create_sessions_table
        
        # Get database path
        from db_utils import DB_NAME
        print(f"Using database at: {DB_NAME}")
        
        # Test connection
        conn = get_db_connection()
        print(f"Database connection successful")
        
        # Test table creation
        create_sessions_table()
        print(f"Sessions table created or already exists")
        
        # Check if we can query the table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        table_exists = cursor.fetchone()
        print(f"Sessions table exists: {table_exists is not None}")
        
        # Close connection
        conn.close()
        
        return True
    except Exception as e:
        print(f"Database test failed: {str(e)}")
        return False

# Run the database test at startup
test_result = test_db_connection() 
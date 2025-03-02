import streamlit as st
from api_utils import upload_document, list_documents, delete_document, generate_chat_title
import datetime
import os
import sqlite3
import time

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
    # Get current session ID
    current_session_id = st.session_state.get("session_id")
    
    # Load saved sessions from the database
    sessions = load_saved_sessions()
    
    if not sessions:
        st.info("No saved chat sessions.")
        return
    
    # Sort sessions by timestamp (most recent first)
    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Current session indicator
    if current_session_id:
        current_session = next((s for s in sessions if s.get('session_id') == current_session_id), None)
        if current_session:
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #e6f3ff; border-left: 5px solid #1e88e5;">
                <strong>Current Session:</strong> {current_session.get('name', 'Untitled')}
                <br><small>Model: {current_session.get('model', 'Unknown')}</small>
                <br><small>Messages: {current_session.get('message_count', 0)}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Past sessions
    st.markdown("#### Past Sessions")
    
    for session in sessions:
        session_id = session.get('session_id')
        
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
            <small style="color: #666;">{formatted_time} • {session.get('message_count', 0)} msgs</small>
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
            if st.button("Delete", key=f"del_{session_id}", use_container_width=True):
                delete_session(session_id)
                st.success(f"Deleted session: {session.get('name', 'Untitled')}")
                st.rerun()
        
        # Add a subtle divider
        st.markdown('<hr style="margin: 5px 0; border-top: 1px solid #f0f0f0;">', unsafe_allow_html=True)

def display_document_management():
    """
    Display document management section in the sidebar
    """
    # Document upload - fixed section instead of dropdown
    st.subheader("Upload New Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "csv"], key="doc_uploader")
    
    if uploaded_file:
        # Get the current vector DB from session state
        vector_db = st.session_state.get("vector_db", "chromadb")
        
        # Create directories if they don't exist
        upload_dir = os.path.join("..", "backend", "uploads", vector_db)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Call the backend API to process the document
        st.success(f"File uploaded successfully to {vector_db}!")
        
        # Optional: Auto-refresh document list
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
        return
    
    # Connect to the SQLite database
    conn = sqlite3.connect('rag_app.db')
    cursor = conn.cursor()
    
    # Create the sessions table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        timestamp TEXT,
        model TEXT,
        messages TEXT,
        vector_db TEXT,
        embedding_model TEXT
    )
    ''')
    
    # Get current session data
    session_id = st.session_state.get("session_id")
    if not session_id:
        # Generate a new session ID if none exists
        import uuid
        session_id = str(uuid.uuid4())
        st.session_state.session_id = session_id
    
    # Generate a title using the LLM if we have messages
    if len(st.session_state.messages) > 0:
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
    else:
        session_name = f"Session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Convert messages to string format
    import json
    messages_str = json.dumps(st.session_state.messages)
    
    # Current timestamp
    timestamp = datetime.datetime.now().isoformat()
    
    # Current model
    model = st.session_state.get("model", "gpt-4o-mini")
    
    # Current vector DB and embedding model
    vector_db = st.session_state.get("vector_db", "chromadb")
    embedding_model = st.session_state.get("embedding_model", "openai")
    
    # Insert or update the session
    cursor.execute('''
    INSERT OR REPLACE INTO sessions (id, name, timestamp, model, messages, vector_db, embedding_model)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, session_name, timestamp, model, messages_str, vector_db, embedding_model))
    
    conn.commit()
    conn.close()

def load_saved_sessions():
    """Load saved sessions from the database."""
    # Connect to the SQLite database
    try:
        conn = sqlite3.connect('rag_app.db')
        cursor = conn.cursor()
        
        # Create the sessions table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            timestamp TEXT,
            model TEXT,
            messages TEXT,
            vector_db TEXT,
            embedding_model TEXT
        )
        ''')
        
        # Get all sessions
        cursor.execute('SELECT id, name, timestamp, model, messages, vector_db, embedding_model FROM sessions')
        sessions = []
        
        for row in cursor.fetchall():
            session_id, name, timestamp, model, messages_str, vector_db, embedding_model = row
            
            # Count messages
            import json
            try:
                messages = json.loads(messages_str)
                message_count = len(messages)
            except:
                messages = []
                message_count = 0
                
            sessions.append({
                'session_id': session_id,
                'name': name,
                'timestamp': timestamp,
                'model': model,
                'message_count': message_count,
                'messages': messages,
                'vector_db': vector_db,
                'embedding_model': embedding_model
            })
            
        conn.close()
        return sessions
    except Exception as e:
        st.error(f"Error loading sessions: {e}")
        return []

def load_session(session_id):
    """Load a specific session and set it as the current session."""
    sessions = load_saved_sessions()
    session = next((s for s in sessions if s.get('session_id') == session_id), None)
    
    if session:
        # Set session data in session state
        st.session_state.session_id = session_id
        st.session_state.messages = session.get('messages', [])
        st.session_state.model = session.get('model', 'gpt-4o-mini')
        st.session_state.vector_db = session.get('vector_db', 'chromadb')
        st.session_state.embedding_model = session.get('embedding_model', 'openai')
        
        # Ensure we're in a chat session
        st.session_state.in_chat_session = True

def delete_session(session_id):
    """Delete a specific session from the database."""
    try:
        conn = sqlite3.connect('rag_app.db')
        cursor = conn.cursor()
        
        # Delete the session
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        
        conn.commit()
        conn.close()
        
        # If we deleted the current session, clear it
        if st.session_state.get("session_id") == session_id:
            st.session_state.session_id = None
            st.session_state.messages = []
    except Exception as e:
        st.error(f"Error deleting session: {e}")

def format_timestamp(timestamp_str):
    """Format ISO timestamp string to human-readable format"""
    try:
        if isinstance(timestamp_str, str):
            dt = datetime.datetime.fromisoformat(timestamp_str)
            return dt.strftime("%Y-%m-%d %H:%M")
    except:
        pass
    return "Unknown time" 
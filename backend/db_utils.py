import sqlite3
from datetime import datetime
import json
import logging
import os
import uuid
import sys

# Define path to use the database in the root directory
# This ensures both frontend and backend use the same database file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(ROOT_DIR, "rag_app.db")

# Print the database path for debugging
print(f"Database path: {DB_NAME}")

def get_db_connection():
    """
    Get a connection to the SQLite database with row factory enabled.
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    try:
        # Check if the directory is writable
        db_dir = os.path.dirname(DB_NAME)
        if not os.access(db_dir, os.W_OK):
            print(f"Warning: Directory {db_dir} is not writable!")
        
        # Create connection to database
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        # Re-raise the exception to be handled by the caller
        raise

def create_application_logs():
    """Create the application_logs table if it doesn't exist."""
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    """
    Insert a new log entry into the application_logs table.
    
    Args:
        session_id (str): Session ID
        user_query (str): User's query
        gpt_response (str): AI's response
        model (str): Model used for the response
    """
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    """
    Get the chat history for a specific session.
    
    Args:
        session_id (str): Session ID
        
    Returns:
        list: List of message dictionaries with 'role' and 'content' keys
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

def create_document_store():
    """Create the document_store table if it doesn't exist."""
    conn = get_db_connection()
    
    # Check if the table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_store'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create table with all columns
        conn.execute('''CREATE TABLE document_store
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT,
                         vector_db TEXT DEFAULT 'chromadb',
                         embedding_model TEXT DEFAULT 'openai',
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    else:
        # Check if the columns exist and add them if they don't
        cursor.execute("PRAGMA table_info(document_store)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'vector_db' not in columns:
            conn.execute("ALTER TABLE document_store ADD COLUMN vector_db TEXT DEFAULT 'chromadb'")
        
        if 'embedding_model' not in columns:
            conn.execute("ALTER TABLE document_store ADD COLUMN embedding_model TEXT DEFAULT 'openai'")
    
    conn.commit()
    conn.close()

def insert_document_record(filename, vector_db='chromadb', embedding_model='openai'):
    """
    Insert a new document record into the document_store table.
    
    Args:
        filename (str): Document filename
        vector_db (str, optional): Vector database type. Defaults to 'chromadb'.
        embedding_model (str, optional): Embedding model type. Defaults to 'openai'.
        
    Returns:
        int: Document ID
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename, vector_db, embedding_model) VALUES (?, ?, ?)', 
                  (filename, vector_db, embedding_model))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    """
    Delete a document record from the document_store table.
    
    Args:
        file_id (int): Document ID
        
    Returns:
        bool: Whether the operation was successful
    """
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents():
    """
    Get all document records from the document_store table.
    
    Returns:
        list: List of document dictionaries
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_store'")
        if not cursor.fetchone():
            print("Warning: document_store table doesn't exist")
            conn.close()
            return []
        
        # Check which columns exist
        cursor.execute("PRAGMA table_info(document_store)")
        columns = [info[1] for info in cursor.fetchall()]
        
        # Build a query based on available columns
        select_columns = ['id', 'filename']
        if 'vector_db' in columns:
            select_columns.append('vector_db')
        else:
            print("Warning: vector_db column not found in document_store table")
            
        if 'embedding_model' in columns:
            select_columns.append('embedding_model')
        else:
            print("Warning: embedding_model column not found in document_store table")
            
        # Use created_at for sorting if available, otherwise don't sort
        sort_clause = ""
        if 'created_at' in columns:
            select_columns.append('created_at')
            sort_clause = " ORDER BY created_at DESC"
            
        # Build and execute query
        query = f"SELECT {', '.join(select_columns)} FROM document_store{sort_clause}"
        cursor.execute(query)
        documents = cursor.fetchall()
        conn.close()
        return [dict(doc) for doc in documents]
    except Exception as e:
        print(f"Error in get_all_documents: {str(e)}")
        # Return empty list instead of failing
        return []

def get_documents_by_vector_db(vector_db):
    """
    Get document records for a specific vector database.
    
    Args:
        vector_db (str): Vector database type
        
    Returns:
        list: List of document dictionaries
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_store'")
        if not cursor.fetchone():
            print("Warning: document_store table doesn't exist")
            conn.close()
            return []
        
        # Check which columns exist
        cursor.execute("PRAGMA table_info(document_store)")
        columns = [info[1] for info in cursor.fetchall()]
        
        # Check if vector_db column exists
        if 'vector_db' not in columns:
            print("Warning: vector_db column not found in document_store table")
            conn.close()
            return []
        
        # Build a query based on available columns
        select_columns = ['id', 'filename', 'vector_db']
        if 'embedding_model' in columns:
            select_columns.append('embedding_model')
            
        # Use created_at for sorting if available, otherwise don't sort
        sort_clause = ""
        if 'created_at' in columns:
            select_columns.append('created_at')
            sort_clause = " ORDER BY created_at DESC"
            
        # Build and execute query
        query = f"SELECT {', '.join(select_columns)} FROM document_store WHERE vector_db = ?{sort_clause}"
        cursor.execute(query, (vector_db,))
        documents = cursor.fetchall()
        conn.close()
        return [dict(doc) for doc in documents]
    except Exception as e:
        print(f"Error in get_documents_by_vector_db: {str(e)}")
        # Return empty list instead of failing
        return []

# Session management functions that can be shared with frontend
def create_sessions_table():
    """Create the sessions table if it doesn't exist."""
    try:
        conn = get_db_connection()
        conn.execute('''
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
        conn.commit()
        conn.close()
        print("Sessions table created successfully")
        return True
    except Exception as e:
        print(f"Error creating sessions table: {str(e)}")
        return False

def save_session(session_id, session_name, messages, model="gpt-4o-mini", vector_db="chromadb", embedding_model="openai"):
    """
    Save a chat session to the sessions table.
    
    Args:
        session_id (str): Session ID
        session_name (str): Session name
        messages (list): List of message dictionaries
        model (str, optional): Model used for the session. Defaults to "gpt-4o-mini".
        vector_db (str, optional): Vector database type. Defaults to "chromadb".
        embedding_model (str, optional): Embedding model type. Defaults to "openai".
        
    Returns:
        str: Session ID
    """
    # Ensure the sessions table exists
    create_sessions_table()
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert messages to JSON string
    messages_str = json.dumps(messages)
    
    # Current timestamp
    timestamp = datetime.now().isoformat()
    
    # Insert or replace the session
    cursor.execute('''
    INSERT OR REPLACE INTO sessions (id, name, timestamp, model, messages, vector_db, embedding_model)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (session_id, session_name, timestamp, model, messages_str, vector_db, embedding_model))
    
    conn.commit()
    conn.close()
    
    return session_id

def get_session(session_id):
    """
    Get a chat session from the sessions table.
    
    Args:
        session_id (str): Session ID
        
    Returns:
        dict: Session data or None if not found
    """
    # Ensure the sessions table exists
    create_sessions_table()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
    session = cursor.fetchone()
    
    conn.close()
    
    if session:
        session_dict = dict(session)
        try:
            session_dict['messages'] = json.loads(session_dict['messages'])
        except:
            session_dict['messages'] = []
        return session_dict
    
    return None

def delete_session(session_id):
    """
    Delete a chat session from the sessions table.
    
    Args:
        session_id (str): Session ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure the sessions table exists
    create_sessions_table()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    
    conn.commit()
    conn.close()
    
    return True

def get_all_sessions():
    """
    Get all chat sessions from the sessions table.
    
    Returns:
        list: List of session dictionaries
    """
    # Ensure the sessions table exists
    create_sessions_table()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, timestamp, model, vector_db, embedding_model, messages FROM sessions ORDER BY timestamp DESC')
    sessions = cursor.fetchall()
    
    conn.close()
    
    result = []
    for session in sessions:
        session_dict = dict(session)
        
        # Default message count to 0
        session_dict['message_count'] = 0
        
        # Try to parse the messages JSON
        if 'messages' in session_dict and session_dict['messages']:
            try:
                # Parse messages JSON
                messages = json.loads(session_dict['messages'])
                
                # Update message count if messages is a list
                if isinstance(messages, list):
                    session_dict['message_count'] = len(messages)
                    print(f"Session {session_dict['name']} has {len(messages)} messages")
                else:
                    print(f"Session {session_dict['name']} has messages but not in list format")
            except Exception as e:
                print(f"Error parsing messages for session {session_dict['name']}: {str(e)}")
        else:
            print(f"Session {session_dict['name']} has no messages field or it's empty")
        
        # Remove the messages field to avoid sending large amounts of data
        if 'messages' in session_dict:
            del session_dict['messages']
        
        result.append(session_dict)
    
    return result

def debug_sessions_table():
    """Debug function to print raw session data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, name, messages FROM sessions')
    sessions = cursor.fetchall()
    
    conn.close()
    
    result = []
    for session in sessions:
        session_dict = dict(session)
        
        try:
            # Parse messages JSON
            messages = json.loads(session_dict['messages'])
            # Print summary
            print(f"Session: {session_dict['name']}")
            print(f"Message count: {len(messages)}")
            print(f"First few messages: {messages[:2]}")  # Print first 2 messages for debugging
            print("---")
        except Exception as e:
            print(f"Error parsing messages for session {session_dict['name']}: {str(e)}")
        
        result.append(session_dict)
    
    return result

# Initialize the database tables
create_application_logs()
create_document_store()
create_sessions_table()
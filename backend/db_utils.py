import sqlite3
from datetime import datetime

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
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
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
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
    conn = get_db_connection()
    
    # Check if the table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_store'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create table with new columns
        conn.execute('''CREATE TABLE document_store
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT,
                         vector_db TEXT DEFAULT 'chromadb',
                         embedding_model TEXT DEFAULT 'openai',
                         upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename, vector_db, embedding_model) VALUES (?, ?, ?)', 
                  (filename, vector_db, embedding_model))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, vector_db, embedding_model, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

def get_documents_by_vector_db(vector_db):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, vector_db, embedding_model, upload_timestamp FROM document_store WHERE vector_db = ? ORDER BY upload_timestamp DESC', 
                  (vector_db,))
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

# Initialize the database tables
create_application_logs()
create_document_store()
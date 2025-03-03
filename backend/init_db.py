"""
Database Initialization Script

This script initializes the SQLite database and creates all necessary tables.
Run this script before starting the application to ensure the database is properly set up.
"""

import os
import sqlite3

# Define path to use the database in the root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(ROOT_DIR, "rag_app.db")

def init_database():
    """Initialize the database and create all necessary tables."""
    try:
        # Check if directory exists and is writable
        db_dir = os.path.dirname(DB_NAME)
        if not os.path.exists(db_dir):
            print(f"Creating directory {db_dir}")
            os.makedirs(db_dir)
        
        if not os.access(db_dir, os.W_OK):
            print(f"WARNING: Directory {db_dir} is not writable!")
        
        # Connect to database
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Create application_logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS application_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT,
            gpt_response TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create document_store table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            vector_db TEXT,
            embedding_model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create sessions table
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
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Database initialized successfully at {DB_NAME}")
        print("All required tables created.")
        
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    init_database() 
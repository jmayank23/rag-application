"""
Application Initialization Script

This script initializes the RAG application by setting up the database and required tables.
Run this script before starting the application to ensure everything is properly configured.
"""

import os
import sys

# Add the backend directory to the path using absolute path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(BACKEND_DIR)

def initialize_application():
    """Initialize the application by setting up the database and required tables."""
    try:
        # Import and run database initialization
        from init_db import init_database
        
        print("Initializing RAG Application...")
        
        # Initialize database
        db_success = init_database()
        if db_success:
            print("Database initialization successful!")
        else:
            print("Database initialization failed!")
            return False
        
        print("Application initialization complete. You can now start the application.")
        return True
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        return False

if __name__ == "__main__":
    initialize_application() 
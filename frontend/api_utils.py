import requests
import streamlit as st

class RagApiClient:
    """
    Reusable client for interacting with the RAG application's backend API.
    Implements standardized error handling and common request patterns.
    """
    
    def __init__(self, base_url="http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url (str): Base URL for the API
        """
        self.base_url = base_url
        self.json_headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def _handle_response(self, response, error_message="API request failed"):
        """
        Handle API response with standardized error handling.
        
        Args:
            response (requests.Response): Response object
            error_message (str): Custom error message prefix
            
        Returns:
            dict or None: Response JSON if successful, None otherwise
        """
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"{error_message}: {response.status_code} - {response.text}")
            return None
    
    def _make_request(self, method, endpoint, data=None, files=None, headers=None, error_message="API request failed"):
        """
        Make an API request with standardized error handling.
        
        Args:
            method (str): HTTP method (GET, POST, etc.)
            endpoint (str): API endpoint
            data (dict): Request data
            files (dict): Request files
            headers (dict): Request headers
            error_message (str): Custom error message prefix
            
        Returns:
            dict or None: Response JSON if successful, None otherwise
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers or self.json_headers)
            elif method.upper() == "POST":
                if files:
                    # Don't include Content-Type header with files
                    file_headers = {'accept': 'application/json'} if headers is None else headers
                    response = requests.post(url, files=files, data=data, headers=file_headers)
                else:
                    response = requests.post(url, json=data, headers=headers or self.json_headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            return self._handle_response(response, error_message)
        except Exception as e:
            st.error(f"{error_message}: {str(e)}")
            return None
    
    def chat(self, question, session_id=None, model="gpt-4o-mini", vector_db=None, embedding_model=None):
        """
        Send a chat request to the API.
        
        Args:
            question (str): User's question
            session_id (str, optional): Session ID
            model (str, optional): Model name
            vector_db (str, optional): Vector database type
            embedding_model (str, optional): Embedding model type
            
        Returns:
            dict or None: Response JSON if successful, None otherwise
        """
        data = {
            "question": question,
            "model": model
        }
        
        if session_id:
            data["session_id"] = session_id
        
        if vector_db:
            data["vector_db"] = vector_db
        
        if embedding_model:
            data["embedding_model"] = embedding_model
            
        return self._make_request(
            "POST", 
            "chat", 
            data=data, 
            error_message="Chat request failed"
        )
    
    def upload_document(self, file, vector_db=None, embedding_model=None):
        """
        Upload a document to the API.
        
        Args:
            file (UploadedFile): File to upload
            vector_db (str, optional): Vector database type
            embedding_model (str, optional): Embedding model type
            
        Returns:
            dict or None: Response JSON if successful, None otherwise
        """
        files = {"file": (file.name, file, file.type)}
        data = {}
        
        if vector_db:
            data["vector_db"] = vector_db
        
        if embedding_model:
            data["embedding_model"] = embedding_model
            
        return self._make_request(
            "POST", 
            "upload-doc", 
            data=data, 
            files=files, 
            error_message="Document upload failed"
        )
    
    def list_documents(self):
        """
        List all documents from the API.
        
        Returns:
            list or None: List of documents if successful, None otherwise
        """
        return self._make_request(
            "GET", 
            "list-docs", 
            error_message="Failed to fetch document list"
        )
    
    def delete_document(self, file_id, vector_db=None, embedding_model=None):
        """
        Delete a document from the API.
        
        Args:
            file_id (int): File ID
            vector_db (str, optional): Vector database type
            embedding_model (str, optional): Embedding model type
            
        Returns:
            dict or None: Response JSON if successful, None otherwise
        """
        data = {"file_id": file_id}
        
        if vector_db:
            data["vector_db"] = vector_db
        
        if embedding_model:
            data["embedding_model"] = embedding_model
            
        return self._make_request(
            "POST", 
            "delete-doc", 
            data=data, 
            error_message="Document deletion failed"
        )

# Create a global instance of the API client
api_client = RagApiClient()

# Wrapper functions to maintain backward compatibility
def get_api_response(question, session_id, model, vector_db=None, embedding_model=None):
    return api_client.chat(
        question=question,
        session_id=session_id,
        model=model,
        vector_db=vector_db,
        embedding_model=embedding_model
    )

def format_source_document(source):
    """
    Format a source document for display in the frontend
    
    Args:
        source: Source document dictionary from API response
        
    Returns:
        tuple: (display_name, content_preview, metadata_str)
    """
    # Default values
    display_name = source.get('filename', 'Unknown Source')
    content = source.get('content', 'No content available')
    metadata = source.get('metadata', {})
    
    # Create a preview of the content (first 150 chars)
    content_preview = content[:150] + '...' if len(content) > 150 else content
    
    # Format metadata as a string
    metadata_items = []
    for key, value in metadata.items():
        if key != 'source':  # Skip source as we already display the filename
            metadata_items.append(f"{key}: {value}")
    
    metadata_str = ", ".join(metadata_items)
    
    return display_name, content_preview, metadata_str

def generate_chat_title(messages, model="gpt-4o-mini"):
    """
    Generate a concise title for a chat session using the LLM.
    
    Args:
        messages: List of chat message dictionaries with 'role' and 'content'
        model: The LLM model to use
        
    Returns:
        A short, concise title for the chat session
    """
    if not messages:
        return "New Chat Session"
    
    # Create a prompt for title generation
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    
    if not user_messages:
        return "New Chat Session"
    
    # Use only the first 2-3 messages to keep the context manageable
    context = " | ".join(user_messages[:3])
    
    prompt = f"Please generate a very concise title (3-5 words max) that summarizes this conversation: '{context}'. " \
             f"Return only the title, no quotes or explanations."
    
    # Call the API using our global client
    result = api_client.chat(question=prompt, model=model)
    
    if result:
        title = result.get('answer', '').strip()
        
        # Ensure the title is not too long
        if len(title) > 40:
            title = title[:37] + "..."
            
        return title
    else:
        # Fallback to using the first user message
        return user_messages[0][:30] + ("..." if len(user_messages[0]) > 30 else "")

def upload_document(file, vector_db=None, embedding_model=None):
    return api_client.upload_document(
        file=file,
        vector_db=vector_db,
        embedding_model=embedding_model
    )

def list_documents():
    return api_client.list_documents() or []

def delete_document(file_id, vector_db=None, embedding_model=None):
    return api_client.delete_document(
        file_id=file_id,
        vector_db=vector_db,
        embedding_model=embedding_model
    )
import requests
import streamlit as st

def get_api_response(question, session_id, model, vector_db=None, embedding_model=None):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
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

    try:
        response = requests.post("http://localhost:8000/chat", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

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
    
    # Call the API
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "question": prompt,
        "model": model
    }
    
    try:
        response = requests.post("http://localhost:8000/chat", headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            title = result.get('answer', '').strip()
            
            # Ensure the title is not too long
            if len(title) > 40:
                title = title[:37] + "..."
                
            return title
        else:
            # Fallback to using the first user message
            return user_messages[0][:30] + ("..." if len(user_messages[0]) > 30 else "")
    except Exception as e:
        # Fallback to using the first user message
        return user_messages[0][:30] + ("..." if len(user_messages[0]) > 30 else "")

def upload_document(file, vector_db=None, embedding_model=None):
    print("Uploading file...")
    try:
        files = {"file": (file.name, file, file.type)}
        data = {}
        
        if vector_db:
            data["vector_db"] = vector_db
        
        if embedding_model:
            data["embedding_model"] = embedding_model
            
        response = requests.post("http://localhost:8000/upload-doc", files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to upload file. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while uploading the file: {str(e)}")
        return None

def list_documents():
    try:
        response = requests.get("http://localhost:8000/list-docs")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch document list. Error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"An error occurred while fetching the document list: {str(e)}")
        return []

def delete_document(file_id, vector_db=None, embedding_model=None):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {"file_id": file_id}
    
    if vector_db:
        data["vector_db"] = vector_db
    
    if embedding_model:
        data["embedding_model"] = embedding_model

    try:
        response = requests.post("http://localhost:8000/delete-doc", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to delete document. Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"An error occurred while deleting the document: {str(e)}")
        return None
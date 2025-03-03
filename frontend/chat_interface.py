import streamlit as st
from api_utils import get_api_response, get_streaming_api_response, format_source_document
import datetime
import json
import base64
import urllib.parse
import os
import sys
from pathlib import Path
import mimetypes
import logging
import time

# Add the backend directory to the path using absolute path
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.append(BACKEND_DIR)

def get_mime_type(filename):
    """Get the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        return mime_type
    return "application/octet-stream"

def display_file_content(file_path, file_name, metadata=None):
    """Display file content based on file type."""
    mime_type = get_mime_type(file_path)
    
    # Handle different file types
    if mime_type.startswith('text/'):
        # For text files, display content directly
        with open(file_path, 'r', errors='replace') as f:
            content = f.read()
        st.text_area("File Content", content, height=300)
    
    elif mime_type.startswith('image/'):
        # For images, display the image
        st.image(file_path, caption=file_name)
    
    elif mime_type == 'application/pdf':
        # Get the page number from metadata if available
        page_number = None
        if metadata and 'page' in metadata:
            try:
                page_number = int(metadata['page'])
            except (ValueError, TypeError):
                pass
        
        # Create URL to the backend endpoint
        vector_db = st.session_state.get("vector_db", "chromadb")
        base_name = os.path.basename(file_name)
        if base_name.startswith('temp_'):
            base_name = base_name[5:]  # Remove temp_ prefix for cleaner URLs
            
        file_url = f"http://localhost:8000/files/{vector_db}/{base_name}"
        if page_number:
            file_url += f"#page={page_number}"
        
        # Create a button-styled link that opens in a new tab
        st.markdown(
            f'''
            <a href="{file_url}" target="_blank" style="
                display: inline-block;
                padding: 0.5em 1em;
                color: white;
                background-color: #FF4B4B;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px 0;
                text-align: center;
                ">
                Open PDF in New Tab {f"(Page {page_number})" if page_number else ""}
            </a>
            ''',
            unsafe_allow_html=True
        )
    
    else:
        # For other file types, show a message
        st.info(f"Preview not available for this file type ({mime_type}). Use the download button below.")

def get_document_path(filename, vector_db=None):
    """
    Get the full path to a document file
    
    Args:
        filename: Name of the document file
        vector_db: Vector database where the file is stored
        
    Returns:
        Path object for the document or None if not found
    """
    if vector_db is None:
        vector_db = st.session_state.get("vector_db", "chromadb")
    
    # Handle the "temp_" prefix discrepancy
    filenames_to_try = [filename]
    
    # If filename has temp_ prefix, also try without it
    if filename.startswith("temp_"):
        filenames_to_try.append(filename[5:])  # Remove "temp_" prefix
    # If filename doesn't have temp_ prefix, also try with it
    else:
        filenames_to_try.append(f"temp_{filename}")
    
    # Get the absolute path to the frontend directory
    frontend_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(frontend_dir, "..", "backend"))
    
    # Try multiple possible locations for each filename variation
    for name in filenames_to_try:
        possible_paths = [
            # Backend directory paths (primary location)
            os.path.join(backend_dir, "uploads", vector_db, name),
            
            # Legacy paths for backward compatibility
            os.path.join(frontend_dir, "uploads", vector_db, name),
            os.path.join("uploads", vector_db, name),
            os.path.join("..", "uploads", vector_db, name),
        ]
        
        # Try each path
        for path in possible_paths:
            if os.path.exists(path):
                return path
    
    # If we get here, file wasn't found with any name variation
    return None

def display_chat_interface():
    # Chat header with session info and model selector
    col1, col2 = st.columns([2, 4])
    
    with col1:
        # LLM Model Selection dropdown in top left
        model_options = ["gpt-4o", "gpt-4o-mini"]
        current_model = st.session_state.get("model", "gpt-4o-mini")
        new_model = st.selectbox(
            "Model", 
            options=model_options,
            index=model_options.index(current_model),
            key="model_selector"
        )
        
        # Update model if changed
        if new_model != current_model:
            st.session_state.model = new_model
    
    with col2:
        # Display current session settings as badges
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px;">
                <span style="background-color: #1e2130; padding: 5px 10px; border-radius: 20px; font-size: 0.8em; color: #ffffff; border: 1px solid #4e5d95;">
                    Vector DB: {st.session_state.get("vector_db", "chromadb")}
                </span>
                <span style="background-color: #1e2130; padding: 5px 10px; border-radius: 20px; font-size: 0.8em; color: #ffffff; border: 1px solid #4e5d95;">
                    Embeddings: {st.session_state.get("embedding_model", "openai")}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Horizontal line to separate chat header from messages
    st.markdown("<hr style='margin-top: 0; margin-bottom: 20px;'>", unsafe_allow_html=True)
    
    # Create a container for chat messages with fixed height
    chat_container = st.container()
    
    # Chat input box - keep at bottom by placing after chat container
    # Check if st.chat_input is available (Streamlit >= 1.20.0)
    chat_input_available = hasattr(st, 'chat_input')
    
    # Use chat_input if available, otherwise use a regular text input
    if chat_input_available:
        prompt = st.chat_input("Ask a question about your documents...")
    else:
        prompt = st.text_input("Ask a question about your documents...", key="query_input")
        submit = st.button("Send")
        if not submit and not prompt:
            prompt = None
            
    # Chat history inside the container
    with chat_container:
        if not st.session_state.messages:
            # Welcome message for new chat
            st.markdown(
                f"""
                <div style="text-align: center; padding: 30px;">
                    <h2>Welcome to RAG Chatbot</h2>
                    <p>Using <strong>{st.session_state.get("vector_db", "chromadb").capitalize()}</strong> with 
                    <strong>{st.session_state.get("embedding_model", "openai")}</strong> embeddings</p>
                    <p>Upload documents in the sidebar and ask questions about them.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            # Display existing chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    # Enable proper markdown rendering including bullet points, code blocks, etc.
                    st.markdown(message["content"], unsafe_allow_html=True)
                    
                    # For assistant messages, display the Response Details if available
                    if message["role"] == "assistant" and "details" in message:
                        with st.expander("Response Details"):
                            # Document Sources section
                            st.subheader("Document Sources")
                            if message["details"].get("sources"):
                                # Display sources with content viewer
                                for i, source in enumerate(message["details"]["sources"]):
                                    display_name, content_preview, metadata = format_source_document(source)
                                    
                                    # Get source document path
                                    file_path = get_document_path(display_name)
                                    
                                    # Display source information card
                                    st.markdown(f"""
                                    <div style="margin-bottom: 10px; padding: 10px; border-radius: 5px; border: 1px solid #f0f0f0;">
                                        <strong>Source {i+1}:</strong> {display_name}
                                        <div style="margin-top: 5px;">
                                            <strong>Preview:</strong> <em>{content_preview[:100]}{'...' if len(content_preview) > 100 else ''}</em>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show content if file exists
                                    if file_path and os.path.exists(file_path):
                                        # Use tabs instead of expanders
                                        document_tabs = st.tabs(["Content", "Metadata"])
                                        
                                        with document_tabs[0]:  # Content tab
                                            # Display file content based on type
                                            display_file_content(file_path, display_name, source.get('metadata'))
                                        
                                        with document_tabs[1]:  # Metadata tab
                                            # Add metadata if available
                                            if source.get('metadata'):
                                                for key, value in source.get('metadata', {}).items():
                                                    if key != 'source':  # Skip source as we already display the filename
                                                        st.markdown(f"**{key}:** {value}")
                                            else:
                                                st.info("No metadata available for this document.")
                            else:
                                # If sources are not in the response, show a message
                                st.info("Sources information not available in this response.")
                            
                            # Model details
                            st.subheader("Model Used")
                            st.code(message["details"].get('model', st.session_state.model))
                            st.subheader("Vector Database")
                            st.code(message["details"].get('vector_db', st.session_state.vector_db))
                            st.subheader("Embedding Model")
                            st.code(message["details"].get('embedding_model', st.session_state.embedding_model))
                            st.subheader("Session ID")
                            st.code(message["details"].get('session_id'))

    # Process the user input if provided
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Clear the regular text input if using that method
        if not chat_input_available and 'query_input' in st.session_state:
            st.session_state.query_input = ""

        # Initialize variables to store response data
        final_response = {"answer": "", "session_id": st.session_state.session_id}
        sources = []
        
        # Create a placeholder for the assistant's message
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Get streaming response
            streaming_response = get_streaming_api_response(
                prompt,
                st.session_state.session_id,
                st.session_state.model,
                st.session_state.vector_db,
                st.session_state.embedding_model
            )
            
            if streaming_response:
                # Process the streaming response
                for line in streaming_response.iter_lines():
                    if line:
                        # Strip the "data: " prefix
                        line_text = line.decode('utf-8').strip()
                        if line_text.startswith('data: '):
                            data = line_text[6:]
                            
                            # Debug logging to check if newlines are in the data
                            if '\n' in data:
                                logging.info(f"Received data with newlines. Raw data (first 100 chars): {repr(data[:100])}")
                            
                            # Try to parse as JSON first
                            try:
                                parsed_data = json.loads(data)
                                
                                # Check if it's session info (has end and session_id keys)
                                if isinstance(parsed_data, dict) and 'end' in parsed_data and 'session_id' in parsed_data:
                                    final_response["session_id"] = parsed_data.get("session_id")
                                    # Skip adding this to the displayed response
                                    continue
                                    
                                # If it's not session info, it's the JSON-encoded text content
                                # Add it as-is to maintain newlines
                                chunk_text = parsed_data
                                full_response += chunk_text
                            except json.JSONDecodeError:
                                # Not JSON or failed to parse, treat as regular content
                                full_response += data
                                
                            # Use a container to style the streaming text to look like Markdown output
                            # but without actually parsing it as Markdown during streaming
                            response_placeholder.empty()
                            with response_placeholder.container():
                                # During streaming, ensure newlines are preserved by using HTML
                                # Replace newlines with <br> tags for proper display
                                formatted_stream = full_response.replace('\n', '<br>\n')
                                st.markdown(f"""
                                <div>
                                {formatted_stream}▌
                                </div>
                                """, unsafe_allow_html=True)
                
                # At the end, render the complete text with proper Markdown formatting
                response_placeholder.empty()
                
                # Ensure markdown is rendered properly with all newlines and formatting preserved
                # Use the streamlit markdown component with unsafe_allow_html=True
                # This ensures bullet points, code blocks, and other markdown elements render correctly
                response_placeholder.markdown(full_response, unsafe_allow_html=True)
                
                # Update session state
                st.session_state.session_id = final_response["session_id"]
                final_response["answer"] = full_response
                
                # Now get the full response with sources for the details
                with st.spinner("Retrieving source documents..."):
                    complete_response = get_api_response(
                        prompt, 
                        st.session_state.session_id, 
                        st.session_state.model,
                        st.session_state.vector_db,
                        st.session_state.embedding_model
                    )
                    
                    if complete_response and 'sources' in complete_response:
                        sources = complete_response.get('sources', [])
                        final_response.update({
                            "model": complete_response.get('model'),
                            "vector_db": complete_response.get('vector_db'),
                            "embedding_model": complete_response.get('embedding_model'),
                            "sources": sources
                        })
                
                # Save to message history with all details
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "details": {
                        "model": final_response.get('model', st.session_state.model),
                        "vector_db": final_response.get('vector_db', st.session_state.vector_db),
                        "embedding_model": final_response.get('embedding_model', st.session_state.embedding_model),
                        "session_id": final_response.get('session_id'),
                        "sources": final_response.get('sources', [])
                    }
                })
                
                # Display Response Details for the current response
                with st.expander("Response Details"):
                    # Add Document Sources section
                    st.subheader("Document Sources")
                    if sources:
                        # Display sources with content viewer
                        for i, source in enumerate(sources):
                            display_name, content_preview, metadata = format_source_document(source)
                            
                            # Get source document path
                            file_path = get_document_path(display_name)
                            
                            # Show content if file exists
                            if file_path and os.path.exists(file_path):
                                # Use tabs instead of expanders
                                document_tabs = st.tabs(["Content", "Metadata"])
                                
                                with document_tabs[0]:  # Content tab
                                    # Display file content based on type
                                    display_file_content(file_path, display_name, source.get('metadata'))
                                
                                with document_tabs[1]:  # Metadata tab
                                    # Add metadata if available
                                    if source.get('metadata'):
                                        for key, value in source.get('metadata', {}).items():
                                            if key != 'source':  # Skip source as we already display the filename
                                                st.markdown(f"**{key}:** {value}")
                                    else:
                                        st.info("No metadata available for this document.")
                    else:
                        # If sources are not in the response, show a message
                        st.info("Sources information not available in this response.")
                    
                    # Model details
                    st.subheader("Model Used")
                    st.code(final_response.get('model', st.session_state.model))
                    st.subheader("Vector Database")
                    st.code(final_response.get('vector_db', st.session_state.vector_db))
                    st.subheader("Embedding Model")
                    st.code(final_response.get('embedding_model', st.session_state.embedding_model))
                    st.subheader("Session ID")
                    st.code(final_response.get('session_id'))
                
                # Auto-save the session after each response
                try:
                    from sidebar import save_current_session
                    save_current_session()
                except Exception as e:
                    st.warning(f"Could not save session: {str(e)}")
                    print(f"Error saving session: {str(e)}")
            else:
                st.error("Failed to get a response from the API. Please try again.")

    # Apply consistent font to all markdown content for consistency
    st.markdown(
        """
        <style>
        /* Apply consistent font to all markdown content for consistency */
        .stMarkdown, .stMarkdown p, .stMarkdown ol, .stMarkdown ul, .stMarkdown li {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
        }
        
        /* Keep code blocks in monospace font as that's the standard expectation */
        .stMarkdown pre, .stMarkdown code {
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
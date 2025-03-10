from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic_models import (
    QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, 
    UploadDocumentRequest, VectorDBType, EmbeddingModelType, SourceDocument
)
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record, get_documents_by_vector_db
from vector_store_utils import index_document, delete_document
import os
import uuid
import logging
import shutil
import traceback
from typing import Optional, List, Dict
import mimetypes
import json

# Set up more detailed logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add console handler to the root logger
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)

app = FastAPI()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, "
                f"Model: {query_input.model.value}, Vector DB: {query_input.vector_db.value}, "
                f"Embedding Model: {query_input.embedding_model.value}")
    
    if not session_id:
        session_id = str(uuid.uuid4())

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(
        model=query_input.model.value,
        vector_db=query_input.vector_db.value,
        embedding_model=query_input.embedding_model.value
    )
    
    # Invoke the RAG chain and capture both the answer and context documents
    result = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })
    
    answer = result['answer']
    
    # Process the documents to include as sources
    sources = []
    if 'context' in result and result['context']:
        for doc in result['context']:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # Extract filename from metadata or use a default
                filename = doc.metadata.get('source', 'Unknown Source')
                if isinstance(filename, str) and '/' in filename:
                    # Extract just the filename from the path
                    filename = filename.split('/')[-1]
                
                sources.append(
                    SourceDocument(
                        filename=filename,
                        content=doc.page_content,
                        metadata=doc.metadata
                    )
                )
    
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    
    return QueryResponse(
        answer=answer, 
        session_id=session_id, 
        model=query_input.model,
        vector_db=query_input.vector_db,
        embedding_model=query_input.embedding_model,
        sources=sources
    )

@app.post("/chat/stream")
async def chat_stream(query_input: QueryInput):
    """Stream the response from the LLM as it's generated"""
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, "
                f"Model: {query_input.model.value}, Vector DB: {query_input.vector_db.value}, "
                f"Embedding Model: {query_input.embedding_model.value}")
    
    if not session_id:
        session_id = str(uuid.uuid4())

    chat_history = get_chat_history(session_id)
    
    # Get the RAG chain with streaming enabled
    rag_chain = get_rag_chain(
        model=query_input.model.value,
        vector_db=query_input.vector_db.value,
        embedding_model=query_input.embedding_model.value,
        streaming=True
    )
    
    # Process sources and metadata outside of the stream
    # We don't want to stream this information, just the answer text
    context_docs = []
    
    async def generate_stream():
        answer_text = ""
        
        # Invoke the RAG chain and stream the response
        async for chunk in rag_chain.astream({
            "input": query_input.question,
            "chat_history": chat_history
        }):
            if "answer" in chunk:
                answer_piece = chunk["answer"]
                # Preserve the raw text - no need to modify newlines here
                answer_text += answer_piece
                # JSON encode the answer to preserve newlines during transmission
                # This ensures newlines won't be lost in the SSE transmission
                encoded_piece = json.dumps(answer_piece)
                yield f"data: {encoded_piece}\n\n"
            if "context" in chunk and chunk["context"]:
                context_docs.extend(chunk["context"])
        
        # Return the session ID and end of stream marker
        session_info = {"session_id": session_id, "end": True}
        yield f"data: {json.dumps(session_info)}\n\n"
        
        # Save the complete response to the database
        insert_application_logs(session_id, query_input.question, answer_text, query_input.model.value)
        
    return StreamingResponse(generate_stream(), media_type="text/event-stream")

def get_upload_doc_params(
    vector_db: str = Form(VectorDBType.CHROMADB.value),
    embedding_model: str = Form(EmbeddingModelType.OPENAI.value),
) -> UploadDocumentRequest:
    return UploadDocumentRequest(
        vector_db=VectorDBType(vector_db),
        embedding_model=EmbeddingModelType(embedding_model)
    )

@app.post("/upload-doc")
def upload_and_index_document(
    file: UploadFile = File(...),
    params: UploadDocumentRequest = Depends(get_upload_doc_params)
):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    file_id = None
    
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Pass vector_db and embedding_model to insert_document_record
        file_id = insert_document_record(
            file.filename, 
            vector_db=params.vector_db.value,
            embedding_model=params.embedding_model.value
        )
        
        try:
            logging.info(f"Starting indexing of document '{file.filename}' with file_id {file_id}, "
                         f"vector_db: {params.vector_db.value}, embedding_model: {params.embedding_model.value}")
            
            # Use the unified index_document function from vector_store_utils
            success = index_document(
                temp_file_path, 
                file_id,
                vector_db=params.vector_db.value,
                embedding_model=params.embedding_model.value
            )
            
            if success:
                return {
                    "message": f"File {file.filename} has been successfully uploaded and indexed.",
                    "file_id": file_id,
                    "vector_db": params.vector_db.value,
                    "embedding_model": params.embedding_model.value
                }
            else:
                if file_id:
                    delete_document_record(file_id)
                raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
        except ValueError as e:
            logging.error(f"Error indexing document: {str(e)}")
            
            # Check if it's a Pinecone-specific error
            error_msg = str(e)
            detail = f"Failed to index {file.filename}. Error: {error_msg}"
            
            if "Pinecone" in error_msg:
                # Give more helpful guidance for Pinecone errors
                if "region" in error_msg or "cloud" in error_msg:
                    detail = f"Pinecone configuration error: {error_msg}. Please use a supported region for your Pinecone account."
                elif "dimension" in error_msg:
                    detail = f"Pinecone dimension error: {error_msg}. The embedding dimension must match your Pinecone index configuration."
            
            if file_id:
                delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=detail)
        except Exception as e:
            logging.error(f"Error indexing document: {str(e)}")
            if file_id:
                delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}. Error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents(vector_db: str = None):
    """
    List all documents or filter by vector database.
    
    Args:
        vector_db (str, optional): Filter by vector database type
        
    Returns:
        list[DocumentInfo]: List of document records
    """
    if vector_db:
        return get_documents_by_vector_db(vector_db)
    else:
        return get_all_documents()

@app.post("/delete-doc")
def delete_document_api(request: DeleteFileRequest):
    try:
        logging.info(f"Deleting document with file_id {request.file_id}, "
                     f"vector_db: {request.vector_db.value}, embedding_model: {request.embedding_model.value}")
        
        # Delete from Vector DB using the unified function
        vector_delete_success = delete_document(
            request.file_id,
            vector_db=request.vector_db.value,
            embedding_model=request.embedding_model.value
        )

        if vector_delete_success:
            # If successfully deleted from vector DB, delete from our database
            db_delete_success = delete_document_record(request.file_id)
            if db_delete_success:
                return {
                    "message": f"Successfully deleted document with file_id {request.file_id} from the system.",
                    "vector_db": request.vector_db.value,
                    "embedding_model": request.embedding_model.value
                }
            else:
                return {
                    "error": f"Deleted from {request.vector_db.value} but failed to delete document with file_id {request.file_id} from the database."
                }
        else:
            # For vector store failures, check if the error is recoverable
            # If the document doesn't exist in the vector store but is in the database, we can still delete from DB
            logging.warning(f"Document with file_id {request.file_id} not found in vector store "
                          f"or error occurred during deletion. Attempting to delete from database anyway.")
            
            db_delete_success = delete_document_record(request.file_id)
            if db_delete_success:
                return {
                    "message": f"Document with file_id {request.file_id} deleted from database, but was not found in {request.vector_db.value}.",
                    "warning": "The document may have already been removed from the vector store."
                }
            else:
                return {
                    "error": f"Failed to delete document with file_id {request.file_id} from both {request.vector_db.value} and the database."
                }
    except Exception as e:
        logging.error(f"Error in delete_document: {str(e)}")
        if "Pinecone" in str(e) and "configuration" in str(e):
            return {
                "error": f"Pinecone configuration error: {str(e)}",
                "suggestion": "Check your Pinecone API key and region settings."
            }
        else:
            return {
                "error": f"Error deleting document with file_id {request.file_id}: {str(e)}"
            }

@app.get("/files/{vector_db}/{filename}")
async def get_file(vector_db: str, filename: str):
    """Serve files from the uploads directory"""
    # Get the absolute path to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute path to the uploads directory
    uploads_dir = os.path.join(backend_dir, "uploads")
    # Construct the full file path
    file_path = os.path.join(uploads_dir, vector_db, filename)
    
    if not os.path.exists(file_path):
        # Try with temp_ prefix
        temp_file_path = os.path.join(uploads_dir, vector_db, f"temp_{filename}")
        if os.path.exists(temp_file_path):
            file_path = temp_file_path
        else:
            raise HTTPException(status_code=404, detail="File not found")
    
    # Get the media type
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    
    # For PDFs, set Content-Disposition to inline to suggest browser display
    headers = {}
    if media_type == "application/pdf":
        headers["Content-Disposition"] = "inline"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename,
        headers=headers
    )
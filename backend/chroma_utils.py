# This file is being deprecated in favor of vector_store_utils.py
# It remains only for backward compatibility

import logging
from vector_store_utils import (
    index_document as index_document_to_chroma,
    delete_document as delete_doc_from_chroma,
    get_vector_store as get_vectorstore
)

# Log a warning about the deprecation
logging.warning("chroma_utils.py is deprecated and will be removed in a future version. Use vector_store_utils.py instead.")

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os
from pydantic_models import VectorDBType, EmbeddingModelType
import traceback

# Default settings
default_vector_db = VectorDBType.CHROMADB
default_embedding_model = EmbeddingModelType.OPENAI

# Different text splitter configurations for different vector stores
def get_text_splitter(vector_db=None):
    """Get appropriate text splitter based on vector store type"""
    vector_db = vector_db or default_vector_db.value
    
    if vector_db == VectorDBType.PINECONE.value:
        # Smaller chunks for Pinecone to reduce token count per chunk
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    else:
        # Default larger chunks for Chroma
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

def get_vectorstore(vector_db=None, embedding_model=None):
    """
    Get the vector store with the specified settings
    
    Args:
        vector_db (str, optional): The vector DB to use. Defaults to the default value.
        embedding_model (str, optional): The embedding model to use. Defaults to the default value.
        
    Returns:
        VectorStore: The configured vector store
    """
    vector_db = vector_db or default_vector_db.value
    embedding_model = embedding_model or default_embedding_model.value
    return get_vectorstore(vector_db, embedding_model)

# Get default vector store
vectorstore = get_vectorstore()

def load_and_split_document(file_path: str, vector_db=None) -> List[Document]:
    """Load and split a document into chunks based on the vector store type."""
    try:
        # Select appropriate loader based on file extension
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.html'):
            loader = UnstructuredHTMLLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Load the document
        logging.info(f"Loading document from {file_path}")
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} pages/sections from document")
        
        # Get the appropriate text splitter
        text_splitter = get_text_splitter(vector_db)
        
        # Split the document
        splits = text_splitter.split_documents(documents)
        logging.info(f"Split document into {len(splits)} chunks")
        
        return splits
    except Exception as e:
        logging.error(f"Error loading and splitting document: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

def index_document_to_chroma(file_path: str, file_id: int, vector_db=None, embedding_model=None) -> bool:
    """
    Index a document to the vector store
    
    Args:
        file_path (str): Path to the document
        file_id (int): File ID for metadata
        vector_db (str, optional): Vector DB to use. Defaults to None.
        embedding_model (str, optional): Embedding model to use. Defaults to None.
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Get the vector store with the specified settings
        vector_store = get_vectorstore(vector_db, embedding_model)
        
        # Load and split the document - pass vector_db to use appropriate chunking
        splits = load_and_split_document(file_path, vector_db)
        
        # Convert file_id to string for consistent handling
        file_id_str = str(file_id)
        
        logging.info(f"Indexing document with file_id {file_id_str} to {vector_db} with {len(splits)} chunks")
        
        # Enhance metadata for each split
        for i, split in enumerate(splits):
            split.metadata['file_id'] = file_id_str
            split.metadata['filename'] = os.path.basename(file_path)
            split.metadata['chunk_id'] = i
            
            # Ensure metadata doesn't contain any non-string values
            for key in list(split.metadata.keys()):
                if not isinstance(split.metadata[key], str):
                    split.metadata[key] = str(split.metadata[key])
        
        # Add documents to the vector store
        return add_documents_to_vectorstore(vector_store, splits)
    except Exception as e:
        logging.error(f"Error indexing document: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def delete_doc_from_chroma(file_id: int, vector_db=None, embedding_model=None):
    """
    Delete a document from the vector store
    
    Args:
        file_id (int): File ID to delete
        vector_db (str, optional): Vector DB to use. Defaults to None.
        embedding_model (str, optional): Embedding model to use. Defaults to None.
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Get the vector store with the specified settings
        vector_store = get_vectorstore(vector_db, embedding_model)
        
        # Convert file_id to string for consistent handling
        file_id_str = str(file_id)
        logging.info(f"Deleting document with file_id {file_id_str} from {vector_db}")
        
        return delete_documents_from_vectorstore(vector_store, file_id_str)
    except Exception as e:
        logging.error(f"Error deleting document with file_id {file_id}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False
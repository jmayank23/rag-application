from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_pinecone import Pinecone
import os
import pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Literal, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import logging
import time
import traceback
from pydantic_models import VectorDBType, EmbeddingModelType

# Make sure environment variables are loaded
load_dotenv()

# Constants for embedding models and vector stores
OPENAI_EMBEDDING = "openai"
HUGGINGFACE_EMBEDDING = "huggingface"
CHROMA_DB = "chromadb"
PINECONE_DB = "pinecone"

# Directory to store model weights
MODEL_WEIGHTS_DIR = "model_weights"
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)

# Environment variables for Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Default settings - using values from pydantic models for consistency
DEFAULT_VECTOR_DB = VectorDBType.CHROMADB.value
DEFAULT_EMBEDDING_MODEL = EmbeddingModelType.OPENAI.value

def get_embedding_function(embedding_model: str):
    """
    Creates and returns an embedding function based on the specified model.
    
    Args:
        embedding_model (str): The embedding model to use (openai or huggingface)
        
    Returns:
        Embedding function that can be used with vector stores
    """
    if embedding_model == OPENAI_EMBEDDING:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    elif embedding_model == HUGGINGFACE_EMBEDDING:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=os.path.join(os.getcwd(), MODEL_WEIGHTS_DIR)
        )
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")

def get_text_splitter(vector_db: str = None):
    """
    Get appropriate text splitter based on vector store type
    
    Args:
        vector_db (str, optional): Vector database type to determine chunking strategy
        
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    vector_db = vector_db or DEFAULT_VECTOR_DB
    
    if vector_db == PINECONE_DB:
        # Smaller chunks for Pinecone to reduce token count per chunk
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    else:
        # Default larger chunks for Chroma
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

def get_vector_store(vector_db: str = None, embedding_model: str = None) -> VectorStore:
    """
    Creates and returns a vector store based on the specified database and embedding model.
    
    Args:
        vector_db (str, optional): The vector database to use (chromadb or pinecone)
        embedding_model (str, optional): The embedding model to use (openai or huggingface)
        
    Returns:
        A vector store that can be used for document storage and retrieval
    """
    vector_db = vector_db or DEFAULT_VECTOR_DB
    embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
    
    embedding_function = get_embedding_function(embedding_model)
    
    if vector_db.lower() == CHROMA_DB:
        return Chroma(
            persist_directory=f"./chroma_db_{embedding_model}", 
            embedding_function=embedding_function
        )
    elif vector_db.lower() == PINECONE_DB:
        # Get Pinecone API key from environment
        api_key = PINECONE_API_KEY
        
        if not api_key:
            error_msg = "Pinecone API key not found in environment variables"
            logging.error(error_msg)
            raise ValueError(error_msg + ". Please set PINECONE_API_KEY in your .env file.")
        
        if not PINECONE_ENVIRONMENT:
            logging.warning("PINECONE_ENVIRONMENT not found in environment variables, using default 'gcp-starter'")
        
        # Log the Pinecone configuration (without showing the actual API key)
        logging.info(f"Initializing Pinecone with environment: {PINECONE_ENVIRONMENT}")
        
        try:
            # Initialize Pinecone
            pc = pinecone.Pinecone(api_key=api_key)
            
            # Check if an index name was provided in the environment
            index_name = PINECONE_INDEX_NAME
            
            if not index_name:
                # Use a default index name
                index_name = "langchain-rag-index"
                logging.warning(f"PINECONE_INDEX_NAME not found in environment variables, using default '{index_name}'")
            
            # Check if index exists, and create if not
            try:
                # Get list of indexes
                indexes = pc.list_indexes()
                
                if not any(idx.name == index_name for idx in indexes):
                    logging.info(f"Creating Pinecone index '{index_name}'...")
                    
                    # Dimension is based on the embedding model
                    dimension = 1536 if embedding_model == OPENAI_EMBEDDING else 384
                    
                    # Create the index with the new API format
                    try:
                        from pinecone import ServerlessSpec
                        
                        # Create a ServerlessSpec object
                        spec = ServerlessSpec(
                            cloud="aws",
                            region=PINECONE_ENVIRONMENT or "us-east-1"  # Use environment variable or default
                        )
                        
                        # Create the index using the new API format
                        try:
                            pc.create_index(
                                name=index_name,
                                dimension=dimension,
                                metric="cosine",
                                spec=spec
                            )
                        except TypeError as e:
                            # Handle case where the API has changed or needs different parameters
                            if "missing 1 required positional argument" in str(e):
                                logging.error("Pinecone API has changed. You may need to update your Pinecone client.")
                                logging.error("Please check the Pinecone documentation for the latest API usage.")
                                logging.error("Try reinstalling packages with: pip install -U pinecone-client langchain-pinecone")
                                raise ValueError("Pinecone API compatibility issue. Please update your libraries.")
                            else:
                                raise
                                
                        # Wait for index to be ready
                        retries = 0
                        max_retries = 10
                        while retries < max_retries:
                            try:
                                indexes = pc.list_indexes()
                                if any(idx.name == index_name and idx.status.get('ready', False) for idx in indexes):
                                    logging.info(f"Pinecone index '{index_name}' is ready")
                                    break
                                logging.info("Waiting for Pinecone index to be ready...")
                                time.sleep(5)
                                retries += 1
                            except Exception as e:
                                logging.warning(f"Error checking index status: {str(e)}")
                                time.sleep(5)
                                retries += 1
                    except ImportError:
                        logging.error("Failed to import ServerlessSpec. You may need to update the pinecone-client package.")
                        raise ValueError("Incompatible Pinecone client version. Please update to the latest version.")
            except Exception as e:
                error_msg = f"Error creating or checking Pinecone index: {str(e)}"
                logging.error(error_msg)
                raise ValueError(error_msg)
                
            # Return the Pinecone vector store
            try:
                return Pinecone(
                    index=pc.Index(index_name),
                    embedding=embedding_function
                )
            except Exception as e:
                error_msg = f"Error initializing Pinecone vector store: {str(e)}"
                logging.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error initializing Pinecone: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)
    else:
        raise ValueError(f"Unsupported vector database: {vector_db}")

def load_and_split_document(file_path: str, vector_db: str = None) -> List[Document]:
    """
    Load and split a document into chunks based on the vector store type.
    
    Args:
        file_path (str): Path to the document file
        vector_db (str, optional): Vector database type to determine chunking strategy
        
    Returns:
        List[Document]: List of document chunks
    """
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

def index_document(file_path: str, file_id: Union[int, str], vector_db: str = None, embedding_model: str = None) -> bool:
    """
    Index a document to the vector store
    
    Args:
        file_path (str): Path to the document
        file_id (Union[int, str]): File ID for metadata
        vector_db (str, optional): Vector DB to use
        embedding_model (str, optional): Embedding model to use
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Get the vector store with the specified settings
        vector_store = get_vector_store(vector_db, embedding_model)
        
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

def delete_document(file_id: Union[int, str], vector_db: str = None, embedding_model: str = None) -> bool:
    """
    Delete a document from the vector store
    
    Args:
        file_id (Union[int, str]): File ID to delete
        vector_db (str, optional): Vector DB to use
        embedding_model (str, optional): Embedding model to use
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Get the vector store with the specified settings
        vector_store = get_vector_store(vector_db, embedding_model)
        
        # Convert file_id to string for consistent handling
        file_id_str = str(file_id)
        logging.info(f"Deleting document with file_id {file_id_str} from {vector_db}")
        
        return delete_documents_from_vectorstore(vector_store, file_id_str)
    except Exception as e:
        logging.error(f"Error deleting document with file_id {file_id}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def get_embedding_dimension(embedding_model: str) -> int:
    """
    Returns the dimension of the embeddings for the specified model.
    
    Args:
        embedding_model (str): The embedding model
        
    Returns:
        int: The embedding dimension
    """
    if embedding_model == OPENAI_EMBEDDING:
        return 1536  # text-embedding-3-small dimension
    elif embedding_model == HUGGINGFACE_EMBEDDING:
        return 384  # all-MiniLM-L6-v2 dimension
    else:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

def add_documents_to_vectorstore(
    vector_store: VectorStore, 
    documents: List[Document]
) -> bool:
    """
    Adds documents to the specified vector store.
    
    Args:
        vector_store: The vector store to add documents to
        documents: The documents to add
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        logging.info(f"Adding {len(documents)} document chunks to vector store type: {type(vector_store).__name__}")
        
        # Handle differently based on vector store type for better control
        if isinstance(vector_store, Chroma):
            logging.info("Using Chroma vector store for document storage")
            vector_store.add_documents(documents)
        elif isinstance(vector_store, Pinecone):
            logging.info("Using Pinecone vector store for document storage")
            
            # For Pinecone, let's add documents in smaller batches to prevent timeouts
            batch_size = 32  # Smaller batch size for Pinecone
            successful_chunks = 0
            failed_chunks = 0
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                logging.info(f"Processing Pinecone batch {batch_num}/{total_batches} with {len(batch)} documents")
                
                try:
                    # Check sample metadata to ensure it's properly formatted
                    if batch:
                        sample_metadata = batch[0].metadata
                        logging.info(f"Sample document metadata: {sample_metadata}")
                    
                    vector_store.add_documents(batch)
                    successful_chunks += len(batch)
                    logging.info(f"Successfully added batch {batch_num} ({len(batch)} chunks)")
                except Exception as batch_error:
                    failed_chunks += len(batch)
                    logging.error(f"Error adding batch {batch_num} to Pinecone: {batch_error}")
                    import traceback
                    logging.error(f"Batch error traceback: {traceback.format_exc()}")
                    # Continue with next batch rather than failing entire process
                    continue
            
            # Report summary of results
            if failed_chunks > 0:
                logging.warning(f"Added {successful_chunks} chunks successfully, but {failed_chunks} chunks failed")
                if successful_chunks == 0:
                    return False
            else:
                logging.info(f"Successfully added all {successful_chunks} document chunks to Pinecone")
        else:
            # Default handling for other vector stores
            logging.info(f"Using {type(vector_store).__name__} vector store")
            vector_store.add_documents(documents)
            
        logging.info(f"Successfully completed vector store document addition")
        return True
    except Exception as e:
        logging.error(f"Error adding documents to vector store: {str(e)}")
        # Provide more detailed error logging
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def delete_documents_from_vectorstore(
    vector_store: VectorStore, 
    file_id: str
) -> bool:
    """
    Deletes documents with the specified file_id from the vector store.
    
    Args:
        vector_store: The vector store to delete documents from
        file_id: The file ID to delete (as a string)
        
    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Different implementations for different vector stores
        if isinstance(vector_store, Chroma):
            # ChromaDB implementation
            docs = vector_store.get(where={"file_id": file_id})
            logging.info(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
            vector_store._collection.delete(where={"file_id": file_id})
        elif isinstance(vector_store, Pinecone):
            # Pinecone implementation - using the filter parameter
            vector_store.delete(filter={"file_id": file_id})
        else:
            logging.warning(f"Unsupported vector store type: {type(vector_store)}")
            return False
            
        logging.info(f"Deleted all documents with file_id {file_id}")
        return True
    except Exception as e:
        logging.error(f"Error deleting documents with file_id {file_id}: {str(e)}")
        return False 
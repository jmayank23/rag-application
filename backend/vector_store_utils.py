from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_pinecone import Pinecone
import os
import pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Literal
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
import logging
import time

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

def get_vector_store(vector_db: str, embedding_model: str) -> VectorStore:
    """
    Creates and returns a vector store based on the specified database and embedding model.
    
    Args:
        vector_db (str): The vector database to use (chromadb or pinecone)
        embedding_model (str): The embedding model to use (openai or huggingface)
        
    Returns:
        A vector store that can be used for document storage and retrieval
    """
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
            raise ValueError("Pinecone API key not found in environment variables")
        
        if not PINECONE_ENVIRONMENT:
            logging.warning("PINECONE_ENVIRONMENT not found in environment variables, using default 'gcp-starter'")
        
        # Log the Pinecone configuration (without showing the actual API key)
        logging.info(f"Initializing Pinecone with environment: {PINECONE_ENVIRONMENT}")
        
        # Initialize Pinecone with Pinecone v3 client
        pc = pinecone.Pinecone(
            api_key=api_key,
            environment=PINECONE_ENVIRONMENT  # Add the environment parameter
        )
        
        # Use a safe index name (lowercase, no special chars) or from environment variable
        index_name = PINECONE_INDEX_NAME or f"ragapp-{embedding_model.lower()}"
        index_name = index_name.lower()  # Ensure lowercase
        
        try:
            # Get the embedding dimension
            dimension = get_embedding_dimension(embedding_model)
            
            # Check if index exists
            indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in indexes:
                logging.info(f"Creating new Pinecone index: {index_name} with dimension {dimension}")
                try:
                    # Determine if we should use serverless or pod-based based on environment
                    is_serverless = PINECONE_ENVIRONMENT == "gcp-starter"
                    
                    if is_serverless:
                        # Create serverless index
                        pc.create_index(
                            name=index_name,
                            dimension=dimension,
                            metric="cosine",
                            spec=pinecone.ServerlessSpec(
                                cloud="aws",
                                region="us-east-1"
                            )
                        )
                    else:
                        # Create pod-based index with smallest pod size
                        pc.create_index(
                            name=index_name,
                            dimension=dimension,
                            metric="cosine",
                            spec=pinecone.PodSpec(
                                environment=PINECONE_ENVIRONMENT,
                                pod_type="p1.x1",  # Smallest pod size
                                pods=1
                            )
                        )
                    
                    logging.info(f"Created Pinecone index: {index_name}")
                    
                    # Wait for index to be ready
                    time.sleep(15)  # Give Pinecone more time to initialize the index
                except Exception as e:
                    logging.error(f"Error creating Pinecone index: {str(e)}")
                    raise ValueError(f"Failed to create Pinecone index: {str(e)}")
            
            # Get the index
            try:
                index = pc.Index(index_name)
                logging.info(f"Successfully connected to Pinecone index: {index_name}")
            except Exception as e:
                logging.error(f"Error connecting to Pinecone index: {str(e)}")
                raise ValueError(f"Failed to connect to Pinecone index: {str(e)}")
                
            # Create and return the vector store
            try:
                return Pinecone(index, embedding_function, text_key="text")
            except Exception as e:
                logging.error(f"Error creating Pinecone vector store: {str(e)}")
                raise ValueError(f"Failed to create Pinecone vector store: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error initializing Pinecone: {str(e)}")
            raise
    else:
        raise ValueError(f"Unsupported vector database: {vector_db}")

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
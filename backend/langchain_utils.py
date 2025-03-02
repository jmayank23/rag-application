from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import os
import logging
from chroma_utils import get_vectorstore
import traceback

# Custom retriever with error handling
class SafeRetriever(BaseRetriever):
    """Custom retriever that adds error handling to any other retriever."""
    
    def __init__(self, base_retriever):
        """Initialize with a base retriever."""
        super().__init__()
        # Store the retriever with a name that won't conflict with BaseRetriever attributes
        self._base_retriever = base_retriever
    
    def _get_relevant_documents(self, query, **kwargs):
        """Override to add error handling to document retrieval."""
        try:
            logging.info(f"Retrieving documents for query: {query[:50]}...")
            docs = self._base_retriever.get_relevant_documents(query, **kwargs)
            logging.info(f"Retrieved {len(docs)} documents")
            
            # Log document content for debugging
            for i, doc in enumerate(docs):
                logging.info(f"Document {i+1}: {doc.page_content[:100]}...")
                logging.info(f"Document {i+1} metadata: {doc.metadata}")
            
            return docs
        except Exception as e:
            logging.error(f"Error retrieving documents: {str(e)}")
            logging.error(f"Retrieval error traceback: {traceback.format_exc()}")
            # Return empty list rather than failing
            return []

# Set up prompts and chains
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain the information needed, say so and answer based on your knowledge."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Prompt for when vector DB is unavailable
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. The retrieval system is currently unavailable, so you'll answer based on your general knowledge."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="gpt-4o-mini", vector_db=None, embedding_model=None):
    """
    Creates a RAG chain with the specified model, vector database, and embedding model.
    
    Args:
        model (str): The LLM model to use for responding
        vector_db (str, optional): The vector database to use. Defaults to None.
        embedding_model (str, optional): The embedding model to use. Defaults to None.
        
    Returns:
        Chain: A retrieval chain that can be used to answer questions
    """
    try:
        # Get vector store with the specified params
        logging.info(f"Initializing vector store with {vector_db} and {embedding_model}")
        
        # Create simple LLM for fallback case
        llm = ChatOpenAI(model=model)
        
        try:
            vectorstore = get_vectorstore(vector_db, embedding_model)
            logging.info(f"Successfully obtained vector store of type: {type(vectorstore).__name__}")
            
            # Set up retriever with error handling
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            logging.info(f"Created base retriever of type: {type(base_retriever).__name__}")
            
            retriever = SafeRetriever(base_retriever)
            logging.info(f"Created SafeRetriever wrapper for document retrieval")
            
            # Create the contextualize question chain
            contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
            
            def contextualized_question(input_dict):
                if input_dict.get("chat_history"):
                    return contextualize_q_chain.invoke({
                        "input": input_dict.get("input"),
                        "chat_history": input_dict.get("chat_history")
                    })
                return input_dict.get("input")
            
            # Create the RAG chain with the new API
            rag_chain = (
                {
                    "context": lambda input_dict: retriever.get_relevant_documents(
                        contextualized_question(input_dict)
                    ),
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x.get("chat_history", [])
                }
                | RunnablePassthrough.assign(
                    answer=qa_prompt | llm | StrOutputParser()
                )
            )
            
            return rag_chain
            
        except ValueError as e:
            # Handle specific Pinecone errors and provide helpful guidance
            error_msg = str(e)
            if "Pinecone" in error_msg:
                logging.error(f"Pinecone configuration error: {error_msg}")
                if "free tier" in error_msg or "region" in error_msg:
                    logging.warning("Using fallback LLM chain due to Pinecone region configuration issue")
                else:
                    logging.warning("Using fallback LLM chain due to Pinecone configuration issue")
            else:
                logging.error(f"Vector store error: {error_msg}")
                logging.warning("Using fallback LLM chain due to vector store error")
            
            # Fall back to a simple LLM chain without retrieval
            return create_fallback_chain(llm)
        
    except Exception as e:
        logging.error(f"Error creating RAG chain: {str(e)}")
        logging.error(f"RAG chain error traceback: {traceback.format_exc()}")
        
        # If we can't create the chain with the specified params, fall back to a simple LLM chain
        logging.info("Falling back to simple LLM without retrieval")
        
        # Create a fallback LLM
        try:
            llm = ChatOpenAI(model=model)
            return create_fallback_chain(llm)
        except Exception as llm_error:
            logging.error(f"Error creating fallback LLM: {str(llm_error)}")
            # Ultimate fallback - just return a function that provides a fixed response
            def ultimate_fallback(inputs):
                return {
                    "answer": "I'm sorry, I'm having trouble connecting to knowledge sources right now. Please try again later or ask a different question."
                }
            return ultimate_fallback

def create_fallback_chain(llm):
    """Create a fallback chain that doesn't use retrieval"""
    chain = fallback_prompt | llm | StrOutputParser()
    
    # Create a chain that returns the answer in the expected format
    return (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | RunnablePassthrough.assign(
            answer=fallback_prompt | llm | StrOutputParser()
        )
    )
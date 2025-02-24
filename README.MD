# Langchain RAG Chatbot

A full-stack Retrieval-Augmented Generation (RAG) chatbot with FastAPI (backend) and Streamlit (frontend). It supports multiple chats, document uploads (with deletion), and lets users choose an LLM for answering questions based on uploaded documents.

## Features
- **Chat Interface**: Simple and interactive.  
- **Document Handling**: Upload, view, and delete PDFs, DOCX, and HTML files.  
- **Session Management**: Supports multiple users and chats via session IDs.  
- **Model Selection**: Choose between different GPT models.  

## Tech Stack
- **Backend:**
   - [FastAPI](https://fastapi.tiangolo.com/): For building the RESTful API.
   - [LangChain](https://www.langchain.com/): For implementing the RAG pipeline, including document processing, retrieval, and response generation.
   - [OpenAI API](https://openai.com/api/): Used as the Large Language Model (LLM) for question answering and contextualization.
   - [Sentence Transformers](https://www.sbert.net/): Used for generating document embeddings (alternative to OpenAI embeddings).
- **Frontend:**
   - [Streamlit](https://streamlit.io/):  For creating the interactive web user interface.
- **Database:**
   - [SQLite](https://www.sqlite.org/index.html):  For storing conversation history (session data) and document metadata. _(Easily adaptable to other SQL databases like PostgreSQL or MySQL.)_
   - [ChromaDB](https://www.trychroma.com/):  Vector database for storing document embeddings and enabling efficient semantic search.
- **Debugging/Tracing:**
   - [LangSmith](https://smith.langchain.com/): Used to debug and trace the LangChain components of the application

## Setup and Installation

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### Steps to Run the Application

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and Activate Conda Environment**

   ```bash
   conda create --name rag-application python=3.10
   conda activate rag-application
   ```

3. **Install Dependencies**

   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Set Up Environment Variables**

   Create a `.env` file in the project directory (`rag-application`) with the following content:

   ```plaintext
   OPENAI_API_KEY="your-openai-api-key"
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY="your-langchain-api-key"
   LANGCHAIN_PROJECT="rag-application"
   ```

5. **Run the Backend Server**

   Navigate to the `backend` directory and start the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```

6. **Run the Frontend Application**

   In a new terminal, navigate to the `frontend` directory and start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

7. **Access the Application**

   Open your web browser and go to `http://localhost:8501` to interact with the chatbot.

## Usage

- **Upload Documents**: Use the sidebar to upload documents for indexing.
- **Chat**: Enter your queries in the chat input and receive responses from the selected GPT model.
- **Manage Documents**: View and delete documents from the sidebar.

## Logging

- Application logs are stored in `backend/app.log` for debugging and monitoring purposes.
- Log in to [LangSmith](https://smith.langchain.com) to view traces.

## Reference

- [Langchain RAG Course](https://github.com/PradipNichite/Youtube-Tutorials/tree/main/Langchain%20RAG%20Course%202024): Used for starter code. Built on top by adding more features.
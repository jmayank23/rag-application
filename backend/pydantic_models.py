from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any

class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class VectorDBType(str, Enum):
    CHROMADB = "chromadb"
    PINECONE = "pinecone"

class EmbeddingModelType(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)
    vector_db: VectorDBType = Field(default=VectorDBType.CHROMADB)
    embedding_model: EmbeddingModelType = Field(default=EmbeddingModelType.OPENAI)

class SourceDocument(BaseModel):
    """Model for document source information"""
    filename: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    vector_db: VectorDBType
    embedding_model: EmbeddingModelType
    sources: Optional[List[SourceDocument]] = None

class DocumentInfo(BaseModel):
    id: int
    filename: str
    vector_db: str = Field(default="chromadb")
    embedding_model: str = Field(default="openai")
    created_at: datetime

class DeleteFileRequest(BaseModel):
    file_id: int
    vector_db: VectorDBType = Field(default=VectorDBType.CHROMADB)
    embedding_model: EmbeddingModelType = Field(default=EmbeddingModelType.OPENAI)

class UploadDocumentRequest(BaseModel):
    vector_db: VectorDBType = Field(default=VectorDBType.CHROMADB)
    embedding_model: EmbeddingModelType = Field(default=EmbeddingModelType.OPENAI)
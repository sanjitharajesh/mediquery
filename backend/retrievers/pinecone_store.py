from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "mediquery"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class PineconeRetriever:
    def __init__(self, index, embeddings, top_k: int = 5):
        self.index = index
        self.embeddings = embeddings
        self.top_k = top_k

    def invoke(self, query: str):
        vector = self.embeddings.embed_query(query)
        response = self.index.query(
            vector=vector,
            top_k=self.top_k,
            include_metadata=True,
        )
        matches = response.get("matches", []) if isinstance(response, dict) else response.matches

        docs = []
        for match in matches:
            metadata = dict(match.get("metadata", {}) if isinstance(match, dict) else match.metadata or {})
            text = (
                metadata.pop("text", None)
                or metadata.pop("page_content", None)
                or metadata.pop("content", None)
                or ""
            )
            if text:
                docs.append(Document(page_content=text, metadata=metadata))
        return docs

def get_pinecone_retriever():
    """Get Pinecone retriever"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    return PineconeRetriever(index=index, embeddings=embeddings, top_k=5)

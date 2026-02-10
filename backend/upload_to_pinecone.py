"""
One-time script to upload existing ChromaDB data to Pinecone
"""
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "mediquery"
CHROMA_DIR = "../chroma_db"  # Adjust path if needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def upload_to_pinecone():
    print("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Loading ChromaDB...")
    chroma_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    # Get all documents from ChromaDB
    print("Fetching documents from ChromaDB...")
    docs = chroma_store.get()
    documents = [doc for doc in chroma_store.similarity_search("", k=10000)]
    print(f"Found {len(documents)} documents")
    
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    print("Uploading to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    print("✅ Upload complete!")

if __name__ == "__main__":
    upload_to_pinecone()
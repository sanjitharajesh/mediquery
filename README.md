# MediQuery

AI-powered FDA drug information assistant designed to curb medical hallucination by utilizing only evidence-based official drug labels. Built with FastAPI and RAG to provide accurate, source-backed medication information.

## Features

- **Evidence-Based Answers**: Responses sourced exclusively from official FDA drug labels
- **Common Questions**: Quick-access buttons for frequently asked queries
- **Personalized Context**: Optional user information (age, conditions, medications) for tailored responses
- **Source Attribution**: All answers include references to FDA documentation

## Tech Stack

**Backend:**
- FastAPI
- LangChain
- Pinecone (Vector Database)
- Groq LLM API
- HuggingFace Embeddings

**Frontend:**
- HTML/CSS/JavaScript

**Deployment:**
- Render (Backend hosting)

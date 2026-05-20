"""
FastAPI backend for MediQuery
Reuses existing RAG chain from backend/rag/chain.py
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.chain import get_rag_chain

# Initialize FastAPI app
app = FastAPI(title="MediQuery API", version="1.0.0")

# Enable CORS for localhost development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load RAG chain once at startup
print("Loading RAG chain...")
chain = get_rag_chain()
print("RAG chain loaded successfully!")

# Request/Response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

class EvalAnswerResponse(BaseModel):
    answer: str
    retrieved_contexts: list[str]

# API endpoints
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Process a question using the RAG chain and return the answer
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Invoke the existing RAG chain
        answer = chain.invoke(request.question.strip(), verbose=False)

        # Apply the same formatting as Streamlit version
        answer = answer.replace("WARNING:", "## WARNING\n")
        answer = answer.replace("PRECAUTIONS:", "## PRECAUTIONS\n")
        answer = answer.replace("POTENTIAL ADVERSE EFFECTS:", "## POTENTIAL SIDE EFFECTS\n")

        # Add source and disclaimer
        answer += "\n\n---\n\n**Source:** FDA-approved drug labels and prescribing information"
        answer += "\n\n**Medical Disclaimer:** This information is for educational purposes only. Always consult your healthcare provider before starting, stopping, or changing any medication."

        return AnswerResponse(answer=answer)

    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/eval-ask", response_model=EvalAnswerResponse)
async def eval_ask_question(request: QuestionRequest):
    """
    Evaluation-only endpoint that returns the raw answer plus retrieved contexts.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = chain.invoke(request.question.strip(), verbose=False)
        return EvalAnswerResponse(
            answer=answer,
            retrieved_contexts=chain._last_contexts,
        )

    except Exception as e:
        print(f"Error processing eval question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing eval question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MediQuery API"}

# Serve frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the main HTML page"""
    return FileResponse("../frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# backend/llm.py
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"  # 2x faster, slightly lower quality

def generate_answer(prompt: str, verbose: bool = False) -> str:
    """
    Call Ollama with streaming enabled for faster feedback.
    Reduced context window and token limits for speed.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": 200,      # Max output tokens (SHORT answers)
            "num_ctx": 1024,         # Reduced context window
            "temperature": 0.1,
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            stream=True, 
            timeout=60  # Reduced from 120
        )
        response.raise_for_status()
        
        answer = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                answer += token
                
                if verbose:
                    print(token, end="", flush=True)
                
                if chunk.get("done"):
                    break
        
        if verbose:
            print()  # newline
        
        return answer.strip()
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Try reducing context or using a smaller model."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

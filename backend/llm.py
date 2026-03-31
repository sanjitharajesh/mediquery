# backend/llm.py
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configuration
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

# Populated after each Groq call; read by chain.py for Langfuse metadata.
_last_token_usage: dict = {}
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"

def generate_answer(prompt: str, verbose: bool = False) -> str:
    
    if USE_GROQ:
        return _generate_groq(prompt, verbose)
    else:
        return _generate_ollama(prompt, verbose)


def _generate_groq(prompt: str, verbose: bool = False) -> str:
    """Call Groq API (fast cloud inference)"""
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY not set. Get one from https://console.groq.com/"
        
        if verbose:
            print("[Using Groq API - Cloud]")
        
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,  # Slightly increased for better extraction
            max_tokens=800,  # Increased for comprehensive extraction
            top_p=0.9,
        )
        
        answer = response.choices[0].message.content.strip()

        global _last_token_usage
        _last_token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        if verbose:
            print(f"[Groq: {len(answer)} chars received]")

        return answer
        
    except ImportError:
        return "Error: Run 'pip install groq' first"
    except Exception as e:
        if verbose:
            print(f"[Groq Error: {str(e)}]")
        # Fallback to Ollama if Groq fails
        print("Groq failed, falling back to Ollama...")
        return _generate_ollama(prompt, verbose)


def _generate_ollama(prompt: str, verbose: bool = False) -> str:
    """Call Ollama (local inference)"""
    if verbose:
        print("[Using Ollama - Local]")
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": 300,
            "num_ctx": 1500,
            "temperature": 0.2,  # Slightly increased for better extraction
            "top_p": 0.9,
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL, 
            json=payload, 
            stream=True, 
            timeout=60
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
            print()
        
        return answer.strip()
    
    except requests.exceptions.Timeout:
        return "Error: Request timed out."
    except Exception as e:
        return f"Error: {str(e)}"
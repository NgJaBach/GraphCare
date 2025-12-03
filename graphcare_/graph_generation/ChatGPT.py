"""
LLM API Interface for EMERGE - Ollama Integration

This module provides a clean interface to Ollama API for:
- Named Entity Recognition (NER) from clinical notes
- Summary generation from EHR + notes + KG context
"""

from dotenv import load_dotenv
import os
import json
import re
import requests
import time
from typing import List, Optional

# Load environment variables
load_dotenv()
OLLAMA_BASE_URL = os.getenv("BAILAB_HTTP")

# Ensure base URL format
if not OLLAMA_BASE_URL.startswith("http"):
    OLLAMA_BASE_URL = f"http://{OLLAMA_BASE_URL}"
OLLAMA_BASE_URL = OLLAMA_BASE_URL.rstrip("/")

# API endpoints
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

class OllamaClient:
    """Clean Ollama API client with proper error handling"""
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 180):
        self.base_url = base_url
        self.generate_url = f"{base_url}"
        self.timeout = timeout
    
    def generate(
        self,
        prompt: str,
        model: str = "gpt-oss:20b",
        thinking: bool = False,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        system: Optional[str] = None,
    ) -> str:
        """
        Generate text using Ollama API
        
        Args:
            prompt: User prompt
            model: Model name (e.g., "qwen:7b", "deepseek-v2:16b")
            thinking: Whether to enable "thinking" mode
            temperature: Sampling temperature (0.0 = deterministic)
            num_ctx: Context window size
            system: Optional system prompt
            
        Returns:
            Generated text
            
        Raises:
            requests.exceptions.RequestException: On API failure
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": thinking,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_gpu": -1,
                # "main_gpu": 1
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama API timeout after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Ollama API error: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from Ollama")

# Global client instance
ollama_client = OllamaClient()

def remove_reasoning(response_content: str) -> str:
    """Remove reasoning part if present"""
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response_content.strip()

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = "gpt-oss:20b",
    thinking: bool = False,
    max_tokens: int = 8192,
    temperature: float = 0.3,
    infinite_retry: bool = False,
) -> str:
    """
    Wrapper for backward compatibility with original EMERGE code
    
    Args:
        user_prompt: User prompt
        sys_prompt: System prompt (optional)
        model_name: Model name
        thinking: Whether to enable "thinking" mode
        max_tokens: Context window size
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    
    if infinite_retry:
        while True:
            try:
                response = ollama_client.generate(
                    prompt=user_prompt,
                    model=model_name,
                    thinking=thinking,
                    temperature=temperature,
                    num_ctx=max_tokens,
                    system=sys_prompt if sys_prompt else None,
                )
                return remove_reasoning(response)
            except Exception as e:
                continue

    response = ollama_client.generate(
        prompt=user_prompt,
        model=model_name,
        thinking=thinking,
        temperature=temperature,
        num_ctx=max_tokens,
        system=sys_prompt if sys_prompt else None,
    )
    return remove_reasoning(response)

class ChatGPT:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        self.messages = []
    
    def chat(self,
        message: str,
        sys_prompt: str = "",
        model_name: str = "gpt-oss:20b",
        thinking: bool = "low",
        max_tokens: int = 8192,
        temperature: float = 0.3,
        infinite_retry: bool = True,
    ) -> str:
        """
        Wrapper for backward compatibility with original EMERGE code
        
        Args:
            user_prompt: User prompt
            sys_prompt: System prompt (optional)
            model_name: Model name
            thinking: Whether to enable "thinking" mode
            max_tokens: Context window size
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        return ask(
            user_prompt=message,
            model_name=model_name,
            thinking=thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            sys_prompt=sys_prompt if sys_prompt else None,
            infinite_retry=infinite_retry,
        )
    
if __name__ == "__main__":
    print("Testing ChatGPT wrapper with Ollama API...")
    chatgpt = ChatGPT()
    response = chatgpt.chat("Hello, how are you?")
    print(response)
import os
import base64
from typing import Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file, overriding any system env vars
load_dotenv(override=True)

# We will use OpenRouter for Qwen Vision models
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# The user requested Qwen 3VL 32B (matching closest available on OpenRouter, usually qwen/qwen-2.5-vl-72b-instruct or similar)
# We can make this configurable via env var
MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", "qwen/qwen3-vl-32b-instruct")

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

def encode_image(image_path: str) -> str:
    """Encodes an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vision_llm(prompt: str, image_path: str, response_format: str = "json_object") -> str:
    """
    Calls the Qwen Vision model via OpenRouter.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in the .env file.")

    base64_image = encode_image(image_path)
    
    # OpenRouter requires specific headers for analytics, but they are optional.
    # We use standard OpenAI chat completions format for vision.
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    # Note: OpenRouter / Qwen supports JSON mode by including 'response_format' if supported,
    # or just prompting it to return JSON.
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1, # Low temp for deterministic extraction
        "max_tokens": 4096,
    }
    
    # If the model strictly supports response_format
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    
    return response.choices[0].message.content

def call_text_llm(prompt: str, response_format: str = "json_object") -> str:
    """
    Calls a text-only LLM if needed for non-vision tasks.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in the .env file.")

    messages = [{"role": "user", "content": prompt}]
    
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
    }
    
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    
    return response.choices[0].message.content

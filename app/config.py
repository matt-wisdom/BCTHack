import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini")
    MODEL_NAME = os.getenv("MODEL_NAME")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "300"))

config = Config()

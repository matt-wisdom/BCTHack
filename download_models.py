import os
import sys
# Add current directory to path so we can import app
sys.path.append(os.getcwd())

from app.memory import memory_manager
import time

def download():
    print("Starting full system pre-download...")
    print("Downloading embeddings and initializing ChromaDB (Size: ~80MB).")
    
    start_time = time.time()
    try:
        # This will trigger EVERYTHING: HF download and Chroma collection setup
        memory_manager.initialize()
        
        duration = time.time() - start_time
        print(f"Success! All models and databases are ready in {duration:.2f} seconds.")
        print("Tip: You can now start the server with: uv run uvicorn app.main:app")
    except Exception as e:
        print(f"Error during initialization: {e}")
        print("Tip: Ensure your internet is connected and try again.")

if __name__ == "__main__":
    download()

from fastapi import FastAPI
from app.routers import reviews, recommendations
from app.memory import memory_manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-load embeddings and ChromaDB
    print("Starting eager initialization...")
    memory_manager.initialize()
    yield
    # Shutdown: Clean up if needed
    pass

app = FastAPI(title="DSN Rec Agent API", lifespan=lifespan)

app.include_router(reviews.router)
app.include_router(recommendations.router)

@app.get("/")
async def root():
    return {"message": "DSN Rec Agent API is running"}

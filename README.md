# RecAgent Framework

RecAgent is a containerized FastAPI application implementing a cognitive-neuroscience-inspired consumer agent framework for the BCT Hackathon. It uses LangGraph for reasoning, ChromaDB for tri-layer memory, and web search for real-time economic context.

## Core Features
- **Tri-Layer Memory Module**:
  - **Sensory Memory**: Rapidly processes raw environmental and product data.
  - **Short-term Memory**: Manages session context and recurring theme identification.
  - **Long-term Memory (ChromaDB)**: Persists abstract insights and past behaviors with a mathematical forgetting mechanism.
- **Product Discovery Modes**:
  - **Offline Mode**: Searches a local vector database of products (no internet).
  - **Online Mode**: Uses real-time web search with neutral queries, manually filtered by the LLM.
- **Multi-Source Catalog**: Supports multiple datasets (Amazon, Jumia, Local Store) with the ability to filter per request.
- **Dynamic Reasoning Loop**:
  - Fetches real-time country-specific economic indicators (Inflation, PPI, Exchange Rates).
  - LLM-based reasoning engine that weighs personal income against localized economic reality.
- **Multi-Task Endpoints**:
  - **Reviews**: Generates localized Nigerian reviews and ratings for specific products.
  - **Recommendations**: Generates personalized product recommendations based on persona interests.

## Tech Stack
- **LangChain / LangGraph**: Orchestration and state management.
- **FastAPI**: RESTful API layer.
- **ChromaDB**: Vector database for memory and product catalog.
- **HuggingFace Local Embeddings**: `all-MiniLM-L6-v2` for semantic similarity.
- **Gemini / GPT-4o**: Large Language Model for complex reasoning.
- **DuckDuckGo Search**: Real-time web data fetching (Online mode).

## Getting Started

### Prerequisites
- Docker and Docker Compose.
- Google Gemini API Key (or an Ollama instance).

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd dsn-rec-agent
   ```

2. **Configure Environment Variables**:
   Create a `.env` file based on `.env.example`.

3. **Pre-download Models & Initialize DB**:
   ```bash
   uv run python download_models.py
   ```

4. **Seed Product Catalog**:
   Place your CSV files in `dsn-rec-agent/dataset/` and run:
   ```bash
   uv run python seed_catalog.py
   ```

5. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

## API Usage

### 1. Simulate a Product Review
`POST /reviews/simulate`

**Payload**:
```json
{
  "persona_text": "A 28-year-old tech enthusiast from Lagos...",
  "product_text": "Samsung Galaxy S24 Ultra...",
  "mode": "offline",
  "dataset_source": "store" 
}
```

### 2. Generate Recommendations
`POST /recommendations/generate`

**Payload**:
```json
{
  "persona_text": "A price-conscious student who loves photography...",
  "mode": "offline",
  "dataset_source": "all"
}
```

## Implementation Details
- **Forgetting Mechanism**: Implemented in `app/memory.py` using a power function to balance recency and importance.
- **Dataset Sources**: Options include `all`, `amazon`, `jumia`, and `store`.

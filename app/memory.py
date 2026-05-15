import os
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime, timedelta
import math
import json
from typing import List, Dict, Any, Optional
from app.models import SensoryObservation, EconomicContext
from app.config import config

class LangChainEmbeddingAdapter(EmbeddingFunction):
    """Adapts LangChain embeddings for use in ChromaDB."""
    def __init__(self, ef):
        self.ef = ef
    def __call__(self, input: Documents) -> Embeddings:
        return self.ef.embed_documents(input)

class MemoryManager:
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.persist_directory = persist_directory
        self._client = None
        self._memory_collection = None
        self._product_collection = None
        self._economic_collection = None
        self._embeddings = None

    def initialize(self):
        """Eagerly initialize embeddings and ChromaDB client. Raises exception on failure."""
        if self._embeddings is None:
            print("Starting Loading HuggingFace Embeddings (all-MiniLM-L6-v2)...")
            self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("Embeddings loaded.")
        
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Long-term memory collection
            self._memory_collection = self._client.get_or_create_collection(
                name="long_term_memory",
                embedding_function=LangChainEmbeddingAdapter(self._embeddings),
                metadata={"hnsw:space": "cosine"}
            )
            
            # Product catalog collection
            self._product_collection = self._client.get_or_create_collection(
                name="product_catalog",
                embedding_function=LangChainEmbeddingAdapter(self._embeddings),
                metadata={"hnsw:space": "cosine"}
            )

            # Economic context cache collection
            self._economic_collection = self._client.get_or_create_collection(
                name="economic_cache",
                embedding_function=LangChainEmbeddingAdapter(self._embeddings),
                metadata={"hnsw:space": "cosine"}
            )
            print("ChromaDB initialized with Shared Embeddings and three collections.")

    @property
    def memory_collection(self):
        if self._memory_collection is None:
            self.initialize()
        return self._memory_collection

    @property
    def product_collection(self):
        if self._product_collection is None:
            self.initialize()
        return self._product_collection

    @property
    def economic_collection(self):
        if self._economic_collection is None:
            self.initialize()
        return self._economic_collection

    def add_to_long_term(self, content: str, metadata: Dict[str, Any]):
        self.memory_collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[f"mem_{datetime.now().timestamp()}"]
        )

    def retrieve_relevant(self, query: str, n_results: int = 5) -> List[str]:
        results = self.memory_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

    def retrieve_products(self, query: str, n_results: int = 5, dataset_source: str = "all") -> List[str]:
        """Query the product catalog collection with optional source filtering."""
        where_clause = None
        print("Retrieving products with query:", query)
        if dataset_source != "all":
            where_clause = {"dataset_source": dataset_source}
            
        results = self.product_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )
        return results['documents'][0] if results['documents'] else []

    def cache_economic_data(self, country: str, context: EconomicContext):
        """Store economic data in cache."""
        context.last_updated = datetime.now().isoformat()
        content = f"Economic context for {country}: {context.model_dump_json()}"
        
        # Upsert: remove old record if exists
        self.economic_collection.delete(ids=[f"econ_{country.lower()}"])
        
        self.economic_collection.add(
            documents=[content],
            metadatas=[{"country": country.lower(), "timestamp": context.last_updated}],
            ids=[f"econ_{country.lower()}"]
        )

    def get_cached_economic_data(self, country: str) -> Optional[EconomicContext]:
        """Retrieve economic data from cache if fresh (within 24h)."""
        results = self.economic_collection.get(
            ids=[f"econ_{country.lower()}"]
        )
        
        if not results['documents']:
            return None
            
        metadata = results['metadatas'][0]
        timestamp = datetime.fromisoformat(metadata['timestamp'])
        if datetime.now() - timestamp > timedelta(hours=24):
            return None # Expired
            
        # Extract JSON from content
        doc = results['documents'][0]
        json_str = doc.split(":", 1)[1].strip()
        return EconomicContext.model_validate_json(json_str)

    def forgetting_mechanism(self, importance_score: float, timestamp: str, beta: float = 0.5, delta: float = 0.1) -> float:
        start_time = datetime.fromisoformat(timestamp)
        now = datetime.now()
        delta_time = (now - start_time).total_seconds()
        recency = math.exp(-delta_time / 86400)
        strength = importance_score
        forgetting_prob = ((strength + recency) / 2) * max(math.pow(recency, beta), delta)
        return 1 - forgetting_prob

    def clean_old_memories(self):
        all_mems = self.memory_collection.get()
        ids_to_delete = []
        for i, metadata in enumerate(all_mems['metadatas']):
            score = self.forgetting_mechanism(
                metadata.get('importance_score', 0.5),
                metadata.get('timestamp', datetime.now().isoformat())
            )
            if score < 0.2:
                ids_to_delete.append(all_mems['ids'][i])
        
        if ids_to_delete:
            self.memory_collection.delete(ids=ids_to_delete)

memory_manager = MemoryManager()

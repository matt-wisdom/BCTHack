from pydantic import BaseModel, Field
from typing import List, Dict, Any, Annotated, TypedDict, Optional, Literal
import operator
from datetime import datetime

class Persona(BaseModel):
    age: int
    traits: List[str]
    interests: List[str]
    country: str = "Nigeria"
    income_level: Optional[str] = "medium"
    behavioral_feature: Optional[str] = "Explorer" # e.g., Explorer, Critic, Saver
    preferred_language: Optional[str] = "English"

class Product(BaseModel):
    name: str
    specs: Dict[str, Any]
    description: str
    price: Optional[float] = None
    currency: Optional[str] = "NGN"

class SensoryObservation(BaseModel):
    content: str
    importance_score: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class EconomicContext(BaseModel):
    country: str = "Nigeria"
    inflation_rate: Optional[float] = None
    purchasing_power_index: Optional[float] = None
    exchange_rate: Optional[Dict[str, float]] = None
    last_updated: Optional[str] = None

class Recommendation(BaseModel):
    product_name: str
    reason: str
    rank: int

class ReviewOutput(BaseModel):
    review: str
    rating: int = Field(ge=1, le=10)
    economic_justification: str

class RecommendationOutput(BaseModel):
    recommendations: List[Recommendation]
    economic_justification: str

class SimulateReviewRequest(BaseModel):
    persona_text: str = Field(..., description="Free-form description of the user persona.")
    product_text: str = Field(..., description="Free-form description of the product.")
    mode: Literal["online", "offline"] = "offline"
    dataset_source: Literal["all", "amazon", "jumia", "store"] = "all"

class RecommendationRequest(BaseModel):
    persona_text: str = Field(..., description="Free-form description of the user persona.")
    mode: Literal["online", "offline"] = "offline"
    dataset_source: Literal["all", "amazon", "jumia", "store"] = "all"

class RecAgentState(TypedDict):
    persona: Persona
    product: Optional[Product]
    task_type: Literal["review", "recommendation"]
    mode: Literal["online", "offline"]
    dataset_source: Literal["all", "amazon", "jumia", "store"]
    # Tri-layer memory
    sensory_memory: Annotated[List[SensoryObservation], operator.add]
    short_term_memory: Annotated[List[str], operator.add]
    long_term_memory: List[str] # Retrieved from ChromaDB
    # Environment
    economic_context: EconomicContext
    # Reasoning
    reasoning_log: Annotated[List[str], operator.add]
    # Output
    review_output: Optional[ReviewOutput]
    recommendation_output: Optional[RecommendationOutput]

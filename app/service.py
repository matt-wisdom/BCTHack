import json
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from app.config import config
from app.models import (
    RecAgentState, Persona, Product, ReviewOutput, RecommendationOutput,
    SensoryObservation, EconomicContext, Recommendation
)
from app.memory import memory_manager
from datetime import datetime

COUNTRY_DEFAULTS = {
    "nigeria": {
        "inflation_rate": 29.9,
        "ppi": 35.0,
        "exchange_rate": {"USD": 1500.0}
    },
    "kenya": {
        "inflation_rate": 6.3,
        "ppi": 40.0,
        "exchange_rate": {"USD": 130.0}
    },
    "global": {
        "inflation_rate": 5.0,
        "ppi": 50.0,
        "exchange_rate": {"USD": 1.0}
    }
}

class RecAgentService:
    def __init__(self):
        self._model = None
        self._search = None

    @property
    def search(self):
        if self._search is None:
            self._search = DuckDuckGoSearchRun()
        return self._search

    @property
    def model(self):
        if self._model is None:
            if config.MODEL_PROVIDER == "gemini":
                self._model = ChatGoogleGenerativeAI(
                    model=config.MODEL_NAME or "gemini-2.0-flash-exp",
                    google_api_key=config.GOOGLE_API_KEY,
                )
            elif config.MODEL_PROVIDER == "ollama":
                self._model = ChatOllama(
                    model=config.MODEL_NAME or "llama3",
                    base_url=config.OLLAMA_BASE_URL,
                )
            else:
                raise ValueError(f"Unsupported model provider: {config.MODEL_PROVIDER}")
        return self._model

    async def env_context_node(self, state: RecAgentState) -> Dict[str, Any]:
        mode = state.get("mode", "offline")
        country = state["persona"].country.lower()
        print(f"[NODE: env_context] Fetching context for {country} (Mode: {mode})")
        
        cached = memory_manager.get_cached_economic_data(country)
        if cached:
            return {
                "economic_context": cached,
                "short_term_memory": [f"Observed economic context: Inflation {cached.inflation_rate}%, PPI {cached.purchasing_power_index}."],
                "reasoning_log": [f"Using cached economic context for {country}."]
            }

        if mode == "offline":
            data = COUNTRY_DEFAULTS.get(country, COUNTRY_DEFAULTS["global"])
            econ_context = EconomicContext(country=country.capitalize(), **data)
            return {
                "economic_context": econ_context,
                "short_term_memory": [f"Observed economic context: Inflation {econ_context.inflation_rate}%, PPI {econ_context.purchasing_power_index}."],
                "reasoning_log": [f"Using static offline defaults for {country}."]
            }
        
        search_query = f"current inflation rate purchasing power index and exchange rate for {country} {datetime.now().year}"
        try:
            economic_data_raw = self.search.run(search_query)
            prompt = f"Extract indicators for {country} from: {economic_data_raw}. JSON ONLY: {{\"inflation_rate\": float, \"ppi\": float, \"exchange_rate\": float}}"
            response = await self.model.ainvoke(prompt)
            content = response.content if isinstance(response.content, str) else json.dumps(response.content)
            data = json.loads(content.strip().replace("```json", "").replace("```", ""))
            econ_context = EconomicContext(country=country.capitalize(), inflation_rate=data.get("inflation_rate"), purchasing_power_index=data.get("ppi"), exchange_rate={"USD": data.get("exchange_rate")})
            memory_manager.cache_economic_data(country, econ_context)
        except:
            data = COUNTRY_DEFAULTS.get(country, COUNTRY_DEFAULTS["global"])
            econ_context = EconomicContext(country=country.capitalize(), **data)

        return {
            "economic_context": econ_context,
            "short_term_memory": [f"Observed economic context: Inflation {econ_context.inflation_rate}%, PPI {econ_context.purchasing_power_index}."],
            "reasoning_log": [f"Context node completed for {country}."]
        }

    async def memory_retriever_node(self, state: RecAgentState) -> Dict[str, Any]:
        task = state["task_type"]
        query = f"Persona interests: {', '.join(state['persona'].interests)}. History for {state['product'].name if state['product'] else 'shopping'}."
        long_term = memory_manager.retrieve_relevant(query)
        return {
            "long_term_memory": long_term,
            "reasoning_log": [f"Retrieved {len(long_term)} history items."]
        }

    async def self_reflection_node(self, state: RecAgentState) -> Dict[str, Any]:
        if not state["long_term_memory"]:
            return {"short_term_memory": ["No long-term memories found."]}
        prompt = f"Memories: {state['long_term_memory']}. 1. Generate a high-level abstract insight generalizing these past behaviors. 2. Detect linguistic style. JSON: {{\"insight\": string, \"preferred_style\": string}}"
        response = await self.model.ainvoke(prompt)
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        data = json.loads(content.strip().replace("```json", "").replace("```", ""))
        return {
            "short_term_memory": [f"Abstract Reflection on Past: {data['insight']}", f"Linguistic Style: {data['preferred_style']}"],
            "reasoning_log": [f"Self-reflection: {data['insight']}"]
        }

    async def product_discovery_node(self, state: RecAgentState) -> Dict[str, Any]:
        task = state["task_type"]
        mode = state["mode"]
        dataset_source = state.get("dataset_source", "all")
        country = state["persona"].country
        print(f"[NODE: product_discovery] Mode: {mode}, Task: {task}")
        
        observations = []
        search_target = state["product"].name if (task == "review" and state["product"]) else ", ".join(state["persona"].interests)
        local_results = memory_manager.retrieve_products(search_target, n_results=10, dataset_source=dataset_source)
        
        if local_results:
            observations.append(SensoryObservation(
                content=f"ACTUAL PRODUCTS FROM LOCAL DATABASE:\n{chr(10).join(local_results)}",
                importance_score=1.0
            ))
            print(f"[product_discovery] Found {len(local_results)} local products.")
        
        if mode == "online":
            web_query = f"products matching {search_target} prices in {country} {datetime.now().year}"
            print(f"[product_discovery] Web search: {web_query}")
            try:
                discovery_raw = self.search.run(web_query)
                observations.append(SensoryObservation(
                    content=f"WEB DISCOVERY (FOR SUPPLEMENTAL CONTEXT):\n{discovery_raw[:1500]}",
                    importance_score=0.7
                ))
            except Exception as e:
                print(f"[product_discovery] Web search failed: {e}")

        if not observations:
            observations.append(SensoryObservation(content="CRITICAL: No products found in database or web.", importance_score=1.0))

        return {
            "sensory_memory": observations,
            "short_term_memory": [f"Discovered {len(local_results)} items from local catalog."],
            "reasoning_log": [f"Discovery completed. Found {len(local_results)} local items."]
        }

    async def reasoning_engine_node(self, state: RecAgentState) -> Dict[str, Any]:
        task = state["task_type"]
        persona = state["persona"]
        econ = state["economic_context"]
        
        discovery_context = "\n---\n".join([obs.content for obs in state["sensory_memory"]])
        short_term_context = "\n".join(state["short_term_memory"])
        
        print(f"[NODE: reasoning_engine] Reasoning for {task}")
        
        if task == "review":
            prompt = f"""Evaluate this purchase for a {persona.age}yo {persona.behavioral_feature} in {econ.country}.
            Inflation: {econ.inflation_rate}%. Product: {state['product'].name if state['product'] else 'item'}.
            
            Short-Term Memory (Recent Context & Reflections):\n{short_term_context}
            Long-Term Memory (Past Behaviors):\n{state['long_term_memory']}
            Data Sources (Sensory):\n{discovery_context}
            
            Is this wise? Use the informative short-term memory and relevant long-term memory (including reflections). Use ONLY the provided data. Think step-by-step."""
        else:
            prompt = f"""Select 3 products for a {persona.age}yo {persona.behavioral_feature} in {econ.country}.
            Interests: {', '.join(persona.interests)}. Inflation: {econ.inflation_rate}%.
            
            CRITICAL INSTRUCTION: You MUST select products ONLY from the 'ACTUAL PRODUCTS FROM LOCAL DATABASE' list provided below. 
            Do NOT hallucinate or suggest products that are not in the list.
            
            Short-Term Memory (Recent Context & Reflections):\n{short_term_context}
            Long-Term Memory (Past Behaviors):\n{state['long_term_memory']}
            Data Sources (Sensory):\n{discovery_context}
            
            Reasoning Steps:
            1. Look at the local database results.
            2. Filter those that match interests and budget.
            3. Use informative short-term memory and relevant long-term memory (including reflections) to pick the top 3 and justify."""
            
        response = await self.model.ainvoke(prompt)
        reasoning = response.content if isinstance(response.content, str) else json.dumps(response.content)
        
        return {
            "reasoning_log": [f"Reasoning summary: {reasoning[:300]}..."],
            "short_term_memory": [f"Reasoning step: {reasoning[:300]}"]
        }

    async def task_output_node(self, state: RecAgentState) -> Dict[str, Any]:
        task = state["task_type"]
        persona = state["persona"]
        detected_style = persona.preferred_language
        for mem in state["short_term_memory"]:
            if "Linguistic Style:" in str(mem):
                detected_style = mem.split(":", 1)[1].strip()
                break

        discovery_context = "\n---\n".join([obs.content for obs in state["sensory_memory"]])

        if task == "review":
            structured_model = self.model.with_structured_output(ReviewOutput)
            prompt = f"Generate review. Reasoning: {state['reasoning_log'][-1]}. Context: {discovery_context}. Style: {detected_style}."
            output = await structured_model.ainvoke(prompt)
            
            # Summarizing short-term memories to be transferred to the long-term memory
            short_term_str = "\n".join(state["short_term_memory"])
            summary_prompt = f"Summarize these short-term memories into a concise memory to be transferred to long-term memory for future retrieval:\n{short_term_str}\nFinal Output Review: {output.review}"
            summary_response = await self.model.ainvoke(summary_prompt)
            summary = summary_response.content if isinstance(summary_response.content, str) else json.dumps(summary_response.content)
            memory_manager.add_to_long_term(content=summary, metadata={"importance_score": 0.8, "timestamp": datetime.now().isoformat()})
            
            return {"review_output": output}
        else:
            structured_model = self.model.with_structured_output(RecommendationOutput)
            prompt = f"""Generate 3 ranked recommendations. 
            Reasoning: {state['reasoning_log'][-1]}
            Available Catalog:\n{discovery_context}
            Style: {detected_style}
            CRITICAL: Use ONLY products mentioned in the catalog. DO NOT halluncinate."""
            output = await structured_model.ainvoke(prompt)
            
            # Summarizing short-term memories to be transferred to the long-term memory
            short_term_str = "\n".join(state["short_term_memory"])
            summary_prompt = f"Summarize these short-term memories into a concise memory to be transferred to long-term memory for future retrieval:\n{short_term_str}\nFinal Output Recommendations: {[r.product_name for r in output.recommendations]}"
            summary_response = await self.model.ainvoke(summary_prompt)
            summary = summary_response.content if isinstance(summary_response.content, str) else json.dumps(summary_response.content)
            memory_manager.add_to_long_term(content=summary, metadata={"importance_score": 0.8, "timestamp": datetime.now().isoformat()})
            
            return {"recommendation_output": output}

    def _build_workflow(self):
        workflow = StateGraph(RecAgentState)
        workflow.add_node("env_context", self.env_context_node)
        workflow.add_node("memory_retriever", self.memory_retriever_node)
        workflow.add_node("self_reflection", self.self_reflection_node)
        workflow.add_node("product_discovery", self.product_discovery_node)
        workflow.add_node("reasoning_engine", self.reasoning_engine_node)
        workflow.add_node("task_output", self.task_output_node)
        
        workflow.add_edge(START, "env_context")
        workflow.add_edge("env_context", "memory_retriever")
        workflow.add_edge("memory_retriever", "self_reflection")
        workflow.add_edge("self_reflection", "product_discovery")
        workflow.add_edge("product_discovery", "reasoning_engine")
        workflow.add_edge("reasoning_engine", "task_output")
        workflow.add_edge("task_output", END)
        return workflow.compile()

    async def extract_persona(self, text: str) -> Persona:
        structured_model = self.model.with_structured_output(Persona)
        prompt = f"Extract structured persona from: {text}. Detect country and preferred_language/style."
        return await structured_model.ainvoke(prompt)

    async def extract_product(self, text: str) -> Product:
        structured_model = self.model.with_structured_output(Product)
        prompt = f"Extract product: {text}"
        return await structured_model.ainvoke(prompt)

    async def simulate_review(self, persona_text: str, product_text: str, mode: str = "offline", dataset_source: str = "all") -> ReviewOutput:
        persona = await self.extract_persona(persona_text)
        product = await self.extract_product(product_text)
        app = self._build_workflow()
        initial_state = {"persona": persona, "product": product, "task_type": "review", "mode": mode, "dataset_source": dataset_source, "sensory_memory": [], "short_term_memory": [], "long_term_memory": [], "economic_context": EconomicContext(country=persona.country), "reasoning_log": []}
        result = await app.ainvoke(initial_state)
        return result["review_output"]

    async def generate_recommendations(self, persona_text: str, mode: str = "offline", dataset_source: str = "all") -> RecommendationOutput:
        persona = await self.extract_persona(persona_text)
        app = self._build_workflow()
        initial_state = {"persona": persona, "product": None, "task_type": "recommendation", "mode": mode, "dataset_source": dataset_source, "sensory_memory": [], "short_term_memory": [], "long_term_memory": [], "economic_context": EconomicContext(country=persona.country), "reasoning_log": []}
        result = await app.ainvoke(initial_state)
        return result["recommendation_output"]

rec_agent_service = RecAgentService()

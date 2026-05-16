import json
from typing import Any, Dict, List, Optional, Literal
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
                    model=config.MODEL_NAME or "gemini-2.5-flash-lite",
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
            prompt = f"""You are a data extraction specialist focused on macroeconomic indicators.
            
CONTEXT:
- Target Country: {country}
- Raw Search Data: {economic_data_raw}

INSTRUCTIONS:
1. Extract the current inflation rate as a float.
2. Extract the Purchasing Power Index (PPI) as a float.
3. Extract the current exchange rate to 1 USD as a float.

CONSTRAINTS:
- Output MUST be valid JSON only.
- Use the following keys: "inflation_rate", "ppi", "exchange_rate".
- If a value is missing, provide a reasonable estimate based on the context.
"""
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
        persona = state["persona"]
        persona_desc = f"Persona: Age {persona.age}, {persona.behavioral_feature}, Interests: {', '.join(persona.interests)}, Traits: {', '.join(persona.traits)}, Country: {persona.country}"
        
        query = f"Find past interactions for similar persona: {persona_desc}. Context: {state['product'].name if state['product'] else 'shopping'}."
        long_term = memory_manager.retrieve_relevant(query, n_results=10)
        return {
            "long_term_memory": long_term,
            "reasoning_log": [f"Retrieved {len(long_term)} history items from similar personas."]
        }

    async def self_reflection_node(self, state: RecAgentState) -> Dict[str, Any]:
        if not state["long_term_memory"]:
            return {"short_term_memory": ["No long-term memories found for similar personas."]}
        prompt = f"""You are a behavioral analyst specializing in consumer psychology.

CONTEXT:
- Historical memories of similar personas: {state['long_term_memory']}

INSTRUCTIONS:
1. Generate a high-level abstract insight generalizing how this type of persona behaves, what they prefer, or how their tastes evolve based on these past interactions.

CONSTRAINTS:
- Output MUST be JSON ONLY.
- Use key: "insight" (string).
- If no memories are present, return generic but relevant insights for the persona type.
"""
        response = await self.model.ainvoke(prompt)
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        data = json.loads(content.strip().replace("```json", "").replace("```", ""))
        return {
            "short_term_memory": [f"Abstract Reflection on Past: {data.get('insight', '')}"],
            "reasoning_log": [f"Self-reflection: {data.get('insight', '')}"]
        }

    async def product_discovery_node(self, state: RecAgentState) -> Dict[str, Any]:
        task = state["task_type"]
        mode = state["mode"]
        dataset_source = state.get("dataset_source", "all")
        country = state["persona"].country
        print(f"[NODE: product_discovery] Mode: {mode}, Task: {task}")
        
        if task == "review":
            return {
                "sensory_memory": [],
                "short_term_memory": ["Skipped discovery catalog search for review task."],
                "reasoning_log": ["Review task: using user-provided product details."]
            }
        
        observations = []
        search_target = ", ".join(state["persona"].interests)
        local_results = memory_manager.retrieve_products(search_target, n_results=10, dataset_source=dataset_source)
        
        if local_results:
            # Strip "Source: ..." labels to prevent store-focused reasoning
            cleaned_results = [
                "\n".join([line for line in res.split("\n") if not line.startswith("Source:")])
                for res in local_results
            ]
            observations.append(SensoryObservation(
                content=f"ACTUAL PRODUCTS FROM LOCAL DATABASE:\n{chr(10).join(cleaned_results)}",
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
            product = state["product"]
            product_details = f"Name: {product.name}\nPrice: {product.price} {product.currency}\nSpecs: {product.specs}\nDescription: {product.description}"
            
            prompt = f"""You are a knowledgeable product expert and consumer advocate.

PRIMARY CONTEXT:
- Persona: {persona.age}yo {persona.behavioral_feature} in {econ.country}
- Product to Evaluate:
{product_details}
- Recent Context (Short-Term Memory): {short_term_context}
- Past Behaviors (Long-Term Memory): {state['long_term_memory']}

CONTEXTUAL ECONOMIC DATA:
- Local Economic Indicators: Inflation {econ.inflation_rate}%, PPI {econ.purchasing_power_index}

INSTRUCTIONS:
1. Evaluate the product's features and utility specifically for the persona described.
2. Determine the persona's likely HONEST rating on a scale of 1-5.
3. SCALE THE IMPORTANCE OF ECONOMIC DATA:
   - If the persona is highly price-sensitive (e.g., 'Saver', 'Frugal', or has low income), then economic data (inflation, PPI, price) should STRONGLY influence their rating and reasoning.
   - Otherwise, economic data should matter much less than in recommendations; focus predominantly on user experience and utility.
4. Be critical: If the product is overpriced or poorly suited for a {persona.behavioral_feature}, explain why.
5. Think step-by-step and provide a logical justification focusing on user experience, fit, and value.

CONSTRAINTS:
- Use ONLY the provided data.
- Avoid positivity bias. If the fit is bad, the reasoning must clearly justify a low score.
- Be concise but thorough in your reasoning.
"""
        else:
            prompt = f"""You are an expert shopping curator and personal assistant.

PRIMARY CONTEXT:
- Persona: {persona.age}yo {persona.behavioral_feature} in {econ.country}
- Interests: {', '.join(persona.interests)}
- Available Products (Sensory Catalog - Local + Web): {discovery_context}
- Recent Context (Short-Term Memory): {short_term_context}
- Past Behaviors (Long-Term Memory): {state['long_term_memory']}

SECONDARY ECONOMIC CONTEXT:
- Local Economic Indicators: Inflation {econ.inflation_rate}%, PPI {econ.purchasing_power_index}

INSTRUCTIONS:
1. Review the products in the Sensory Catalog (Local Database and Web Discovery).
2. Select the top 3 products that best match the persona's interests and behavioral features.
3. Use the economic context only as a minor filter to ensure the selections are reasonable for their likely budget.
4. Justify why these specific 3 items provide the best utility and "fit" for the persona.
5. Think step-by-step.

CONSTRAINTS:
- Use ONLY products mentioned in the provided Sensory Catalog. 
- In ONLINE mode, you SHOULD prioritize fresh products from Web Discovery if they are more relevant than the Local Database items.
- Prioritize "fit" and "utility" over financial analysis.
"""
            
        response = await self.model.ainvoke(prompt)
        reasoning = response.content if isinstance(response.content, str) else json.dumps(response.content)

        return {
            "reasoning_log": [reasoning],
            "short_term_memory": [f"Reasoning output: {reasoning}"]
        }

    async def verification_node(self, state: RecAgentState) -> Dict[str, Any]:
        task = state["task_type"]
        mode = state["mode"]
        discovery_context = "\n---\n".join([obs.content for obs in state["sensory_memory"]])
        last_reasoning = state["reasoning_log"][-1]
        
        print(f"[NODE: verification] Auditing reasoning for {task} (Mode: {mode})")
        
        if task == "review":
            prompt = f"""You are a quality control auditor for an AI recommendation agent.

CONTEXT:
- Task: {task}
- Reasoning to Verify: {last_reasoning}

INSTRUCTIONS:
1. Verify if the reasoning provides a sensible evaluation of the product based on the persona's needs.
2. Check for general logical consistency.
3. Ensure the rating (on a scale of 1-5) is justified by the reasoning provided.
4. If valid, return "VALID".
5. If invalid, provide a specific and constructive CRITIQUE.

CONSTRAINTS:
- Output MUST be either "VALID" or a critique starting with "CRITIQUE:".
- Prioritize product relevance over macroeconomic analysis.
"""
        else:
            # For recommendations
            if mode == "online":
                prompt = f"""You are a quality control auditor for an AI recommendation agent.

CONTEXT:
- Task: {task}
- Mode: ONLINE (Web discovery is active)
- Discovery Context (Local + Web): {discovery_context}
- Reasoning to Verify: {last_reasoning}

INSTRUCTIONS:
1. Verify if the reasoning for selecting these 3 items is sensible and aligns with the persona.
2. Check for general logical consistency.
3. Since we are in ONLINE mode, products can come from either the local catalog or the web discovery context. Verify they are based on the provided discovery context.
4. If valid, return "VALID".
5. If invalid, provide a specific and constructive CRITIQUE.

CONSTRAINTS:
- Output MUST be either "VALID" or a critique starting with "CRITIQUE:".
- Focus on logical fit and relevance.
"""
            else:
                prompt = f"""You are a quality control auditor for an AI recommendation agent.

CONTEXT:
- Task: {task}
- Mode: OFFLINE (Strict local catalog only)
- Available Catalog from Local Database: {discovery_context}
- Reasoning to Verify: {last_reasoning}

INSTRUCTIONS:
1. Check if the products selected actually exist in the 'Available Catalog'.
2. Verify if the reasoning for selecting these 3 items is sensible and aligns with the persona.
3. Check for general logical consistency.
4. If valid, return "VALID".
5. If invalid, provide a specific and constructive CRITIQUE.

CONSTRAINTS:
- Output MUST be either "VALID" or a critique starting with "CRITIQUE:".
- CRITICAL: Zero hallucinations (only catalog items).
"""
        response = await self.model.ainvoke(prompt)
        verification_result = response.content if isinstance(response.content, str) else json.dumps(verification_result)
        print(verification_result)
        if "VALID" in verification_result.upper() and "CRITIQUE" not in verification_result.upper():
            return {
                "reasoning_log": ["Verification: PASSED"],
                "short_term_memory": ["Verification step: All recommendations and economic logic verified as valid."]
            }
        else:
            return {
                "reasoning_log": [f"Verification: FAILED. {verification_result}"],
                "short_term_memory": [f"CRITIQUE FROM AUDITOR: {verification_result}"]
            }

    async def task_output_node(self, state: RecAgentState) -> Dict[str, Any]:
        task = state["task_type"]
        persona = state["persona"]
        # Use persona's style directly from ground-truth input
        detected_style = persona.preferred_language

        discovery_context = "\n---\n".join([obs.content for obs in state["sensory_memory"]])

        persona_desc = f"Persona: Age {persona.age}, {persona.behavioral_feature}, Interests: {', '.join(persona.interests)}, Traits: {', '.join(persona.traits)}, Country: {persona.country}"

        if task == "review":
            product = state["product"]
            product_details = f"Name: {product.name}\nPrice: {product.price} {product.currency}\nSpecs: {product.specs}\nDescription: {product.description}"

            structured_model = self.model.with_structured_output(ReviewOutput)
            prompt = f"""You are a professional product reviewer adopting the persona of a {persona.age}yo {persona.behavioral_feature} from {persona.country}.

CONTEXT:
- Evaluated Product: 
{product_details}
- Inner Reasoning: {state['reasoning_log'][-1]}

INSTRUCTIONS:
1. Write a review reflecting the persona's HONEST sentiment about the product's value and utility.
2. Provide a rating from 1 to 5. 
   - 1: Very Poor / Deal-breaker
   - 2: Poor / Not recommended
   - 3: Average / Mediocre
   - 4: Good / Recommended
   - 5: Excellent / Highly recommended
3. Be critical: Use the FULL range of the scale. If the persona traits (e.g., {persona.behavioral_feature}) or the reasoning suggests dissatisfaction, provide a low rating (1-2).
4. Provide a brief economic justification ONLY if relevant or if the persona is highly price-sensitive (e.g., 'Saver', 'Frugal'). Otherwise, keep it extremely brief or omit detailed financial analysis.
5. Embody the user's geographic location and writing habits. Write using the vocabulary, regional slang, idioms, and grammatical quirks typical of someone from {persona.country} with the following linguistic traits: {detected_style}.
6. Adjust the review length based on the item type (e.g., shorter for simple products, longer for complex items or movies) unless the persona's linguistic traits suggest they are naturally very concise or very verbose.

CONSTRAINTS:
- Write strictly in the voice and vocabulary of the localized persona.
- DO NOT default to a high rating. Be as critical as the persona dictates.
- Do NOT mention catalog availability.
- Output MUST match the requested JSON schema exactly.
"""
            output = await structured_model.ainvoke(prompt)
            
            # Summarizing short-term memories to be transferred to the long-term memory
            short_term_str = "\n".join(state["short_term_memory"])
            summary_prompt = f"""You are a memory architect. Your goal is to condense a session into a durable, objective behavioral record.

CONTEXT:
- Persona: {persona_desc}
- Short-Term Reasoning Logs: {short_term_str}
- Decision: Rating of {output.rating}/5

INSTRUCTIONS:
1. Create a concise summary of the persona's behavior and decisions during this session.
2. Focus on the core behavioral patterns: why did the persona make this specific choice or rating?
3. Describe the outcome objectively (e.g., "The persona gave a low rating due to X") without using the persona's first-person voice or original review text.

CONSTRAINTS:
- Max 3 sentences.
- Ensure high semantic relevance for future vector searches of similar personas.
- DO NOT include the AI's generated review text or adopt the persona's voice.
"""
            summary_response = await self.model.ainvoke(summary_prompt)
            summary = summary_response.content if isinstance(summary_response.content, str) else json.dumps(summary_response.content)
            memory_manager.add_to_long_term(content=summary, metadata={"importance_score": 0.8, "timestamp": datetime.now().isoformat(), "persona_country": persona.country})
            
            return {"review_output": output}
        else:
            structured_model = self.model.with_structured_output(RecommendationOutput)
            prompt = f"""You are a helpful shopping assistant. Help a {persona.age}yo {persona.behavioral_feature} from {persona.country} find the best products.

CONTEXT:
- Available Catalog (Local + Web): {discovery_context}
- Inner Reasoning: {state['reasoning_log'][-1]}

INSTRUCTIONS:
1. Select the top 3 ranked products from the catalog provided in the CONTEXT.
2. Provide a clear reason for each recommendation, focusing on product utility and persona fit.
3. Provide a brief economic justification as a supporting note.
4. Embody the shopping assistant's geographic location. Write using the vocabulary, regional slang, idioms, and grammatical quirks typical of someone from {persona.country} with the following linguistic traits: {detected_style}.

CONSTRAINTS:
- Use ONLY products mentioned in the provided 'Available Catalog'. 
- In ONLINE mode, you should prefer fresh products from Web Discovery if they are more relevant.
- Prioritize product utility over financial indicators.
- Write strictly in the voice and vocabulary of the localized assistant.
- Output MUST match the requested JSON schema exactly.
"""
            output = await structured_model.ainvoke(prompt)
            
            # Summarizing short-term memories to be transferred to the long-term memory
            short_term_str = "\n".join(state["short_term_memory"])
            summary_prompt = f"""You are a memory architect. Your goal is to condense a session into a durable, objective behavioral record.

CONTEXT:
- Persona: {persona_desc}
- Short-Term Reasoning Logs: {short_term_str}
- Decision: Recommended {[r.product_name for r in output.recommendations]}

INSTRUCTIONS:
1. Create a concise summary of the persona's behavior and decisions during this session.
2. Focus on the core behavioral patterns: why did the persona prefer these specific items?
3. Describe the outcome objectively (e.g., "The persona preferred X because of Y") without using the persona's first-person voice.

CONSTRAINTS:
- Max 3 sentences.
- Ensure high semantic relevance for future vector searches of similar personas.
- DO NOT include the AI's generated reasoning text or adopt the persona's voice.
"""
            summary_response = await self.model.ainvoke(summary_prompt)
            summary = summary_response.content if isinstance(summary_response.content, str) else json.dumps(summary_response.content)
            memory_manager.add_to_long_term(content=summary, metadata={"importance_score": 0.8, "timestamp": datetime.now().isoformat(), "persona_country": persona.country})
            
            return {"recommendation_output": output}

    def route_after_verification(self, state: RecAgentState) -> Literal["task_output", "reasoning_engine"]:
        last_log = state["reasoning_log"][-1]
        if "Verification: PASSED" in last_log:
            return "task_output"
        
        # Count failures to prevent infinite loops (max 2 retries)
        failure_count = len([log for log in state["reasoning_log"] if "Verification: FAILED" in log])
        if failure_count >= 2:
            print(f"[workflow] Max retries reached ({failure_count}). Proceeding to output.")
            return "task_output"
        
        print(f"[workflow] Verification failed. Looping back to reasoning_engine (Retry #{failure_count})")
        return "reasoning_engine"

    def _build_workflow(self):
        workflow = StateGraph(RecAgentState)
        workflow.add_node("env_context", self.env_context_node)
        workflow.add_node("memory_retriever", self.memory_retriever_node)
        workflow.add_node("self_reflection", self.self_reflection_node)
        workflow.add_node("product_discovery", self.product_discovery_node)
        workflow.add_node("reasoning_engine", self.reasoning_engine_node)
        workflow.add_node("verification", self.verification_node)
        workflow.add_node("task_output", self.task_output_node)
        
        workflow.add_edge(START, "env_context")
        workflow.add_edge("env_context", "memory_retriever")
        workflow.add_edge("memory_retriever", "self_reflection")
        workflow.add_edge("self_reflection", "product_discovery")
        workflow.add_edge("product_discovery", "reasoning_engine")
        workflow.add_edge("reasoning_engine", "verification")
        
        workflow.add_conditional_edges(
            "verification",
            self.route_after_verification,
            {
                "task_output": "task_output",
                "reasoning_engine": "reasoning_engine"
            }
        )
        
        workflow.add_edge("task_output", END)
        return workflow.compile()

    async def extract_persona(self, text: str) -> Persona:
        structured_model = self.model.with_structured_output(Persona)
        prompt = f"""You are a structured data extractor specializing in consumer behavior analysis.

CONTEXT:
- Input Text (User Review/Description): {text}

INSTRUCTIONS:
1. Extract the structured persona details from the input text.
2. Carefully analyze the TONE of the input. If the user sounds frustrated, angry, or disappointed, reflect this in their "traits" (e.g., "Critical", "Disappointed") and "behavioral_feature" (e.g., "Critic", "Frugal").
3. Determine the user's country. For 'preferred_language', explicitly capture the specific linguistic style, regional dialect, local slang, and grammatical quirks evident in the text (e.g., "British English with informal slang", "Indonesian casual internet slang"). Also, gauge the user's verbosity (e.g., "concise", "verbose", "average") from the input.
4. Identify age, behavioral features, interests, and traits.

CONSTRAINTS:
- Output MUST strictly match the Persona schema.
- If information is missing, infer reasonable defaults based on the text.
"""
        return await structured_model.ainvoke(prompt)

    async def extract_product(self, text: str) -> Product:
        structured_model = self.model.with_structured_output(Product)
        prompt = f"""You are a structured data extractor specializing in product information.

CONTEXT:
- Input Text: {text}

INSTRUCTIONS:
1. Extract the structured product details (name, specs, description, price, currency) from the input text.

CONSTRAINTS:
- Output MUST strictly match the Product schema.
- If specific fields are missing, provide null or reasonable defaults.
"""
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

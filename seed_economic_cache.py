import sys
import os
import asyncio

# Ensure the app module can be imported
sys.path.append(os.getcwd())
from app.service import rec_agent_service
from app.models import RecAgentState, Persona, EconomicContext

async def seed_cache(countries):
    print(f"Starting economic cache seeding for: {', '.join(countries)}")
    
    for country in countries:
        print(f"\nProcessing {country}...")
        try:
            # We mock a minimal state to trigger the env_context_node
            state: RecAgentState = {
                "mode": "online",
                "persona": Persona(age=30, traits=[], interests=[], country=country),
                "reasoning_log": [],
                "task_type": "recommendation",
                "dataset_source": "all",
                "product": None,
                "sensory_memory": [],
                "short_term_memory": [],
                "long_term_memory": [],
                "economic_context": EconomicContext(country=country)
            }
            
            # This node already has the logic to check cache, fetch if missing, and save to cache
            result = await rec_agent_service.env_context_node(state)
            econ = result["economic_context"]
            
            print(f"Successfully cached data for {country}:")
            print(f"  - Inflation: {econ.inflation_rate}%")
            print(f"  - PPI: {econ.purchasing_power_index}")
            print(f"  - Exchange Rate: {econ.exchange_rate}")
            
        except Exception as e:
            print(f"Failed to seed cache for {country}: {e}")

if __name__ == "__main__":
    # Define the list of countries you want to pre-cache
    target_countries = ["Nigeria", "Kenya", "South Africa", "United Kingdom", "United States", "Ghana"]
    
    asyncio.run(seed_cache(target_countries))
    print("\nEconomic cache seeding complete!")

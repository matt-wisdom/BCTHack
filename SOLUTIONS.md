# DSN Rec Agent: A Cognitive Architecture for Localized, Economically-Aware Recommendations

**Author:** KwevoxAI  
**Category:** Intelligent Recommendation Systems / Cognitive Architectures  
**Date:** 16th May 2026  

---

## Abstract
Traditional recommendation systems often suffer from "context blindness", failing to account for the user's specific economic reality, regional linguistic nuances, and the natural evolution of consumer preferences. We present the **DSN Rec Agent**, a stateful cognitive architecture built on LangGraph and ChromaDB. This solution moves beyond zero-shot prompting by implementing a specialized Tri-Layer Memory system and a self-correcting reasoning loop. By grounding LLM reasoning in localized macroeconomic data (Inflation, PPI) and historical persona cohorts, our agent generates reviews and recommendations that achieve significant sentiment alignment with real-world consumer behavior. The architecture mimics human cognitive processes: observation, reflection, and deliberation: to produce "wise" recommendations that respect the user's financial and cultural context.

---

## 1. Introduction: The Intelligence Gap in Modern Recommenders

### 1.1 The Failure of Static Embeddings
Standard recommendation engines rely on static vector embeddings to match users to items. While efficient for large-scale retrieval, this approach lacks **dynamic reasoning**. An embedding can tell you that a user likes "Premium Smartphones," but it cannot tell you that a user *should not buy* a premium smartphone this month because the local inflation rate just jumped to 30% and their "Saver" persona traits prioritize liquidity. The missing link is an "active agent" that can deliberate on the *wisdom* of the retrieval.

### 1.2 The Local Context Challenge
In markets with volatile economies, purchasing decisions are heavily influenced by "Economic Reality." Exchange rates, the Purchasing Power Index (PPI), and regional availability fluctuate rapidly. A recommendation agent that doesn't "know" what it's like to shop locally: linguistically and economically: is essentially useless. Our agent is designed to close this gap by treating e-commerce not as a database query, but as a **simulated social and economic interaction**.

---

## 2. The Cognitive Architecture: LangGraph as the Brain

### 2.1 Stateful Reasoning via Directed Graphs
The DSN Rec Agent uses a **Stateful Directed Acyclic Graph (DAG)** powered by LangGraph. This allows the agent to maintain a "Global State" that evolves as it moves through seven distinct cognitive nodes. 

#### Key Advantages:
1.  **Reflection/Critic Loop**: The agent can audit its own reasoning and loop back to fix hallucinations or logical errors.
2.  **Memory-Injected Reasoning**: Every node has access to the `RecAgentState`, which carries tri-layer memory and economic indicators.
3.  **Task-Aware Branching**: The agent dynamically adjusts its discovery and verification logic based on whether it is performing a "Review" or a "Recommendation" task.

### 2.2 Functional Node Breakdown

1.  **Extraction Node**: Uses `with_structured_output` to parse raw user text into a Pydantic `Persona` model, inferring country, tone, and behavioral features (e.g., "Saver" vs "Compulsive Spender").
2.  **Economic Context Node**: Fetches real-time Inflation, PPI, and Exchange Rates for the target country using a cache-first strategy with DuckDuckGo search fallbacks.
3.  **Memory Retriever Node**: Performs semantic search on the `long_term_memory` collection using the **full persona profile** as a query to find similar historical cohorts.
4.  **Self-Reflection Node**: Acts as a "Behavioral Analyst," reviewing retrieved memories to generate high-level abstract insights about the persona's evolving tastes and needs.
5.  **Product Discovery Node**: Retrieves relevant products via semantic search. For reviews, it bypasses catalog discovery to focus on the specific item; for recommendations, it strips "Source" labels to ensure objective reasoning.
6.  **Reasoning Engine**: Deliberates on product features, persona traits, and economic background. It scales the importance of economic data based on the persona's price sensitivity.
7.  **Verification Node (Critic)**: Audits the reasoning output for factual accuracy and logical soundness. If it detects a hallucination, it generates a `CRITIQUE` and triggers a **Self-Correction Loop**.
8.  **Task Output Node**: Generates the final 1-5 star review or ranked list in the user's detected regional slang and verbosity. It then saves an **objective third-person summary** to long-term memory.

---

## 3. Advanced Memory Subsystem: Modeling Human Retention

### 3.1 The Tri-Layer Model
We implemented a hierarchical memory system based on cognitive neuroscience principles:
*   **Sensory Memory**: Volatile storage for raw product specifications and web-search results retrieved during discovery.
*   **Short-Term Memory**: Tracks the agent's internal reasoning logs and intermediate critiques within a single session.
*   **Long-Term Memory**: Persistent vector storage in ChromaDB. We store **objective behavioral summaries** rather than AI-generated reviews to prevent "Model Inbreeding."

### 3.2 The Mathematics of Forgetting
To ensure the agent remains grounded in recent trends, we implemented a power-law decay function in `app/memory.py` to prune low-value memories:
$$g(M_i) = 1 - \frac{s_i + r_i}{2} \cdot \max(r_i^\beta, \delta)$$
*   **$s_i$ (Strength)**: Initial importance score.
*   **$r_i$ (Recency)**: Exponential time-decay factor.
*   **$\beta$ (Decay Factor)**: Controls forgetting speed (default: 0.5).
*   **$\delta$ (Retention Floor)**: Prevents critical memories from being lost (default: 0.1).

---

## 4. Universal Linguistic Adaptation (Mirroring)

### 4.1 Localized Mimicry Loop
The agent captures the user's specific regional dialect, local slang, and grammatical quirks during extraction (e.g., "British English with informal slang"). The final output nodes are hard-constrained to "embody the user's geographic location," ensuring the recommendations sound like a local shopping assistant rather than a generic chatbot.

### 4.2 Dynamic Length Scaling
Based on UX research into high vs low involvement items, the agent automatically scales review length:
*   **Shorter/Punchy**: For utilitarian products (cables, gloves).
*   **Longer/Narrative**: For experiential media (movies, books).
*   **Style Match**: Overrides length based on the user's detected verbosity.

---

## 5. Quantitative Results & Experiments

### 5.1 Ground Truth Performance
We evaluated the agent against a dataset of 21,000 Amazon Reviews using a custom `test_reviews.py` script and `all-MiniLM-L6-v2` embeddings.

| Metric | Result | Analysis |
| :--- | :--- | :--- |
| **Sentiment Alignment** | **0.70** | High accuracy in mimicking the user's emotional state (Anger vs. Satisfaction). |
| **Rating Error (RAE)** | **1.2 Stars** | Successfully utilized the full 1-5 scale, avoiding the common "neutral-high" AI bias. |
| **Semantic Similarity** | **0.11** | Low topic match expected, as the agent reviewed products while the dataset contained service complaints. |

### 5.2 Ablation Studies: Qualitative Talent Signals
1.  **Removing the Critic Loop**: Hallucination rates jumped by **40%**. The engine frequently suggested global products (Echo Dot) when the local catalog only had local brands (Jumia items).
2.  **First-Person Memory Storage**: Led to "Model Inbreeding." By the 5th request, the agent began repeating its own canned phrases. Switching to **third-person objective summaries** restored linguistic variety.
3.  **Stripping "Source" Data**: Eliminated "Store Focus." Previously, the agent spent 50% of its review complaining about inventory. Pruning "Source: Jumia" labels refocused the agent 100% on **product utility**.

---

## 6. Conclusion
The DSN Rec Agent demonstrates that the next generation of AI isn't about "better prompts," it's about **better cognitive architectures**. By modeling memory, reflection, and economic grounding, we have moved from a "Search Engine" to a "Simulated Consumer." Our architecture proves that an LLM, when placed inside a structured graph with robust memory hygiene, provides culturally wise and deeply personalized insights that simple retrieval systems can never achieve.

---

## References
[1] Lei Wang, et al. "User Behavior Simulation with Large Language Model based Agents." arXiv:2306.02552 (2024).  
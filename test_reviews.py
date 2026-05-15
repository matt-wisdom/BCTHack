import asyncio
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from app.service import rec_agent_service
from app.models import ReviewOutput

# Initialize the embedding model for semantic evaluation
print("Loading evaluation model (all-MiniLM-L6-v2)...")
eval_model = SentenceTransformer('all-MiniLM-L6-v2')

async def evaluate_agent_reviews(csv_path: str, num_samples: int = 5):
    print(f"🚀 Starting Advanced Review Evaluation using {csv_path}")
    
    # Load dataset with robust settings
    try:
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip', encoding='latin-1')
        
    actual_samples = min(num_samples, len(df))
    sampled_df = df.sample(n=actual_samples, random_state=42)
    
    results = []
    
    for index, row in sampled_df.iterrows():
        print(f"\n--- Testing Case {len(results) + 1}/{actual_samples} ---")
        
        ground_truth_text = str(row.get('Review Text', ''))
        ground_truth_rating_str = str(row.get('Rating', ''))
        country = str(row.get('Country', 'Unknown'))
        
        if pd.isna(row['Review Text']) or not ground_truth_text.strip():
            continue

        try:
            ground_truth_rating = int(ground_truth_rating_str.split()[1])
        except (ValueError, IndexError):
            ground_truth_rating = 3 # Default to neutral if parsing fails
            
        persona_text = f"A customer from {country}. Their general attitude: '{ground_truth_text[:200]}'"
        product_text = "The product mentioned in the review context."
        
        try:
            print("⏳ Agent is thinking...")
            generated: ReviewOutput = await rec_agent_service.simulate_review(
                persona_text=persona_text,
                product_text=product_text,
                mode="offline"
            )
            
            # --- Advanced Metrics via Sentence Transformers ---
            # 1. Semantic Similarity Score
            embeddings = eval_model.encode([ground_truth_text, generated.review], convert_to_tensor=True)
            semantic_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            # 2. Sentiment Alignment (Rating-based proxy)
            # Normalize ratings to 0-1 range to compare sentiment proximity
            # (1=0.0, 5=1.0)
            gt_norm = (ground_truth_rating - 1) / 4
            gen_norm = (generated.rating - 1) / 4
            sentiment_alignment = 1 - abs(gt_norm - gen_norm)

            print(f"📊 Semantic Score: {semantic_score:.4f}")
            print(f"🎭 Sentiment Alignment: {sentiment_alignment:.4f}")
            print(f"⭐ Ratings: GT={ground_truth_rating} | Gen={generated.rating}")
            
            results.append({
                "ground_truth": ground_truth_text,
                "generated": generated.review,
                "gt_rating": ground_truth_rating,
                "gen_rating": generated.rating,
                "semantic_score": semantic_score,
                "sentiment_alignment": sentiment_alignment,
                "rating_error": abs(generated.rating - ground_truth_rating)
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            
    if results:
        rdf = pd.DataFrame(results)
        print(f"\n=== 🏁 Final Aggregate Metrics ===")
        print(f"Avg Semantic Similarity: {rdf['semantic_score'].mean():.4f}")
        print(f"Avg Sentiment Alignment: {rdf['sentiment_alignment'].mean():.4f}")
        print(f"Avg Rating Error: {rdf['rating_error'].mean():.2f} stars")
        
        rdf.to_csv("advanced_evaluation_results.csv", index=False)
        print("📁 Detailed report saved to 'advanced_evaluation_results.csv'")

if __name__ == "__main__":
    from app.memory import memory_manager
    memory_manager.initialize()
    asyncio.run(evaluate_agent_reviews("Amazon_Reviews.csv", num_samples=10))

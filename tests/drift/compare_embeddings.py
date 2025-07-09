#!/usr/bin/env python3
"""
Compare drift scores between different embedding models.
"""

import sys
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Please install: pip install sentence-transformers scikit-learn")
    sys.exit(1)

# Test sentences
photography_text = "How do I achieve a blurred background effect?"
finance_text = "How do I create a budget?"

# Models to test
models = [
    "paraphrase-mpnet-base-v2",  # Current default
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
]

print("Comparing embedding models for topic drift detection")
print("=" * 60)
print(f"Text 1 (Photography): {photography_text}")
print(f"Text 2 (Finance): {finance_text}")
print("=" * 60)

results = []

for model_name in models:
    try:
        print(f"\nTesting {model_name}...")
        model = SentenceTransformer(model_name)
        
        # Get embeddings
        emb1 = model.encode(photography_text)
        emb2 = model.encode(finance_text)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        drift_score = 1 - similarity  # Convert to drift score
        
        results.append((model_name, drift_score))
        print(f"  Drift score: {drift_score:.3f}")
        
    except Exception as e:
        print(f"  Error: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY (sorted by drift score):")
print("=" * 60)

results.sort(key=lambda x: x[1], reverse=True)
for model_name, drift_score in results:
    status = "✓ Would trigger" if drift_score >= 0.9 else "✗ Would not trigger"
    print(f"{drift_score:.3f} - {model_name:<30} {status}")

print("\nRecommendation:")
if any(score >= 0.9 for _, score in results):
    best_model = [m for m, s in results if s >= 0.9][0]
    print(f"Use '{best_model}' for stronger topic separation")
else:
    print("Consider lowering the drift threshold or using different models")
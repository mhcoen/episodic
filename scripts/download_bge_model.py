#!/usr/bin/env python3
"""
Download BGE embedding model for testing.
BGE (BAAI General Embedding) models are high-quality embedding models.
"""

import sys

print("Downloading BGE embedding model...")
print("This may take a few minutes on first run...")

try:
    from sentence_transformers import SentenceTransformer
    
    # Download BGE model
    model_name = 'BAAI/bge-base-en-v1.5'
    print(f"\nDownloading: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Test it works
    test_texts = [
        "How do I take better portraits?",
        "What's the best way to create a budget?"
    ]
    embeddings = model.encode(test_texts)
    
    # Calculate similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    print(f"\n✅ Model downloaded successfully!")
    print(f"   Model dimension: {len(embeddings[0])}")
    print(f"   Test similarity (photography vs finance): {1 - similarity:.3f}")
    print(f"   Cache location: ~/.cache/huggingface/")
    
except ImportError:
    print("\n❌ Error: sentence-transformers not installed")
    print("Please run: pip install sentence-transformers scikit-learn")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error downloading model: {e}")
    sys.exit(1)

print("\nTo use this model in Episodic:")
print('  /set drift_embedding_model "BAAI/bge-base-en-v1.5"')
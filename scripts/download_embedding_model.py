#!/usr/bin/env python3
"""
Pre-download embedding models to avoid HuggingFace rate limiting during testing.
"""

import sys

print("Downloading sentence transformer model...")
print("This may take a few minutes on first run...")

try:
    from sentence_transformers import SentenceTransformer
    
    # Download the default model used by episodic
    model_name = 'paraphrase-mpnet-base-v2'
    print(f"\nDownloading: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Test it works
    test_embedding = model.encode("Test sentence")
    print(f"✅ Model downloaded successfully!")
    print(f"   Model dimension: {len(test_embedding)}")
    print(f"   Cache location: ~/.cache/huggingface/")
    
except ImportError:
    print("\n❌ Error: sentence-transformers not installed")
    print("Please run: pip install sentence-transformers")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error downloading model: {e}")
    sys.exit(1)

print("\nYou can now run episodic tests without rate limiting issues!")
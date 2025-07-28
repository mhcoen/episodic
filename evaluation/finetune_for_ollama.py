#!/usr/bin/env python3
"""
Fine-tuning guide for Ollama-compatible models.

Since Ollama models can't be directly fine-tuned through the Ollama interface,
this script provides guidance on the process.
"""

import json
from pathlib import Path
from collections import Counter


def analyze_training_data():
    """Analyze the prepared training data."""
    print("Analyzing Training Data for Fine-tuning")
    print("="*60)
    
    data_dir = Path("evaluation/finetuning_data")
    
    for file_path in data_dir.glob("*.jsonl"):
        print(f"\n{file_path.name}:")
        print("-"*40)
        
        examples = []
        with open(file_path) as f:
            for line in f:
                examples.append(json.loads(line))
        
        # Analyze
        outputs = [ex['output'] for ex in examples]
        output_counts = Counter(outputs)
        
        print(f"Total examples: {len(examples)}")
        print(f"Label distribution: {dict(output_counts)}")
        
        # Sample examples
        print("\nSample examples:")
        for i, ex in enumerate(examples[:2]):
            print(f"\nExample {i+1}:")
            print(f"Instruction: {ex['instruction'][:100]}...")
            print(f"Input: {ex['input'][:150]}...")
            print(f"Output: {ex['output']}")


def create_modelfile_template():
    """Create an Ollama Modelfile template for fine-tuned model."""
    
    modelfile_content = """# Modelfile for fine-tuned topic detection model
# Based on TinyLlama fine-tuned for binary classification

FROM tinyllama

# Set temperature low for consistent outputs
PARAMETER temperature 0.1
PARAMETER num_predict 10
PARAMETER stop "</s>"
PARAMETER stop "<|assistant|>"

# System prompt for topic detection
SYSTEM \"\"\"You are a topic boundary detector. Given two messages or windows of messages, determine if there is a topic change.
Output only '1' for topic change or '0' for same topic. Do not provide any explanation.\"\"\"

# Example template
TEMPLATE \"\"\"<|system|>
{{ .System }}</s>
<|user|>
{{ .Prompt }}</s>
<|assistant|>
\"\"\"
"""
    
    with open("evaluation/finetuning_data/Modelfile.topic-detector", "w") as f:
        f.write(modelfile_content)
    
    print("\nCreated Modelfile template at: evaluation/finetuning_data/Modelfile.topic-detector")


def main():
    """Main function with fine-tuning workflow."""
    print("Fine-tuning Workflow for Ollama Models")
    print("="*60)
    
    # Analyze data
    analyze_training_data()
    
    # Create Modelfile
    create_modelfile_template()
    
    print("\n\nFine-tuning Steps:")
    print("="*60)
    print("""
1. PREPARE BASE MODEL
   - Download TinyLlama from Hugging Face
   - Or use ollama pull tinyllama and export it

2. FINE-TUNE OUTSIDE OLLAMA
   Option A: Use the provided finetune_tinyllama_qlora.py script
   Option B: Use a service like:
   - Hugging Face AutoTrain
   - Modal Labs
   - Replicate
   - Together AI

3. CONVERT TO OLLAMA FORMAT
   - Export fine-tuned model to GGUF format
   - Use llama.cpp's convert.py script
   - Quantize if needed (q4_0, q4_1, q5_0, etc.)

4. CREATE OLLAMA MODEL
   ollama create topic-detector -f Modelfile.topic-detector

5. TEST THE MODEL
   ollama run topic-detector "Message 1: How's the weather?\\nMessage 2: Can you help with Python?"

ALTERNATIVE: Prompt Engineering
Instead of fine-tuning, you could:
1. Use few-shot prompting with examples
2. Create a specialized system prompt
3. Use a larger model that follows instructions better
""")
    
    print("\n\nRecommended Approach for Production:")
    print("="*60)
    print("""
Given the constraints with small models and window_size=3:

1. FOR SIMPLE DEPLOYMENT:
   - Use mistral:instruct or llama3 with good prompts
   - These models already handle instructions well
   - No fine-tuning needed

2. FOR RESOURCE-CONSTRAINED:
   - Fine-tune TinyLlama on window_size=1 only
   - Deploy with very simple prompts
   - Accept the limitations

3. FOR BEST PERFORMANCE:
   - Use Sentence-BERT embeddings
   - Calculate cosine similarity
   - No LLM needed at all

The data we prepared (66K+ examples) is excellent for:
- Training a small classifier on top of embeddings
- Fine-tuning a small transformer
- Creating a specialized model

But given the results, traditional ML approaches might be more suitable
for production deployment than small instruct LLMs.
""")


if __name__ == "__main__":
    analyze_training_data()
    create_modelfile_template()
    main()
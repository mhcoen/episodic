#!/usr/bin/env python3
"""Test the fine-tuned topic detection model."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


class TopicDetectionModel(nn.Module):
    """Simple classification head on top of a small transformer."""
    
    def __init__(self, model_name="microsoft/xtremedistil-l6-h256-uncased", num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def test_model():
    """Test the fine-tuned model on various examples."""
    print("Testing Fine-tuned Topic Detection Model")
    print("="*60)
    
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = Path("evaluation/finetuned_models/topic_detector_xtremedistil.pt")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
    model = TopicDetectionModel()
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Validation metrics: {checkpoint['val_metrics']}")
    
    # Test examples - matching training format exactly
    test_cases = [
        # Same topic examples
        {
            "name": "Weather continuation",
            "instruction": "Determine if there is a topic change between these two messages. Output only '1' for topic change or '0' for same topic.",
            "input": "Message 1: What's the weather like today?\nMessage 2: It's sunny and warm, perfect for outdoor activities.",
            "expected": 0
        },
        {
            "name": "Benefits continuation",
            "instruction": "Determine if there is a topic change between these two messages. Output only '1' for topic change or '0' for same topic.",
            "input": "Message 1: I need information about social security benefits\nMessage 2: Are you asking about retirement benefits specifically?",
            "expected": 0
        },
        # Topic change examples
        {
            "name": "Weather to programming",
            "instruction": "Determine if there is a topic change between these two messages. Output only '1' for topic change or '0' for same topic.",
            "input": "Message 1: The weather is really nice today\nMessage 2: Can you help me debug this Python code?",
            "expected": 1
        },
        {
            "name": "Benefits to documents",
            "instruction": "Determine if there is a topic change between these two messages. Output only '1' for topic change or '0' for same topic.",
            "input": "Message 1: So I'll receive the maximum benefit amount?\nMessage 2: You'll need to bring your birth certificate and ID",
            "expected": 1
        },
        # Edge cases
        {
            "name": "Very short messages",
            "instruction": "Determine if there is a topic change between these two messages. Output only '1' for topic change or '0' for same topic.",
            "input": "Message 1: Yes\nMessage 2: No",
            "expected": 0  # Ambiguous, but likely continuation
        }
    ]
    
    print("\n\nTest Results:")
    print("-"*60)
    
    correct = 0
    with torch.no_grad():
        for test in test_cases:
            # Format exactly like training data
            text = f"{test['instruction']}\n\n{test['input']}"
            
            # Tokenize
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Predict
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred = torch.argmax(logits, dim=1).item()
            
            # Results
            is_correct = pred == test['expected']
            if is_correct:
                correct += 1
            
            print(f"\nTest: {test['name']}")
            print(f"Input preview: {test['input'][:80]}...")
            print(f"Expected: {test['expected']}, Predicted: {pred} {'✓' if is_correct else '✗'}")
            print(f"Confidence: [same: {probs[0]:.3f}, change: {probs[1]:.3f}]")
    
    accuracy = correct / len(test_cases)
    print(f"\n\nOverall Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")
    
    # Compare with the evaluation results
    print("\n\nComparison with Validation Set:")
    print(f"Validation F1: {checkpoint['val_metrics']['f1']:.3f}")
    print(f"Validation Accuracy: {checkpoint['val_metrics']['accuracy']:.3f}")
    print("\nNote: The model shows high recall (0.868) but lower precision (0.588),")
    print("meaning it tends to over-predict topic changes.")


if __name__ == "__main__":
    test_model()
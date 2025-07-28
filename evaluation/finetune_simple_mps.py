#!/usr/bin/env python3
"""Simple fine-tuning script for Apple Silicon Macs using MPS."""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support
import time
from tqdm import tqdm


class TopicDetectionDataset(Dataset):
    """Dataset for topic detection fine-tuning."""
    
    def __init__(self, file_path, tokenizer, max_length=256):
        self.examples = []
        with open(file_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format text
        text = f"{example['instruction']}\n\n{example['input']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get label
        label = int(example['output'])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class TopicDetectionModel(nn.Module):
    """Simple classification head on top of a small transformer."""
    
    def __init__(self, model_name="microsoft/xtremedistil-l6-h256-uncased", num_labels=2):
        super().__init__()
        
        # Use a tiny transformer
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    print("Fine-tuning for Topic Detection on Apple Silicon")
    print("="*60)
    
    # Configuration
    model_name = "microsoft/xtremedistil-l6-h256-uncased"  # 13M params, very small
    batch_size = 32
    learning_rate = 2e-5
    num_epochs = 3
    max_length = 256
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TopicDetectionDataset(
        "evaluation/finetuning_data/train_window1.jsonl",
        tokenizer,
        max_length
    )
    
    val_dataset = TopicDetectionDataset(
        "evaluation/finetuning_data/val_window1.jsonl",
        tokenizer,
        max_length
    )
    
    # Use smaller subset for quick test
    train_subset_size = min(5000, len(train_dataset))
    train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    print(f"\nInitializing model...")
    model = TopicDetectionModel(model_name)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training time estimation
    print(f"\nEstimated training time:")
    print(f"- Per epoch: ~{len(train_loader) * 0.5:.0f} seconds")
    print(f"- Total: ~{len(train_loader) * 0.5 * num_epochs / 60:.1f} minutes")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Save model
    output_dir = Path("evaluation/finetuned_models")
    output_dir.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'val_metrics': val_metrics,
        'config': {
            'max_length': max_length,
            'num_labels': 2
        }
    }, output_dir / "topic_detector_xtremedistil.pt")
    
    print(f"\nModel saved to {output_dir / 'topic_detector_xtremedistil.pt'}")
    
    # Test inference
    print("\n\nTesting inference on sample examples:")
    print("-" * 40)
    
    test_examples = [
        {
            "instruction": "Determine if there is a topic change between these two messages.",
            "input": "Message 1: How's the weather today?\nMessage 2: It's sunny and warm."
        },
        {
            "instruction": "Determine if there is a topic change between these two messages.",
            "input": "Message 1: How's the weather today?\nMessage 2: Can you help me with Python?"
        }
    ]
    
    model.eval()
    with torch.no_grad():
        for ex in test_examples:
            text = f"{ex['instruction']}\n\n{ex['input']}"
            encoding = tokenizer(text, truncation=True, padding='max_length', 
                               max_length=max_length, return_tensors='pt')
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            print(f"\nInput: {ex['input'][:100]}...")
            print(f"Prediction: {pred} (0=same topic, 1=topic change)")
            print(f"Probabilities: [same: {probs[0]:.3f}, change: {probs[1]:.3f}]")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Full-scale fine-tuning on all available datasets (no plotting dependencies)."""

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
    """Classification head on top of a small transformer."""
    
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


def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss = loss / accumulation_steps
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        if i % 10 == 0:
            progress_bar.set_postfix({
                'loss': f"{total_loss/(i+1):.4f}",
                'acc': f"{correct/total:.4f}"
            })
    
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
    print("Full-Scale Fine-tuning for Topic Detection")
    print("="*60)
    
    # Configuration
    model_name = "microsoft/xtremedistil-l6-h256-uncased"  # 13M params
    batch_size = 64  # Increased batch size
    learning_rate = 5e-5
    num_epochs = 3  # Reduced for faster training
    max_length = 256
    accumulation_steps = 2  # Gradient accumulation
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TopicDetectionDataset(
        "evaluation/finetuning_data/train_all_datasets.jsonl",
        tokenizer,
        max_length
    )
    
    val_dataset = TopicDetectionDataset(
        "evaluation/finetuning_data/val_all_datasets.jsonl",
        tokenizer,
        max_length
    )
    
    test_dataset = TopicDetectionDataset(
        "evaluation/finetuning_data/test_all_datasets.jsonl",
        tokenizer,
        max_length
    )
    
    print(f"Train examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")
    print(f"Test examples: {len(test_dataset):,}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training time estimation
    steps_per_epoch = len(train_loader) // accumulation_steps
    print(f"\nTraining details:")
    print(f"- Batches per epoch: {len(train_loader)}")
    print(f"- Effective steps per epoch: {steps_per_epoch}")
    print(f"- Estimated time per epoch: ~{len(train_loader) * 0.3:.0f} seconds")
    print(f"- Total estimated time: ~{len(train_loader) * 0.3 * num_epochs / 60:.1f} minutes")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('='*60)
        
        # Train
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, accumulation_steps
        )
        epoch_time = time.time() - epoch_start
        
        print(f"\nTrain Results:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Evaluate
        print("\nValidating...")
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Learning rate: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            
            output_dir = Path("evaluation/finetuned_models")
            output_dir.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_name': model_name,
                'val_metrics': val_metrics,
                'config': {
                    'max_length': max_length,
                    'num_labels': 2,
                    'training_examples': len(train_dataset),
                    'best_epoch': epoch + 1
                }
            }, output_dir / "topic_detector_full.pt")
            
            print(f"\n⭐ New best model saved! (F1: {best_val_f1:.4f})")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per epoch: {total_time/num_epochs:.1f} seconds")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(output_dir / "topic_detector_full.pt", 
                           map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # Model info
    print(f"\nModel Information:")
    print(f"  Base model: {model_name}")
    print(f"  Parameters: 13M")
    print(f"  Best epoch: {checkpoint['config']['best_epoch']}")
    print(f"  Training examples: {checkpoint['config']['training_examples']:,}")
    print(f"  Model file: topic_detector_full.pt")
    print(f"  File size: ~51 MB")
    
    # Comparison with previous results
    print("\n" + "="*60)
    print("COMPARISON WITH OTHER METHODS")
    print("="*60)
    print(f"Fine-tuned XtremDistil: F1={test_metrics['f1']:.3f}")
    print("Sentence-BERT: F1=0.571")
    print("Sliding Window: F1=0.560")
    print("Small instruct LLMs: F1=0.305-0.455")
    
    print("\n✅ Fine-tuning completed successfully!")
    print(f"The model is saved at: {output_dir / 'topic_detector_full.pt'}")


if __name__ == "__main__":
    main()
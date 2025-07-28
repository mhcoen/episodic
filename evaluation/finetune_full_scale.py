#!/usr/bin/env python3
"""Full-scale fine-tuning on all available datasets."""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
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
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Same Topic', 'Topic Change'],
                yticklabels=['Same Topic', 'Topic Change'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    print("Full-Scale Fine-tuning for Topic Detection")
    print("="*60)
    
    # Configuration
    model_name = "microsoft/xtremedistil-l6-h256-uncased"  # 13M params
    batch_size = 64  # Increased batch size
    learning_rate = 5e-5
    num_epochs = 5  # More epochs
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
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    
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
    print(f"\nEstimated training time:")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Per epoch: ~{steps_per_epoch * 0.5:.0f} seconds")
    print(f"- Total: ~{steps_per_epoch * 0.5 * num_epochs / 60:.1f} minutes")
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, accumulation_steps
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
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
                    'num_labels': 2
                }
            }, output_dir / "topic_detector_best.pt")
            
            print(f"  → New best model saved! (F1: {best_val_f1:.4f})")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    
    # Load best model
    checkpoint = torch.load(output_dir / "topic_detector_best.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # Save confusion matrix
    plot_confusion_matrix(
        test_metrics['labels'], 
        test_metrics['predictions'],
        output_dir / "confusion_matrix.png"
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'])
    plt.title('Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png")
    plt.close()
    
    print(f"\nModel and plots saved to {output_dir}")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Test set F1: {test_metrics['f1']:.4f}")
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Model size: 13M parameters")
    print("\nThe model is ready for deployment!")


if __name__ == "__main__":
    main()
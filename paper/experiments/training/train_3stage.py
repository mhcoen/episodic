"""
Two-stage training: splice pretrain → benchmark fine-tune.

Stage 1: Pretrain on splice boundaries (real text, synthetic labels)
  - Teaches model to detect major semantic transitions
  - Uses DailyDialog + MultiWOZ text with concatenation boundaries

Stage 2: Fine-tune on benchmark datasets (real annotations)
  - Anchors to real annotation style
  - Uses SuperSeg, DialSeg711, TIAGE

Stage 3: Calibration
  - Temperature scaling on held-out data
  - Produces calibrated boundary probabilities
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
from tqdm import tqdm
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256

    # Stage 1: Pretrain
    pretrain_epochs: int = 3
    pretrain_lr: float = 2e-5
    pretrain_batch_size: int = 16

    # Stage 2: Fine-tune
    finetune_epochs: int = 5
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 16

    # General
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42

    # Calibration
    calibration_bins: int = 15


class BoundaryDataset(Dataset):
    """Dataset for boundary detection training."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer: DistilBertTokenizer,
        max_length: int = 256
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Format window as text
        window_text = self._format_window(ex["window"])
        current_text = ex["current_message"]["content"]

        # Combine: [CLS] window [SEP] current [SEP]
        encoding = self.tokenizer(
            window_text,
            current_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(ex["label"], dtype=torch.long),
            "conv_id": ex.get("conversation_id", "unknown"),
            "source": ex.get("source", "unknown"),
            "splice_type": ex.get("splice_type", "unknown"),
        }

    def _format_window(self, window: List[Dict]) -> str:
        """Format window messages as text."""
        parts = []
        for msg in window:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{role}: {content}")
        return " ".join(parts)


def load_splice_data(path: Path) -> Dict[str, List[Dict]]:
    """Load splice boundary training data."""
    with open(path) as f:
        data = json.load(f)
    return {
        "train": data["train"],
        "val": data["val"],
        "test": data["test"],
    }


def load_benchmark_data(datasets_path: Path) -> Dict[str, List[Dict]]:
    """Load and combine benchmark datasets for fine-tuning."""
    train_examples = []
    val_examples = []
    test_examples = []

    # Load each benchmark dataset
    for dataset_name in ["superseg", "dialseg711", "tiage"]:
        dataset_path = datasets_path / dataset_name

        for split, target_list in [
            ("train", train_examples),
            ("val", val_examples),
            ("test", test_examples)
        ]:
            # Try different file patterns
            for pattern in [
                f"segmentation_file_{split}.json",
                "segmentation_file_test.json" if split == "test" else None,
            ]:
                if pattern is None:
                    continue
                file_path = dataset_path / pattern
                if file_path.exists():
                    examples = convert_benchmark_to_windows(file_path, dataset_name)
                    target_list.extend(examples)
                    logger.info(f"Loaded {len(examples)} from {dataset_name}/{pattern}")
                    break

    # If no validation data, split from train (10%)
    if len(val_examples) == 0 and len(train_examples) > 0:
        import random
        random.seed(42)
        random.shuffle(train_examples)
        split_idx = int(len(train_examples) * 0.9)
        val_examples = train_examples[split_idx:]
        train_examples = train_examples[:split_idx]
        logger.info(f"Created val split: {len(val_examples)} from train")

    return {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }


def convert_benchmark_to_windows(
    file_path: Path,
    dataset_name: str,
    window_size: int = 8
) -> List[Dict]:
    """Convert benchmark format to window training format."""
    with open(file_path) as f:
        data = json.load(f)

    examples = []

    # Handle different formats
    dial_data = data.get("dial_data", data)

    for source_name, dialogs in dial_data.items():
        if not isinstance(dialogs, list):
            continue

        for dialog in dialogs:
            turns = dialog.get("turns", [])
            if len(turns) < 4:
                continue

            # Extract boundaries from topic changes
            boundaries = set()
            prev_topic = None
            user_idx = 0

            for turn in turns:
                if turn.get("role") == "user":
                    topic = turn.get("topic_id") or turn.get("topic_name")
                    if prev_topic is not None and topic != prev_topic:
                        boundaries.add(user_idx)
                    prev_topic = topic
                    user_idx += 1

            # Create window examples
            messages = [
                {"role": t["role"], "content": t.get("utterance", t.get("text", ""))}
                for t in turns
            ]

            user_idx = 0
            for i, msg in enumerate(messages):
                if msg["role"] == "user" and user_idx > 0:
                    window_start = max(0, i - window_size)
                    window = messages[window_start:i]

                    examples.append({
                        "window": window,
                        "current_message": msg,
                        "label": 1 if user_idx in boundaries else 0,
                        "conversation_id": dialog.get("dial_id", "unknown"),
                        "turn_index": user_idx,
                        "source": dataset_name,
                    })

                if msg["role"] == "user":
                    user_idx += 1

    return examples


def compute_windowed_f1(pred_boundaries: List[int], true_boundaries: List[int],
                        window: int = 1, total_positions: int = 0) -> Dict[str, float]:
    """
    Compute windowed F1 - a prediction within ±window of true boundary counts as hit.
    """
    if not true_boundaries:
        # No true boundaries - check false positive rate
        fp = len(pred_boundaries)
        return {"w_precision": 0.0 if fp > 0 else 1.0, "w_recall": 1.0, "w_f1": 0.0 if fp > 0 else 1.0}

    if not pred_boundaries:
        return {"w_precision": 0.0, "w_recall": 0.0, "w_f1": 0.0}

    # Count hits (pred within window of any true)
    hits = 0
    matched_true = set()
    for pred in pred_boundaries:
        for true in true_boundaries:
            if abs(pred - true) <= window and true not in matched_true:
                hits += 1
                matched_true.add(true)
                break

    precision = hits / len(pred_boundaries) if pred_boundaries else 0
    recall = hits / len(true_boundaries) if true_boundaries else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"w_precision": precision, "w_recall": recall, "w_f1": f1}


def compute_bor(pred_count: int, true_count: int) -> float:
    """
    Boundary Over-segmentation Ratio.
    BOR = predicted_boundaries / true_boundaries
    BOR=1 is ideal, >1 means over-segmentation, <1 means under-segmentation.
    """
    if true_count == 0:
        return float('inf') if pred_count > 0 else 1.0
    return pred_count / true_count


def compute_salience_concentration(probabilities: np.ndarray, top_k: int = 10) -> float:
    """
    Compute salience peak concentration - what fraction of total probability mass
    is in the top-k predictions. Higher = more decisive/peaky predictions.
    """
    if len(probabilities) < top_k:
        return 0.0
    total = np.sum(probabilities)
    if total == 0:
        return 0.0
    sorted_probs = np.sort(probabilities)[::-1]
    top_k_sum = np.sum(sorted_probs[:top_k])
    return top_k_sum / total


def compute_metrics(predictions: np.ndarray, labels: np.ndarray,
                    conv_ids: List[str] = None, threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    pred_binary = (predictions > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average="binary", zero_division=0
    )

    # Basic metrics
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_rate": np.mean(pred_binary),
        "true_rate": np.mean(labels),
    }

    # BOR
    pred_count = np.sum(pred_binary)
    true_count = np.sum(labels)
    metrics["bor"] = compute_bor(pred_count, true_count)

    # Salience concentration
    metrics["salience_concentration"] = compute_salience_concentration(predictions)

    # Windowed F1 (if we can reconstruct boundaries per conversation)
    if conv_ids is not None:
        # Group by conversation and compute per-conv windowed metrics
        from collections import defaultdict
        conv_preds = defaultdict(list)
        conv_labels = defaultdict(list)
        conv_positions = defaultdict(list)

        for i, (pred, label, cid) in enumerate(zip(pred_binary, labels, conv_ids)):
            conv_preds[cid].append(pred)
            conv_labels[cid].append(label)
            conv_positions[cid].append(i)

        # Compute windowed F1 across all conversations
        all_pred_boundaries = []
        all_true_boundaries = []
        offset = 0
        for cid in conv_preds:
            for i, (p, l) in enumerate(zip(conv_preds[cid], conv_labels[cid])):
                if p == 1:
                    all_pred_boundaries.append(offset + i)
                if l == 1:
                    all_true_boundaries.append(offset + i)
            offset += len(conv_preds[cid])

        w_metrics = compute_windowed_f1(all_pred_boundaries, all_true_boundaries, window=1)
        metrics.update(w_metrics)

        w2_metrics = compute_windowed_f1(all_pred_boundaries, all_true_boundaries, window=2)
        metrics["w2_f1"] = w2_metrics["w_f1"]

    return metrics


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


def calibrate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Learn temperature parameter for calibration."""
    model.eval()

    # Collect logits and labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 1]  # Boundary class logit

            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    # Optimize temperature
    temp_model = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_model(logits)
        loss = criterion(scaled_logits, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)

    return temp_model.temperature.item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    desc: str = "Training"
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
    return_details: bool = False
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model with comprehensive metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_conv_ids = []
    all_sources = []
    all_splice_types = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 1] / temperature
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_conv_ids.extend(batch.get("conv_id", ["unknown"] * len(labels)))
            all_sources.extend(batch.get("source", ["unknown"] * len(labels)))
            all_splice_types.extend(batch.get("splice_type", ["unknown"] * len(labels)))

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # Compute overall metrics with conversation IDs for windowed F1
    metrics = compute_metrics(preds, labels, conv_ids=all_conv_ids)

    if return_details:
        return metrics, preds, labels, all_conv_ids, all_sources, all_splice_types

    return metrics, preds, labels


def evaluate_by_source(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """Evaluate model separately for each data source."""
    metrics, preds, labels, conv_ids, sources, _ = evaluate(
        model, dataloader, device, temperature, return_details=True
    )

    # Group by source
    source_metrics = {}
    unique_sources = set(sources)

    for source in unique_sources:
        mask = np.array([s == source for s in sources])
        if np.sum(mask) == 0:
            continue

        source_preds = preds[mask]
        source_labels = labels[mask]
        source_conv_ids = [c for c, m in zip(conv_ids, mask) if m]

        source_metrics[source] = compute_metrics(
            source_preds, source_labels, conv_ids=source_conv_ids
        )

    return source_metrics


def evaluate_negative_control(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 1.0
) -> Dict[str, float]:
    """
    Evaluate on 'single' (no-splice) examples.
    These should have very low predicted boundary rate.
    """
    metrics, preds, labels, _, _, splice_types = evaluate(
        model, dataloader, device, temperature, return_details=True
    )

    # Filter to single-segment examples only
    mask = np.array([t == "single" for t in splice_types])
    if np.sum(mask) == 0:
        return {"neg_control_pred_rate": -1, "neg_control_count": 0}

    single_preds = preds[mask]
    single_labels = labels[mask]

    pred_binary = (single_preds > 0.5).astype(int)
    pred_rate = np.mean(pred_binary)

    return {
        "neg_control_pred_rate": pred_rate,
        "neg_control_count": int(np.sum(mask)),
        "neg_control_true_rate": np.mean(single_labels),  # Should be 0 for single segments
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pretrain", action="store_true",
                       help="Skip pretraining stage")
    parser.add_argument("--pretrain-only", action="store_true",
                       help="Only run pretraining")
    parser.add_argument("--output-dir", type=str,
                       default=None,
                       help="Output directory (default: ../models)")
    parser.add_argument("--synthetic-data", type=str,
                       default=None,
                       help="Path to synthetic splice data directory")
    parser.add_argument("--datasets-dir", type=str,
                       default=None,
                       help="Path to benchmark datasets directory")
    args = parser.parse_args()

    config = TrainingConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("mps" if torch.backends.mps.is_available()
                         else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup paths
    base_path = Path(__file__).parent
    experiments_dir = base_path.parent

    # Output directory for models
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiments_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic data directory
    if args.synthetic_data:
        synthetic_dir = Path(args.synthetic_data)
    else:
        synthetic_dir = experiments_dir / "data" / "synthetic"

    # Benchmark datasets directory
    if args.datasets_dir:
        datasets_dir = Path(args.datasets_dir)
    else:
        # Default: look for datasets in project root
        datasets_dir = experiments_dir.parent.parent / "datasets"

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)

    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2
    )
    model.to(device)

    # ========================================
    # Stage 1: Pretrain on splice boundaries
    # ========================================
    if not args.skip_pretrain:
        logger.info("="*60)
        logger.info("Stage 1: Pretraining on splice boundaries")
        logger.info("="*60)

        # Load synthetic splice data
        splice_train = synthetic_dir / "synthetic_large_train.json"
        splice_val = synthetic_dir / "synthetic_large_val.json"
        splice_test = synthetic_dir / "synthetic_large_test.json"

        if not splice_train.exists():
            logger.error(f"Synthetic training data not found: {splice_train}")
            logger.error("Run: python data/download_datasets.py first, or provide --synthetic-data")
            return

        # Load split files directly
        with open(splice_train) as f:
            train_data = json.load(f)
        with open(splice_val) as f:
            val_data = json.load(f)

        # Handle different data formats
        if isinstance(train_data, dict) and "train" in train_data:
            splice_data = train_data
        else:
            splice_data = {"train": train_data, "val": val_data}
        logger.info(f"Splice data: {len(splice_data['train'])} train, "
                   f"{len(splice_data['val'])} val")

        train_dataset = BoundaryDataset(splice_data["train"], tokenizer, config.max_length)
        val_dataset = BoundaryDataset(splice_data["val"], tokenizer, config.max_length)

        train_loader = DataLoader(train_dataset, batch_size=config.pretrain_batch_size,
                                 shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.pretrain_batch_size)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.pretrain_lr,
            weight_decay=config.weight_decay
        )
        total_steps = len(train_loader) * config.pretrain_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        best_f1 = 0
        for epoch in range(config.pretrain_epochs):
            loss = train_epoch(model, train_loader, optimizer, scheduler, device,
                             desc=f"Pretrain epoch {epoch+1}/{config.pretrain_epochs}")

            # Comprehensive evaluation
            metrics, _, _ = evaluate(model, val_loader, device)

            # Log key metrics: W-F1, BOR, salience concentration
            w_f1 = metrics.get('w_f1', metrics['f1'])
            bor = metrics.get('bor', -1)
            salience = metrics.get('salience_concentration', -1)

            logger.info(f"Epoch {epoch+1}: loss={loss:.4f}, "
                       f"F1={metrics['f1']:.3f}, W-F1={w_f1:.3f}, "
                       f"BOR={bor:.2f}, Salience={salience:.3f}")

            # Negative control: check prediction rate on single-segment examples
            neg_ctrl = evaluate_negative_control(model, val_loader, device)
            if neg_ctrl['neg_control_count'] > 0:
                logger.info(f"  Negative control (single segments): "
                           f"pred_rate={neg_ctrl['neg_control_pred_rate']:.3f} "
                           f"(n={neg_ctrl['neg_control_count']}, should be low)")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": vars(config),
                    "stage": "pretrain",
                    "best_f1": best_f1,
                    "w_f1": w_f1,
                    "bor": bor,
                }, output_dir / "pretrained_splice.pt")

        logger.info(f"Best pretrain F1: {best_f1:.3f}")

        if args.pretrain_only:
            logger.info("Pretrain only - stopping here")
            return

    # ========================================
    # Stage 2: Fine-tune on benchmark datasets
    # ========================================
    logger.info("="*60)
    logger.info("Stage 2: Fine-tuning on benchmark datasets")
    logger.info("="*60)

    # Load pretrained if skipped
    if args.skip_pretrain:
        pretrain_path = output_dir / "pretrained_splice.pt"
        if pretrain_path.exists():
            checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded pretrained model")
        else:
            logger.warning("No pretrained model found, starting from scratch")

    # Load benchmark data
    benchmark_data = load_benchmark_data(datasets_dir)
    logger.info(f"Benchmark data: {len(benchmark_data['train'])} train, "
               f"{len(benchmark_data['val'])} val, {len(benchmark_data['test'])} test")

    if len(benchmark_data["train"]) == 0:
        logger.error("No benchmark training data found!")
        return

    train_dataset = BoundaryDataset(benchmark_data["train"], tokenizer, config.max_length)
    val_dataset = BoundaryDataset(benchmark_data["val"], tokenizer, config.max_length)
    test_dataset = BoundaryDataset(benchmark_data["test"], tokenizer, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.finetune_batch_size,
                             shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.finetune_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.finetune_batch_size)

    # Optimizer (lower LR for fine-tuning)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.finetune_lr,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.finetune_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_f1 = 0
    for epoch in range(config.finetune_epochs):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device,
                         desc=f"Finetune epoch {epoch+1}/{config.finetune_epochs}")

        # Overall metrics
        metrics, _, _ = evaluate(model, val_loader, device)
        w_f1 = metrics.get('w_f1', metrics['f1'])
        bor = metrics.get('bor', -1)
        salience = metrics.get('salience_concentration', -1)

        logger.info(f"Epoch {epoch+1}: loss={loss:.4f}, "
                   f"F1={metrics['f1']:.3f}, W-F1={w_f1:.3f}, "
                   f"BOR={bor:.2f}, Salience={salience:.3f}")

        # Per-dataset metrics (check if TIAGE improves without wrecking others)
        if epoch == config.finetune_epochs - 1 or metrics["f1"] > best_f1:
            source_metrics = evaluate_by_source(model, val_loader, device)
            for source, smetrics in source_metrics.items():
                s_f1 = smetrics.get('f1', 0)
                s_wf1 = smetrics.get('w_f1', s_f1)
                s_bor = smetrics.get('bor', -1)
                logger.info(f"  {source}: F1={s_f1:.3f}, W-F1={s_wf1:.3f}, BOR={s_bor:.2f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(config),
                "stage": "finetune",
                "best_f1": best_f1,
                "w_f1": w_f1,
                "bor": bor,
            }, output_dir / "finetuned_benchmark.pt")

    logger.info(f"Best fine-tune F1: {best_f1:.3f}")

    # Load best model
    if (output_dir / "finetuned_benchmark.pt").exists():
        checkpoint = torch.load(output_dir / "finetuned_benchmark.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("No best model found, using current weights")

    # ========================================
    # Stage 3: Calibration
    # ========================================
    logger.info("="*60)
    logger.info("Stage 3: Temperature calibration")
    logger.info("="*60)

    temperature = calibrate_model(model, val_loader, device)
    logger.info(f"Learned temperature: {temperature:.4f}")

    # Final evaluation with calibration - overall
    metrics, preds, labels = evaluate(model, test_loader, device, temperature)
    w_f1 = metrics.get('w_f1', metrics['f1'])
    bor = metrics.get('bor', -1)
    logger.info(f"Test (calibrated): F1={metrics['f1']:.3f}, W-F1={w_f1:.3f}, BOR={bor:.2f}")

    # Per-dataset final metrics - check BOR stability across datasets
    logger.info("Per-dataset test results (checking BOR stability):")
    source_metrics = evaluate_by_source(model, test_loader, device, temperature)
    bor_values = []
    for source, smetrics in source_metrics.items():
        s_f1 = smetrics.get('f1', 0)
        s_wf1 = smetrics.get('w_f1', s_f1)
        s_bor = smetrics.get('bor', -1)
        if s_bor > 0:
            bor_values.append(s_bor)
        logger.info(f"  {source}: F1={s_f1:.3f}, W-F1={s_wf1:.3f}, BOR={s_bor:.2f}")

    # BOR stability check
    if len(bor_values) > 1:
        bor_std = np.std(bor_values)
        bor_mean = np.mean(bor_values)
        logger.info(f"BOR stability: mean={bor_mean:.2f}, std={bor_std:.2f} "
                   f"({'STABLE' if bor_std < 0.3 else 'UNSTABLE'})")

    # Save final model with calibration
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(config),
        "temperature": temperature,
        "test_metrics": metrics,
    }, output_dir / "final_calibrated.pt")

    logger.info(f"\nFinal model saved to: {output_dir / 'final_calibrated.pt'}")


if __name__ == "__main__":
    main()

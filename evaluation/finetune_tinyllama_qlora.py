#!/usr/bin/env python3
"""Fine-tune TinyLlama for topic boundary detection using QLoRA."""

import json
import torch
from pathlib import Path
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def load_jsonl_dataset(file_path: Path) -> List[Dict]:
    """Load JSONL dataset."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_prompt(example: Dict) -> str:
    """Format example into prompt for TinyLlama."""
    # TinyLlama uses ChatML format
    prompt = f"""<|system|>
{example['instruction']}</s>
<|user|>
{example['input']}</s>
<|assistant|>
{example['output']}</s>"""
    return prompt


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples."""
    # Format prompts
    prompts = [format_prompt(ex) for ex in examples['example']]
    
    # Tokenize
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Set labels (same as input_ids for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # Mask padding tokens in labels
    model_inputs["labels"][model_inputs["labels"] == tokenizer.pad_token_id] = -100
    
    return model_inputs


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    # For generation tasks, we'd need to decode and parse
    # For now, return dummy metrics
    return {"eval_loss": 0.0}


def main():
    print("Fine-tuning TinyLlama for Topic Detection with QLoRA")
    print("="*60)
    
    # Configuration
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./tinyllama-topic-detection-qlora"
    
    # Load datasets
    print("\nLoading datasets...")
    train_data = load_jsonl_dataset(Path("evaluation/finetuning_data/train_window1.jsonl"))
    val_data = load_jsonl_dataset(Path("evaluation/finetuning_data/val_window1.jsonl"))
    
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Sample for testing
    train_data = train_data[:1000]  # Use subset for testing
    val_data = val_data[:100]
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list([{"example": ex} for ex in train_data])
    val_dataset = Dataset.from_list([{"example": ex} for ex in val_data])
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    print("\nLoading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Low rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Get PEFT model
    print("\nApplying LoRA...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["example"]
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["example"]
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=100,
        logging_steps=25,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    print("This is a demonstration setup. For actual training:")
    print("1. Use the full dataset (remove subsetting)")
    print("2. Adjust hyperparameters based on GPU memory")
    print("3. Implement proper metric computation")
    print("4. Save the LoRA adapters for inference")
    
    # Uncomment to actually train
    # trainer.train()
    
    print("\n\nNext steps for deployment:")
    print("1. Merge LoRA weights with base model")
    print("2. Convert to ONNX or other optimized format")
    print("3. Deploy with Ollama or similar inference engine")
    print("4. Test on window_size=1 and window_size=3 tasks")


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import peft
        import bitsandbytes
        import accelerate
    except ImportError:
        print("Please install required packages:")
        print("pip install transformers peft accelerate bitsandbytes")
        exit(1)
    
    main()
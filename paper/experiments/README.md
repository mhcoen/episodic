# Experiment Replication

This directory contains scripts to reproduce the experiments in:

**"When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation"**

## Dataset Licensing

Due to dataset licensing restrictions, we do not redistribute raw dialogue data.
This repository provides scripts to automatically download and preprocess all
datasets used in the paper.

**Redistributable artifacts included:**
- Synthetic splice training data (`data/synthetic/`)
- Trained model checkpoint (`models/final_calibrated.pt`)
- Figure generation scripts

## Requirements

```bash
pip install torch transformers scikit-learn numpy matplotlib tqdm
```

## Directory Structure

```
experiments/
├── data/
│   ├── download_datasets.py    # Download evaluation datasets
│   ├── preprocess.py           # Convert to canonical format
│   └── synthetic/              # Synthetic training data (included)
├── training/
│   └── train_3stage.py         # 3-stage training pipeline
├── evaluation/
│   └── compute_metrics.py      # Compute Tables 1, 4, 5
├── figures/
│   ├── generate_figure1.py     # Figure 1: Granularity mismatch
│   └── generate_figure2.py     # Figure 2: Adaptive selection
└── models/
    └── final_calibrated.pt     # Trained model (included)
```

## Quick Start: Reproduce Tables

If you just want to reproduce the paper's tables using the pre-trained model:

```bash
# 1. Download datasets
python data/download_datasets.py

# 2. Compute metrics (Tables 1, 4, 5)
python evaluation/compute_metrics.py
```

## Full Reproduction: Train from Scratch

To reproduce the full training pipeline:

```bash
# 1. Download datasets
python data/download_datasets.py

# 2. Preprocess to canonical format
python data/preprocess.py

# 3. Train model (3 stages)
python training/train_3stage.py

# 4. Evaluate
python evaluation/compute_metrics.py

# 5. Generate figures
python figures/generate_figure1.py
python figures/generate_figure2.py
```

## Training Pipeline Details

The training script (`training/train_3stage.py`) implements:

**Stage 1: Synthetic Pretraining** (Table 2)
- Trains on splice boundary detection
- Data: `data/synthetic/synthetic_large_train.json`
- Output: `models/pretrained_splice.pt`

**Stage 2: Supervised Fine-Tuning** (Table 3)
- Fine-tunes on benchmark datasets
- Data: DialSeg711, SuperDialseg, TIAGE
- Output: `models/finetuned_benchmark.pt`

**Stage 3: Temperature Calibration**
- Applies temperature scaling for calibrated probabilities
- Output: `models/final_calibrated.pt`

### Training Options

```bash
# Run all stages
python training/train_3stage.py

# Skip pretraining (use existing checkpoint)
python training/train_3stage.py --skip-pretrain

# Pretrain only
python training/train_3stage.py --pretrain-only

# Custom paths
python training/train_3stage.py \
    --synthetic-data /path/to/synthetic \
    --datasets-dir /path/to/datasets \
    --output-dir /path/to/models
```

## Datasets

The following datasets are used for evaluation:

| Dataset | Source | Citation |
|---------|--------|----------|
| DialSeg711 | [GitHub](https://github.com/Coldog2333/SuperDialseg) | Liu et al., EMNLP 2022 |
| SuperDialseg | [GitHub](https://github.com/Coldog2333/SuperDialseg) | Liu et al., EMNLP 2023 |
| TIAGE | [GitHub](https://github.com/HaoSunTJU/TIAGE) | Sun et al., COLING 2020 |
| DailyDialog | [Website](http://yanran.li/dailydialog) | Li et al., IJCNLP 2017 |
| MultiWOZ | [GitHub](https://github.com/budzianowski/multiwoz) | Budzianowski et al., EMNLP 2018 |
| Taskmaster | [GitHub](https://github.com/google-research-datasets/Taskmaster) | Byrne et al., SIGDIAL 2019 |
| Topical-Chat | [GitHub](https://github.com/alexa/Topical-Chat) | Gopalakrishnan et al., 2019 |
| QMSum | [GitHub](https://github.com/Yale-LILY/QMSum) | Zhong et al., NAACL 2021 |

To download all datasets:
```bash
python data/download_datasets.py

# Or download specific datasets
python data/download_datasets.py --datasets dialseg711 superseg tiage

# List sources only (no download)
python data/download_datasets.py --list-sources
```

## Figures

**Figure 1: Granularity Mismatch**
```bash
python figures/generate_figure1.py
# Output: ../figures/granularity_mismatch.{png,pdf}
```

**Figure 2: Adaptive Boundary Selection**
```bash
python figures/generate_figure2.py
# Output: ../figures/adaptive_commitment_granularity.png
```

## Metrics

The evaluation script computes:
- **W-F1**: Window-tolerant F1 (tolerance = 1)
- **BOR**: Boundary Oversegmentation Ratio
- **Purity**: Segment coherence (predicted segments within gold)
- **Coverage**: Segment coherence (gold segments within predicted)

## Citation

```bibtex
@article{coen2025f1fails,
  title={When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation},
  author={Coen, Michael H.},
  year={2025}
}
```

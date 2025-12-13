# Paper: When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation

This directory contains the LaTeX source, figures, and experiment scripts
used to produce the results reported in the accompanying paper.

The paper is part of the Episodic project but can be built and evaluated
independently.

## Structure

```
paper/
├── topicDetection.tex      # LaTeX source
├── references.bib          # Bibliography
├── figures/                # Generated figures (PNG/PDF)
└── experiments/            # Reproducibility package
    ├── README.md           # Detailed reproduction instructions
    ├── data/               # Dataset download and preprocessing
    ├── training/           # Model training scripts
    ├── evaluation/         # Metrics computation
    ├── figures/            # Figure generation scripts
    └── models/             # Trained model checkpoint
```

## Data

Due to dataset licensing restrictions, raw dialogue datasets are not
redistributed. Scripts are provided to download and preprocess all datasets
from their original sources:

```bash
python experiments/data/download_datasets.py
```

## Reproducibility

All tables and figures in the paper can be reproduced using the scripts
in `experiments/`.

**Quick start (using pre-trained model):**
```bash
# Download datasets
python experiments/data/download_datasets.py

# Reproduce Tables 1, 4, 5
python experiments/evaluation/compute_metrics.py

# Reproduce figures
python experiments/figures/generate_figure1.py
python experiments/figures/generate_figure2.py
```

**Full reproduction (train from scratch):**
```bash
python experiments/training/train_3stage.py
```

See `experiments/README.md` for detailed instructions.

## Building the Paper

```bash
pdflatex topicDetection.tex
bibtex topicDetection
pdflatex topicDetection.tex
pdflatex topicDetection.tex
```

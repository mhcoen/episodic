## Summary Table: Centroid Drift Diagnostics

| Corpus | Type | Dialogues | User Turns | Boundaries | Density | Med Seg Len | D1 (best α) | D2 (best α) | D3 (pairwise) | Shuffled | Label |
|--------|------|-----------|------------|------------|---------|-------------|-------------|-------------|---------------|----------|-------|
| superseg | task-oriented | 1322 | 8590 | 3667 | 0.427 | 1.0 | 0.383 (α=0.2) | 0.324 (α=0.2) | 0.475 | 0.462 | poor |
| dialseg711 | task-oriented | 711 | 9710 | 2604 | 0.268 | 3.0 | 0.645 (α=0.2) | 0.566 (α=0.2) | 0.720 | 0.507 | poor |
| tiage | task-oriented | 100 | 882 | 151 | 0.171 | 3.0 | 0.525 (α=0.2) | 0.462 (α=0.2) | 0.590 | 0.549 | poor |
| multiwoz | task-oriented | 765 | 6264 | 1117 | 0.178 | 3.0 | 0.482 (α=0.05) | 0.427 (α=0.2) | 0.544 | 0.479 | poor |
| dailydialog | open-domain | 100 | 1304 | 200 | 0.153 | 4.0 | 0.726 (α=0.2) | 0.650 (α=0.2) | 0.780 | 0.479 | marginal |
| taskmaster | semi-structured | 100 | 3264 | 196 | 0.060 | 11.0 | 0.502 (α=0.2) | 0.563 (α=0.2) | 0.713 | 0.436 | poor |
| topical_chat | open-domain | 100 | 3329 | 196 | 0.059 | 11.0 | 0.352 (α=0.2) | 0.386 (α=0.2) | 0.741 | 0.429 | poor |
| qmsum | semi-structured | 35 | 10369 | 116 | 0.011 | 39.0 | 0.520 (α=0.05) | 0.513 (α=0.05) | 0.471 | 0.524 | poor |
## Per-Corpus Notes

### superseg
- **Path**: `/Users/mhcoen/proj/episodic/datasets/superseg/segmentation_file_test.json`
- **Format**: JSON with `dial_data.superseg-v2` array
- **Boundary encoding**: segmentation_label
- **Type**: task-oriented
- **Quirks**: short segments (<3 user turns)
- **D2 Distribution**: boundary median=0.6458 vs non-boundary p90=0.9734
  - Overlap ratio (median_b / p90_nb): 0.66
- **Eligibility**: Skipped (D2 AUROC too low)

### dialseg711
- **Path**: `/Users/mhcoen/proj/episodic/datasets/dialseg711/segmentation_file_test.json`
- **Format**: JSON with `dial_data.dialseg711` array
- **Boundary encoding**: segmentation_label
- **Type**: task-oriented
- **Quirks**: none
- **D2 Distribution**: boundary median=0.6974 vs non-boundary p90=0.8283
  - Overlap ratio (median_b / p90_nb): 0.84
- **Useful eligibility band**: θ=0.500 → recall=0.91, precision=0.294, rate=0.89

### tiage
- **Path**: `/Users/mhcoen/proj/episodic/datasets/tiage/segmentation_file_test.json`
- **Format**: JSON with `dial_data.tiage` array
- **Boundary encoding**: segmentation_label
- **Type**: task-oriented
- **Quirks**: none
- **D2 Distribution**: boundary median=0.7502 vs non-boundary p90=0.9104
  - Overlap ratio (median_b / p90_nb): 0.82
- **Eligibility**: Skipped (D2 AUROC too low)

### multiwoz
- **Path**: `/Users/mhcoen/proj/episodic/datasets/multiwoz/segmentation_file_test.json`
- **Format**: JSON with `dial_data.multiwoz` array
- **Boundary encoding**: topic_id_change
- **Type**: task-oriented
- **Quirks**: none
- **D2 Distribution**: boundary median=0.6677 vs non-boundary p90=0.8468
  - Overlap ratio (median_b / p90_nb): 0.79
- **Eligibility**: Skipped (D2 AUROC too low)

### dailydialog
- **Path**: `/Users/mhcoen/proj/episodic/datasets/dailydialog/segmentation_file_test.json`
- **Format**: JSON with `dial_data.dailydialog-synthetic` array
- **Boundary encoding**: topic_id_change
- **Type**: open-domain
- **Quirks**: none
- **D2 Distribution**: boundary median=0.8172 vs non-boundary p90=0.8922
  - Overlap ratio (median_b / p90_nb): 0.92
- **Eligibility band**: None found (recall <0.8 or rate ≥0.95)

### taskmaster
- **Path**: `/Users/mhcoen/proj/episodic/datasets/taskmaster/segmentation_file_test.json`
- **Format**: JSON with `dial_data.taskmaster` array
- **Boundary encoding**: topic_id_change
- **Type**: semi-structured
- **Quirks**: none
- **D2 Distribution**: boundary median=0.6910 vs non-boundary p90=0.8185
  - Overlap ratio (median_b / p90_nb): 0.84
- **Useful eligibility band**: θ=0.500 → recall=0.97, precision=0.067, rate=0.90

### topical_chat
- **Path**: `/Users/mhcoen/proj/episodic/datasets/topical_chat/segmentation_file_test.json`
- **Format**: JSON with `dial_data.topical_chat` array
- **Boundary encoding**: topic_id_change
- **Type**: open-domain
- **Quirks**: none
- **D2 Distribution**: boundary median=0.6182 vs non-boundary p90=0.8749
  - Overlap ratio (median_b / p90_nb): 0.71
- **Eligibility**: Skipped (D2 AUROC too low)

### qmsum
- **Path**: `/Users/mhcoen/proj/episodic/datasets/qmsum/segmentation_file_test.json`
- **Format**: JSON with `dial_data.qmsum` array
- **Boundary encoding**: topic_id_change
- **Type**: semi-structured
- **Quirks**: none
- **D2 Distribution**: boundary median=0.5710 vs non-boundary p90=0.7715
  - Overlap ratio (median_b / p90_nb): 0.74
- **Eligibility**: Skipped (D2 AUROC too low)
## Recommendations

### Separability Classification
- **Good** (D2 AUROC ≥0.70): None
- **Marginal** (0.60-0.70): dailydialog
- **Poor** (<0.60): dialseg711, taskmaster, qmsum, tiage, multiwoz, topical_chat, superseg

### Recommended Corpora for Deeper Ablation
1. **dailydialog** (marginal) - D2 AUROC=0.650
2. **dialseg711** (poor) - D2 AUROC=0.566
3. **taskmaster** (poor) - D2 AUROC=0.563

### Overall Assessment
- Centroid drift vs pairwise: 1 wins, 7 losses, 0 ties
- Corpora with useful eligibility band: 2/8

**DO NOT PROCEED**. Centroid drift does not provide sufficient separability advantage over pairwise baseline.
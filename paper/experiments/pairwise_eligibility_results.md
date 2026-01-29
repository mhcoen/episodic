## Pairwise Eligibility Diagnostics: Summary Table

| Corpus | Base Rate | Best θ | Rate | Recall | Precision | Max Recall (θ) | Useful? |
|--------|-----------|--------|------|--------|-----------|----------------|---------|
| superseg | 0.427 | 0.55 | 0.827 | 0.839 | 0.511 | 0.999 (0.05) | no |
| dialseg711 | 0.268 | 0.75 | 0.574 | 0.801 | 0.404 | 1.000 (0.05) | no |
| tiage | 0.171 | 0.70 | 0.693 | 0.815 | 0.227 | 1.000 (0.05) | no |
| multiwoz | 0.178 | 0.65 | 0.722 | 0.805 | 0.226 | 1.000 (0.05) | no |
| dailydialog | 0.153 | 0.80 | 0.542 | 0.890 | 0.273 | 1.000 (0.05) | no |
| taskmaster | 0.060 | 0.80 | 0.519 | 0.847 | 0.101 | 1.000 (0.05) | no |
| topical_chat | 0.059 | 0.80 | 0.450 | 0.811 | 0.109 | 1.000 (0.05) | YES |
| qmsum | 0.011 | 0.50 | 0.826 | 0.819 | 0.011 | 0.966 (0.05) | no |
## Per-Corpus Analysis

### superseg
- **Base boundary rate**: 0.4269 (3667/8590)
- **Best θ for recall≥0.8**: 0.55
  - Eligibility rate: 0.827
  - Recall: 0.839
  - Precision: 0.511
  - Precision / Base rate: 1.20x
- **Useful eligibility band**: NO
- **Notes**: Recall achieved but not selective enough. rate=0.827, precision=0.511

### dialseg711
- **Base boundary rate**: 0.2682 (2604/9710)
- **Best θ for recall≥0.8**: 0.75
  - Eligibility rate: 0.574
  - Recall: 0.801
  - Precision: 0.404
  - Precision / Base rate: 1.51x
- **Useful eligibility band**: NO
- **Notes**: Recall achieved but not selective enough. rate=0.574, precision=0.404

### tiage
- **Base boundary rate**: 0.1712 (151/882)
- **Best θ for recall≥0.8**: 0.70
  - Eligibility rate: 0.693
  - Recall: 0.815
  - Precision: 0.227
  - Precision / Base rate: 1.33x
- **Useful eligibility band**: NO
- **Notes**: Recall achieved but not selective enough. rate=0.693, precision=0.227

### multiwoz
- **Base boundary rate**: 0.1783 (1117/6264)
- **Best θ for recall≥0.8**: 0.65
  - Eligibility rate: 0.722
  - Recall: 0.805
  - Precision: 0.226
  - Precision / Base rate: 1.27x
- **Useful eligibility band**: NO
- **Notes**: Recall achieved but not selective enough. rate=0.722, precision=0.226

### dailydialog
- **Base boundary rate**: 0.1534 (200/1304)
- **Best θ for recall≥0.8**: 0.80
  - Eligibility rate: 0.542
  - Recall: 0.890
  - Precision: 0.273
  - Precision / Base rate: 1.78x
- **Useful eligibility band**: NO
- **Notes**: Recall achieved but not selective enough. rate=0.542, precision=0.273

### taskmaster
- **Base boundary rate**: 0.0600 (196/3264)
- **Best θ for recall≥0.8**: 0.80
  - Eligibility rate: 0.519
  - Recall: 0.847
  - Precision: 0.101
  - Precision / Base rate: 1.68x
- **Useful eligibility band**: NO
- **Notes**: Recall achieved but not selective enough. rate=0.519, precision=0.101

### topical_chat
- **Base boundary rate**: 0.0589 (196/3329)
- **Best θ for recall≥0.8**: 0.80
  - Eligibility rate: 0.450
  - Recall: 0.811
  - Precision: 0.109
  - Precision / Base rate: 1.86x
- **Useful eligibility band**: YES
- **Notes**: θ=0.80: rate=0.450, recall=0.811, precision=0.109

### qmsum
- **Base boundary rate**: 0.0112 (116/10369)
- **Best θ for recall≥0.8**: 0.50
  - Eligibility rate: 0.826
  - Recall: 0.819
  - Precision: 0.011
  - Precision / Base rate: 1.00x
- **Useful eligibility band**: NO
- **Notes**: Recall achieved but not selective enough. rate=0.826, precision=0.011
## Go/No-Go Conclusion

**Useful eligibility bands found**: 1/8

### VERDICT: STOP - ELIGIBILITY GATE NOT SELECTIVE ENOUGH

Only 1 corpora have useful eligibility bands.
Partial success in: topical_chat

The pairwise drift signal does not provide sufficient selectivity for an eligibility gate.

### Breakdown by Corpus Type

**Task-oriented**: 0/4 useful
  - superseg: ✗ not useful
  - dialseg711: ✗ not useful
  - tiage: ✗ not useful
  - multiwoz: ✗ not useful

**Open-domain**: 1/2 useful
  - dailydialog: ✗ not useful
  - topical_chat: ✓ useful

**Semi-structured**: 0/2 useful
  - taskmaster: ✗ not useful
  - qmsum: ✗ not useful
<p align="center">
  <img src="assets/logo.png" alt="AART logo" width="200">
</p>

# AART: Anchor-Aware Residual Translator

Cross-platform proteomics translation between Olink, SomaScan, and Mass Spectrometry platforms.

## Overview

AART is a hybrid method for mapping between plasma proteomics assays. It combines:

1. **Local Anchor Component**: Per-protein Ridge regression on biologically mapped protein pairs between source and target platforms, using only highly correlated anchors
2. **Global Residual Component**: PCA + Ridge model on the residuals, capturing platform-wide patterns not explained by direct anchors
3. **Reliability-Aware Fusion**: Closed-form per-protein weighting that combines anchor and residual predictions based on each anchor's reliability

All components use scikit-learn (Ridge, PCA) — no deep learning dependencies required.

## Installation

```bash
pip install -e .
```

**Requirements**: Python >= 3.10, scikit-learn >= 1.3

## Study Design

<p align="center">
  <img src="assets/study_design.png" alt="Study design" width="700">
</p>

## Supported Platform Pairs

| Platform Pair | Cohort | Config |
|---------------|--------|--------|
| SomaScan ↔ Olink | CKB (n=3,975) | `configs/aart/soma_to_olink.yaml` / `olink_to_soma.yaml` |
| MS ↔ Olink | PEX-LC (n=88) | `configs/aart/ms_to_olink.yaml` |
| MS ↔ SomaScan | GNPC | `configs/aart/ms_to_soma.yaml` |

## Method

<p align="center">
  <img src="assets/AART_method.png" alt="AART method overview" width="600">
</p>

## Project Structure

```
├── assets/                 # Logo, method PDF, study design PDF
├── src/models/aart/        # Core AART model implementation
├── src/utils/              # Shared utilities (config, metrics, preprocessing)
├── configs/aart/           # Configs per platform pair
├── scripts/
│   ├── training/           # Train and evaluate AART
│   ├── application/        # Apply trained AART to new cohorts (e.g. GNPC)
│   └── data_processing/    # Split data and build annotation table
├── tutorial/               # Jupyter notebooks (train + evaluate)
└── data/
    ├── CKB/                # SomaScan ↔ Olink (see data/CKB/README.md)
    ├── PEX_LC/             # MS ↔ Olink (see data/PEX_LC/README.md)
    └── GNPC_MS_SomaScan/   # MS ↔ SomaScan (see data/GNPC_MS_SomaScan/README.md)
```

## Quick Start

1. Place your paired proteomics data in the appropriate `data/` subdirectory
2. Stage the data (split train/test and build annotation):

```bash
python scripts/data_processing/stage_data_aart.py --config configs/aart/soma_to_olink.yaml
```

3. Train AART:

```bash
python scripts/training/train_aart.py --config configs/aart/soma_to_olink.yaml
```

Or run the full pipeline (stage + train) in one command:

```bash
bash scripts/training/run_aart_pipeline.sh
```

AART trains three variants in a single run:
- `direct_anchor` — per-protein Ridge on matched aptamers only (1-to-1 mapping)
- `hybrid_ridge` — PCA-Ridge residual model (no anchor)
- `aart_closed_form_gate` — full AART with reliability-gated combination

## Application: Cross-Cohort Imputation

Apply a trained AART model to impute Olink levels from SomaScan in a new cohort (e.g. GNPC):

1. **Train** AART on CKB paired data (SomaScan → Olink)
2. **Apply** the trained model to GNPC SomaScan to impute Olink levels

```bash
python scripts/application/apply_aart_to_gnpc.py
```

## Tutorials

Step-by-step Jupyter notebooks in [`tutorial/`](tutorial/):

1. **[01_train_aart.ipynb](tutorial/01_train_aart.ipynb)** — Train AART on CKB paired proteomics data (SomaScan → Olink)
2. **[02_evaluate_and_compare.ipynb](tutorial/02_evaluate_and_compare.ipynb)** — Per-protein and per-sample evaluation, comparison to Direct 1-to-1 mapping

See [`data/CKB/README.md`](data/CKB/README.md) for data download instructions.
If you have any questions, please feel free to contact Yurui Chen (yurui.chen@yale.edu).

## Citation

If you use AART in your research, please cite: 10.64898/2026.06.29.735313v1

```

```

## License

AART is licensed under the MIT License.

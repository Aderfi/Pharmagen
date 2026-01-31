# 🚀 Getting Started Guide

This guide will help you set up your environment and run your first training or prediction with Pharmagen.

## 1. Environment Setup

Ensure you have **Python 3.10** installed. We recommend using `uv` for project management.

```bash
# Dependency installation
uv sync
```

## 2. Data Preparation

Pharmagen expects input data to follow a specific format.

### For Training
The input file (CSV or TSV) must contain columns for:
1.  **Features (Input):** Drug IDs, Genes, Variants, Clinical Phenotypes.
2.  **Targets (Output):** The variable you want to predict (e.g., `Outcome`, `SideEffect`).

Ensure columns match those defined in `config/models.toml` under the `[data]` section.

Example from `models.toml`:
```toml
[Phenotype_Effect_Outcome.data]
features = ["Drug_ID", "Gene_ID", "Variant_ID"]
targets = ["Outcome_Label"]
```

### For Inference (Prediction)
The file must have the same **Feature** columns used to train the model. The Target column is not required.

## 3. Running an Experiment

### Step A: Verify Configuration
Check `config/models.toml` and ensure your model is defined.

### Step B: Train the Model
Run training. If you have a GPU, Pharmagen will detect it automatically.

```bash
python main.py --mode train --model Phenotype_Effect_Outcome --input data/processed/my_dataset.tsv --epochs 20
```

### Step C: Evaluate Results
Upon completion, check the `reports/` folder.
*   Find loss plots in `reports/figures/`.
*   Review final metrics in `results/`.

## 4. Using the Bioinformatics Pipeline (NGS)

If starting from FASTQ files (raw sequencing):

1.  Place your `.fastq.gz` files in `data/raw/`.
2.  Use the interactive menu to launch the cleaning and alignment pipeline.
    ```bash
    python main.py --mode menu
    ```
3.  Select the NGS processing option.
4.  The system will generate BAM and VCF files in `data/processed/{Patient_ID}/`.

## Common Troubleshooting

*   **Memory Error (OOM):** Reduce `batch_size` in arguments (`--batch-size 32`).
*   **Columns Not Found:** Verify CSV headers match `models.toml` *exactly*.
*   **CUDA Not Available:** Ensure NVIDIA drivers and a PyTorch version compatible with your CUDA version are installed.

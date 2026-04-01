# Autoresearch for Waste Classification

Autonomous AI research agent for optimizing waste classification models. Built on the [autoresearch](https://github.com/karpathy/autoresearch) pattern, adapted for waste/recycling in Nigeria.

## What It Does

An AI coding agent autonomously experiments with computer vision models to classify waste into 3 categories:
- **E-waste** (class 0): batteries, phones, electronics
- **Recyclable** (class 1): plastic, glass, metal, paper
- **Organic** (class 2): food waste, compost

The agent modifies `train.py`, runs experiments on HF Jobs GPUs, checks metrics, and iterates — indefinitely.

## Project Structure

```
train.py        — model, training loop, data loading (AGENT MODIFIES THIS)
prepare.py      — fixed evaluation reference (DO NOT MODIFY)
program.md      — agent instructions
results.tsv     — experiment log (untracked)
```

## Quick Start

```bash
# 1. Install hf CLI
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login

# 2. Test single run locally
uv run train.py

# 3. Test on HF Jobs (A100 GPU)
hf jobs uv run --flavor a100-large --timeout 10m train.py
```

## Running the Agent

1. Create experiment branch: `git checkout -b autoresearch/<tag>`
2. Point your AI coding agent at `program.md`
3. The agent handles everything: research, code changes, GPU runs, evaluation

## Metrics

- **val_accuracy**: Top-1 accuracy on waste classification (higher is better)
- **yield_mse**: Mean squared error for yield prediction (lower is better)
- **combined_score**: `accuracy - 0.1 * yield_mse`

## Datasets

Data loads automatically via HuggingFace `datasets`:
- `omasteam/waste-garbage-management-dataset`
- `huaweilin/waste-classification`
- `NeoAivara/Waste_Classification_data`

Optional Kaggle sources available with credentials.

## License

MIT

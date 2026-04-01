# Autoresearch for Waste Classification

Autonomous AI research agent for optimizing waste classification models. Built on the [autoresearch](https://github.com/karpathy/autoresearch) pattern, adapted for waste/recycling in Nigeria.

## What It Does

An AI coding agent autonomously experiments with computer vision models to classify waste into 3 categories:
- **E-waste** (class 0): batteries, phones, electronics
- **Recyclable** (class 1): plastic, glass, metal, paper
- **Organic** (class 2): food waste, compost

The agent modifies `train.py`, runs experiments on Modal GPUs, checks metrics, and iterates — indefinitely.

## Project Structure

```text
train.py        — model, training loop, data loading (AGENT MODIFIES THIS)
modal_app.py    — Modal GPU runner (submits train.py to cloud GPU)
prepare.py      — fixed evaluation reference (DO NOT MODIFY)
program.md      — agent instructions
results.tsv     — experiment log (untracked)
```

## Quick Start

```bash
# 1. Install Modal CLI
pip install modal
modal token new

# 2. (Optional) Add Kaggle credentials for extra data
modal secret create kaggle-credentials KAGGLE_USERNAME=your_username KAGGLE_KEY=your_key

# 3. Test single run locally (CPU)
uv run train.py

# 4. Test on Modal (T4 GPU)
modal run modal_app.py
```

## Running the Agent

1. Create experiment branch: `git checkout -b autoresearch/<tag>`
2. Point your AI coding agent at `program.md`
3. The agent handles everything: research via `hf papers`, code changes, Modal GPU runs, evaluation

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

## Cost

With Modal's $30 free trial:
- T4 GPU: ~$0.005/min
- 5-minute experiment: ~$0.025
- **~1000 experiments with $30 credit**

## License

MIT

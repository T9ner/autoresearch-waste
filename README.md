# Autoresearch for Waste Classification

Autonomous AI research agent for optimizing waste classification models. Modified from karpathy/autoresearch for recycling in Nigeria.

## What It Does

An AI agent autonomously experiments with computer vision models to classify waste (e-waste, plastic, organic) and predict yield percentage. It modifies `train.py`, runs experiments, checks metrics, and iterates overnight.

## Project Structure

```
prepare.py      — fixed data loading, evaluation (DO NOT MODIFY)
train.py        — model, training loop (AGENT MODIFIES THIS)
program.md      — agent instructions (HUMAN MODIFIES THIS)
data/           — downloaded datasets
results.tsv     — experiment log
```

## Quick Start

```bash
# 1. Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download datasets
uv run prepare.py

# 4. Test single run
uv run train.py
```

## Running the Agent

```bash
# Create a new experiment branch
git checkout -b autoresearch/waste-exp1

# Run agent with program.md instructions
# Point your coding agent (Claude, Codex) at program.md
```

## Metrics

- **classification_accuracy**: Top-1 accuracy on waste classification (higher is better)
- **yield_prediction_mse**: Mean squared error for yield prediction (lower is better)
- **combined_score**: Weighted combination: accuracy - 0.1 * yield_mse

## Datasets Used

- `NeoAivara/Waste_Classification_data` — 10-class waste images
- `bryandts/waste_organic_anorganic_classification` — organic/inorganic
- `electricsheepafrica/african-e-waste-flows` — context data

## License

MIT
# Modal Setup for Autoresearch-Waste

This guide shows how to run your waste classification autoresearch on Modal's cloud GPU.

## Step 1: Sign Up for Modal

1. Go to [modal.com](https://modal.com)
2. Sign up with your email (the $30 trial credit is automatic)
3. Install Modal CLI:
```bash
pip install modal
modal token new
```

### (Optional) Kaggle Data

For additional training data from Kaggle:
1. Go to Kaggle > Account > Create New API Token
2. Run:
```bash
modal secret create kaggle-credentials KAGGLE_USERNAME=your_username KAGGLE_KEY=your_key
```

## Step 2: Verify Modal Script

The repository already includes `modal_app.py` which:
- Uses `debian_slim` + pip for dependencies (torch, torchvision, datasets, etc.)
- Mounts the local project directory into the container automatically
- Runs `train.py` on a T4 GPU with a 10-minute training budget
- Caches data on a persistent Modal volume (`autoresearch-waste-cache`)
- Injects Kaggle credentials if the `kaggle-credentials` secret exists
- Parses and returns `val_accuracy` and `yield_mse` from training output

No need to create a new file — just run:

```bash
# Single experiment
modal run modal_app.py::run_single_experiment

# Or deploy for repeated use
modal deploy modal_app.py
```

## Step 3: Run

```bash
# Run a single experiment
modal run modal_app.py::run_single_experiment

# Or start the full autonomous loop (uses Ollama locally + Modal for GPU)
python local_manager.py --model llama3.1:latest --num 10
```

## Step 4: Run the Autonomous Loop

The `local_manager.py` orchestrator handles the full loop:
1. Consults a local Ollama model for code improvements
2. Commits changes to `train.py`
3. Triggers training on Modal (T4 GPU)
4. Parses results and feeds them back to Ollama

```bash
# Make sure Ollama is running first
ollama serve &
ollama pull llama3.1:latest

# Run 10 experiments
python local_manager.py --model llama3.1:latest --num 10
```

## Cost Estimate

With Modal's $30 trial:
- T4 GPU: ~$0.005/minute
- 5-minute experiment: ~$0.025
- 100 experiments: ~$2.50
- **You could run ~1000 experiments with $30 credit!**

## Tips

1. **Start small**: Test with 1-2 experiments first before running overnight
2. **Check logs**: Use `modal logs` to monitor progress
3. **Save checkpoints**: The script pushes to git so you don't lose progress
4. **GPU choice**: T4 is cheapest. A100 is faster but more expensive

## Troubleshooting

- **OOM errors**: Reduce batch size in Config
- **Dataset loading fails**: Check internet connectivity in Modal container

If you open `modal_app.py` directly, the module entrypoint should now be:

```python
if __name__ == "__main__":
    pass  # Use: modal run modal_app.py::run_single_experiment
```

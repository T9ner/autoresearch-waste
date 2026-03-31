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

## Step 2: Create Modal Script

Create a file `modal_app.py`:

```python
import modal
import os

# Create the Modal app
app = modal.App("autoresearch-waste")

# NOTE: Replace YOUR_USERNAME below with your actual GitHub username
GITHUB_REPO = "https://github.com/T9ner/autoresearch-waste.git"

# Image with dependencies
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn9-devel")
    .pip_install("torchvision>=0.15.0", "datasets>=2.14.0", "Pillow>=10.0.0")
    .apt_install("git")
)

# Volume for caching datasets
volume = modal.Volume.from_name("autoresearch-cache", create=True)

@app.function(image=image, gpu="T4", volumes={"/cache": volume}, timeout=600)
def run_experiment():
    """Run a single experiment."""
    import subprocess
    
    # Clone the autoresearch-waste repo
    subprocess.run(["git", "clone", "https://github.com/T9ner/autoresearch-waste.git"], check=True)
    os.chdir("autoresearch-waste")
    
    # Sync dependencies
    subprocess.run(["uv", "sync"], check=True)
    
    # Run the experiment
    result = subprocess.run(["uv", "run", "train.py"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    
    return result.stdout

@app.function(image=image, gpu="T4", volumes={"/cache": volume}, timeout=3600)
def run_overnight():
    """Run experiments overnight (the main autonomous loop)."""
    import subprocess
    import time
    from datetime import datetime
    
    # Clone repo
    subprocess.run(["git", "clone", "https://github.com/T9ner/autoresearch-waste.git"], check=True)
    os.chdir("autoresearch-waste")
    subprocess.run(["uv", "sync"], check=True)
    
    # Create experiment branch
    tag = datetime.now().strftime("%b%d")
    subprocess.run(["git", "checkout", "-b", f"autoresearch/{tag}"], check=True)
    
    # Initialize results file
    with open("results.tsv", "w") as f:
        f.write("commit\taccuracy\tyield_mse\tcombined_score\tmemory_gb\tstatus\tdescription\n")
    
    num_experiments = 0
    max_experiments = 100  # ~8 hours at 5 min/experiment
    
    while num_experiments < max_experiments:
        print(f"\n=== Experiment {num_experiments + 1} ===")
        
        # The agent would modify train.py here
        # For demo, we just run
        
        # Commit
        subprocess.run(["git", "add", "train.py"], check=True)
        subprocess.run(["git", "commit", "-m", f"exp {num_experiments + 1}"], check=True)
        
        # Run training
        result = subprocess.run(
            ["uv", "run", "train.py"], 
            capture_output=True, 
            text=True,
            timeout=600  # 10 min timeout
        )
        
        # Save log
        with open("run.log", "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        
        # Parse results (simplified)
        # In real implementation, parse val_accuracy, etc.
        
        num_experiments += 1
        print(f"Completed experiment {num_experiments}")
        
        # Save checkpoint to volume
        subprocess.run(["git", "push"], check=True)
    
    print(f"Completed {num_experiments} experiments")

if __name__ == "__main__":
    # Deploy the app
    with stub.run():
        run_overnight.spawn()
```

## Step 3: Deploy and Run

```bash
# Deploy to Modal
modal deploy modal_app.py

# Or run a single experiment
modal run modal_app::run_experiment
```

## Step 4: Connect Your Agent

Point your coding agent (Claude, Codex) at the repo and `program.md`:

```
Hi, set up a new experiment. Check program.md and let's start.
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
- **Git errors**: Make sure you have a GitHub repo created first
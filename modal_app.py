"""
Modal app for autoresearch-waste
Run autonomous waste classification experiments on Modal's cloud GPU.

Usage:
    modal run modal_app.py
    modal deploy modal_app.py
"""

import modal
import os
import subprocess

# ============ CONFIG ============
GITHUB_REPO = "https://github.com/T9ner/autoresearch-waste.git"
APP_NAME = "autoresearch-waste"
GPU_TYPE = "T4"  # Cheapest: T4, Faster: A100
TIME_BUDGET = 300  # 5 minutes per experiment

# ============ MODAL SETUP ============
app = modal.App(APP_NAME)

# Base image with PyTorch
image = (
    modal.Image.from_registry("pytorch/pytorch:2.1.0-cuda12.1-cudnn9-devel")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "datasets>=2.14.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    )
    .apt_install("git")
)

# Volume for caching models and data
volume = modal.Volume.from_name(f"{APP_NAME}-cache", create=True)


# ============ FUNCTIONS ============

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": volume},
    timeout=TIME_BUDGET + 60,
    retries=modal.Retries(
        max_retries=2,
        backoff=modal.Backoff(initial_delay=10),
    ),
)
def run_single_experiment():
    """Run a single training experiment."""
    print("=" * 50)
    print("Starting waste classification experiment")
    print("=" * 50)
    
    # Clone repo
    result = subprocess.run(
        ["git", "clone", GITHUB_REPO, "/app/autoresearch"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Clone error: {result.stderr}")
        return {"status": "error", "message": result.stderr}
    
    os.chdir("/app/autoresearch")
    
    # Install deps
    print("Installing dependencies...")
    subprocess.run(["pip", "install", "-e", "."], capture_output=True)
    
    # Run training
    print("Running training...")
    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True,
        text=True,
        timeout=TIME_BUDGET,
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Parse results (simple version)
    output = result.stdout + result.stderr
    accuracy = None
    yield_mse = None
    
    for line in output.split('\n'):
        if 'val_accuracy:' in line:
            try:
                accuracy = float(line.split(':')[1].strip())
            except:
                pass
        if 'yield_mse:' in line:
            try:
                yield_mse = float(line.split(':')[1].strip())
            except:
                pass
    
    return {
        "status": "success" if result.returncode == 0 else "error",
        "accuracy": accuracy,
        "yield_mse": yield_mse,
        "output": output[:2000],  # Last 2000 chars
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": volume},
    timeout=28800,  # 8 hours max
)
def run_autonomous_loop(num_experiments: int = 100):
    """
    Run the full autonomous research loop overnight.
    
    Args:
        num_experiments: Number of experiments to run (default: 100 = ~8 hours at 5 min each)
    """
    import datetime
    from pathlib import Path
    
    print("=" * 50)
    print(f"Starting autonomous loop: {num_experiments} experiments")
    print("=" * 50)
    
    # Clone and setup
    subprocess.run(["git", "clone", GITHUB_REPO, "/app/autoresearch"], check=True)
    os.chdir("/app/autoresearch")
    
    # Create experiment branch
    tag = datetime.datetime.now().strftime("%b%d").lower()
    branch_name = f"autoresearch/{tag}"
    
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    
    # Initialize results file
    results_file = Path("/app/autoresearch/results.tsv")
    if not results_file.exists():
        results_file.write_text(
            "commit\taccuracy\tyield_mse\tcombined_score\tmemory_gb\tstatus\tdescription\n"
        )
    
    # Main loop
    for i in range(num_experiments):
        print(f"\n=== Experiment {i+1}/{num_experiments} ===")
        
        # Commit current state
        subprocess.run(["git", "add", "train.py"], check=True)
        subprocess.run(
            ["git", "commit", "-m", f"exp {i+1}"],
            capture_output=True,
            check=True,
        )
        
        # Run training
        try:
            result = subprocess.run(
                ["python", "train.py"],
                capture_output=True,
                text=True,
                timeout=TIME_BUDGET + 60,
            )
            
            # Save log
            Path("run.log").write_text(result.stdout + result.stderr)
            
            # Parse results
            accuracy = 0.0
            yield_mse = 0.0
            
            for line in result.stdout.split('\n'):
                if 'val_accuracy:' in line:
                    try:
                        accuracy = float(line.split(':')[1].strip())
                    except:
                        pass
                if 'yield_mse:' in line:
                    try:
                        yield_mse = float(line.split(':')[1].strip())
                    except:
                        pass
            
            combined = accuracy - 0.1 * yield_mse
            status = "keep" if combined > 0 else "discard"
            
            # Log result
            commit_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip()
            
            with open(results_file, "a") as f:
                f.write(f"{commit_hash}\t{accuracy}\t{yield_mse}\t{combined}\t0.0\t{status}\texp {i+1}\n")
            
            # If bad result, revert
            if status == "discard":
                subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
            
            print(f"Result: accuracy={accuracy:.2f}, yield_mse={yield_mse:.4f}, combined={combined:.2f}")
            
        except subprocess.TimeoutExpired:
            print(f"Experiment {i+1} timed out - skipping")
            with open(results_file, "a") as f:
                f.write(f"-\t0.0\t0.0\t0.0\t0.0\tcrash\ttimeout\n")
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
        
        except Exception as e:
            print(f"Experiment {i+1} error: {e}")
            with open(results_file, "a") as f:
                f.write(f"-\t0.0\t0.0\t0.0\t0.0\tcrash\t{e}\n")
            subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
    
    # Push results
    print("\nPushing results to GitHub...")
    subprocess.run(["git", "add", "results.tsv"], check=True)
    subprocess.run(["git", "commit", "-m", "results"], capture_output=True)
    subprocess.run(["git", "push", "origin", branch_name], check=True)
    
    print(f"\nCompleted {num_experiments} experiments!")
    print("Check results.tsv for details.")
    
    return {"status": "completed", "experiments": num_experiments}


# ============ MAIN ============

if __name__ == "__main__":
    # Run a single test experiment
    result = run_single_experiment()
    print("\n" + "=" * 50)
    print("RESULT:", result)
    print("=" * 50)
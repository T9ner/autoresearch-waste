import os
import subprocess
from pathlib import Path

import modal

# ============ CONFIGURATION ============

APP_NAME = "autoresearch-waste"
GPU_TYPE = "t4"
TIME_BUDGET = 600  # 10 minute training budget

# ============ MODAL SETUP ============
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "kaggle", 
        "datasets", 
        "torch", 
        "torchvision", 
        "Pillow", 
        "numpy"
    )
    .add_local_dir(Path(__file__).resolve().parent, remote_path="/root/autoresearch-waste")
)

# Volume for caching models and data
volume = modal.Volume.from_name("autoresearch-waste-cache", create_if_missing=True)

try:
    available_secrets = {secret.name for secret in modal.Secret.objects.list()}
except Exception:
    available_secrets = set()

kaggle_secret = (
    modal.Secret.from_name("kaggle-credentials")
    if "kaggle-credentials" in available_secrets
    else modal.Secret.from_dict({})
)

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.cache/autoresearch-waste": volume},
    secrets=[kaggle_secret],
    timeout=TIME_BUDGET + 120,
)
def run_single_experiment():
    """
    Run a single training session on the current state of the local train.py.
    """
    import os
    os.chdir("/root/autoresearch-waste")
    
    # We use local-file mounting to ensure the cloud always has the latest train.py
    # This function is triggered by local_manager.py after a code tweak.
    
    print("-" * 30)
    print("STARTING TRAINING EXPERIMENT")
    print("-" * 30)

    try:
        # Run training
        result = subprocess.run(
            ["python", "train.py"],
            capture_output=True,
            text=True,
            timeout=TIME_BUDGET,
        )

        # Print full output so local_manager.py can parse it
        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            print(f"TRAINING_FAILED: exit code {result.returncode}")
            return {"status": "error", "error": result.stderr}

        # Parse basic metrics for the cloud logs (optional)
        accuracy = 0.0
        yield_mse = 0.0
        for line in result.stdout.split("\n"):
            stripped = line.strip()
            if stripped.startswith("val_accuracy:"):
                try:
                    accuracy = float(stripped.split(":", 1)[1].strip())
                except Exception:
                    pass
            if stripped.startswith("yield_mse:"):
                try:
                    yield_mse = float(stripped.split(":", 1)[1].strip())
                except Exception:
                    pass

        print(f"\nFinal Accuracy: {accuracy:.4f}")
        print(f"Final Yield MSE: {yield_mse:.4f}")
        
        return {
            "status": "success",
            "accuracy": accuracy,
            "yield_mse": yield_mse,
        }

    except subprocess.TimeoutExpired:
        print("TRAINING_TIMEOUT: Experiment exceeded time budget.")
        return {"status": "timeout"}
    except Exception as e:
        print(f"TRAINING_ERROR: {str(e)}")
        return {"status": "error", "error": str(e)}

import os
import subprocess
import re
import argparse
import time
from pathlib import Path

# ============ CONFIGURATION ============
DEFAULT_MODEL = "llama3.1:latest"
MODAL_APP_CMD = ["modal", "run", "modal_app.py::run_single_experiment"]
OLLAMA_TIMEOUT = 600

def consult_ollama(model, current_code, history, rules, last_log):
    """Ask Ollama to propose a code improvement."""
    prompt = f"Improve this waste classification script (train.py) to achieve >95% accuracy. Focus on: {rules[:1000]}... \n\nCURRENT CODE:\n{current_code}\n\nEXPERIMENT HISTORY:\n{history[:500]}\n\nLAST LOG SUMMARY:\n{last_log[-500:]}\n\nTask: Output the FULL updated Python script. Do not explain anything, just give me the code."

    try:
        print(f"--- Consulting Ollama (CLI Mode): {model} ---")
        
        cli_result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        
        if cli_result.returncode != 0:
            print(f"Ollama CLI Error: {cli_result.stderr}")
            return None
            
        content = cli_result.stdout.strip()
        
        # Save raw output for debugging
        Path("last_raw_brain_output.txt").write_text(content, encoding="utf-8")

        # More robust code extraction: look for any block starting with 'import' or 'from'
        if "```" in content:
            # Try to grab the first python block
            blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", content, re.DOTALL)
            if blocks:
                return blocks[0].strip()
        
        # If no triple backticks, check if it starts like a python file
        if "import " in content or "from " in content:
            # If the model didn't use backticks, just return everything from the first import
            start_idx = content.find("import ")
            if start_idx == -1: 
                start_idx = content.find("from ")
            return content[start_idx:].strip()

        print("Ollama output did not contain valid Python code.")
        return None
            
    except Exception as e:
        print(f"Error during Ollama consultation: {e}")
        return None

def parse_metrics(output):
    """Extract metrics from the standardized output format."""
    accuracy = 0.0
    yield_mse = 0.0
    
    acc_match = re.search(r"val_accuracy:\s*([\d\.]+)", output)
    mse_match = re.search(r"yield_mse:\s*([\d\.]+)", output)
    
    if acc_match:
        accuracy = float(acc_match.group(1))
    if mse_match:
        yield_mse = float(mse_match.group(1))
        
    return accuracy, yield_mse

def main():
    parser = argparse.ArgumentParser(description="Autonomous Research Orchestrator (Ollama + Modal)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Local Ollama model name")
    parser.add_argument("--num", type=int, default=10, help="Number of experiments to run")
    args = parser.parse_args()

    # Files
    train_py = Path("train.py")
    results_tsv = Path("results.tsv")
    program_md = Path("program.md")

    if not results_tsv.exists():
        results_tsv.write_text("commit\taccuracy\tyield_mse\tcombined_score\tdescription\n")

    last_log = "Initial run - no previous log."
    rules = program_md.read_text() if program_md.exists() else "No program rules."

    for i in range(args.num):
        print(f"\n{'='*20} Experiment {i+1}/{args.num} {'='*20}")

        current_code = train_py.read_text()
        history = results_tsv.read_text()

        print(f"Waiting up to {OLLAMA_TIMEOUT}s for brain result...")
        # 1. Tweak code
        new_code = consult_ollama(args.model, current_code, history, rules, last_log)
        if not new_code or len(new_code) < 100:
            print("Failed to get valid code from Ollama. Skipping iteration.")
            continue

        # 2. Save and Commit locally (so we can rollback)
        train_py.write_text(new_code)
        
        # Git Commit
        desc = f"Exp {i+1} by {args.model}"
        subprocess.run(["git", "add", "train.py"], capture_output=True)
        subprocess.run(["git", "commit", "-m", desc], capture_output=True)
        
        commit_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], 
                                   capture_output=True, text=True).stdout.strip()
        print(f"Committed tweak: {commit_hash}")

        # 3. Run on Modal
        print(f"Triggering training on Modal (T4 GPU)...")
        start_time = time.time()
        result = subprocess.run(MODAL_APP_CMD, capture_output=True, text=True)
        duration = time.time() - start_time
        
        output = result.stdout + result.stderr
        last_log = output
        
        if result.returncode != 0:
            print("Cloud Training Failed. Rolling back code...")
            subprocess.run(["git", "reset", "--hard", "HEAD^"], capture_output=True)
            continue

        # 4. Record Results
        accuracy, yield_mse = parse_metrics(output)
        combined_score = accuracy - (0.1 * yield_mse)

        print(f"Results: Accuracy={accuracy:.4f}, Yield MSE={yield_mse:.4f}, Score={combined_score:.4f}")
        
        # Append to TSV
        with open(results_tsv, "a") as f:
            f.write(f"{commit_hash}\t{accuracy:.4f}\t{yield_mse:.4f}\t{combined_score:.4f}\t{desc}\n")

        print(f"Iteration complete. Duration: {duration/60:.2f} mins.")

if __name__ == "__main__":
    main()

"""
Lightning AI runner for autoresearch-waste.
Drop-in replacement for modal_app.py — runs on T4 GPU via lightning-sdk.

Setup:
    pip install lightning-sdk
    export LIGHTNING_USER_ID=<your-user-id>
    export LIGHTNING_API_KEY=<your-api-key>

Usage:
    # Single test run
    python lightning_app.py --mode single

    # Full overnight loop (100 experiments)
    python lightning_app.py --mode loop --experiments 100
"""

import argparse
import os
import time

from lightning_sdk import Job, Machine, Status, Studio

# ============ CONFIG ============
GITHUB_REPO   = "https://github.com/T9ner/autoresearch-waste.git"
STUDIO_NAME   = "autoresearch-waste"          # Must match your Studio name in Lightning AI UI
TEAMSPACE     = os.environ.get("LIGHTNING_TEAMSPACE", "")   # set if using a teamspace
USER          = os.environ.get("LIGHTNING_USER_ID", "")
GPU_MACHINE   = Machine.T4
TIME_BUDGET   = 300   # 5 minutes per experiment (seconds)
MAX_TIMEOUT   = 720   # 12 minutes hard kill timeout per job


def get_studio() -> Studio:
    """Connect to the Lightning AI Studio with auto-detection."""
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE")
    user = os.environ.get("LIGHTNING_USERNAME") or os.environ.get("LIGHTNING_USER_ID")

    if not user or not teamspace:
        try:
            from lightning_sdk.lightning_cloud.rest_client import LightningClient
            client = LightningClient()
            if not user:
                user_res = client.auth_service_get_user()
                user = user_res.username
            if not teamspace:
                memberships_res = client.projects_service_list_memberships()
                memberships = memberships_res.memberships
                if memberships:
                    default_memberships = [m for m in memberships if m.is_default]
                    if default_memberships:
                        teamspace = default_memberships[0].name
                    else:
                        teamspace = memberships[0].name
                else:
                    teamspace = "default"
        except Exception as e:
            print(f"Warning: Auto-detection of teamspace/user failed: {e}")
            teamspace = teamspace or "default"
            user = user or ""

    kwargs = {"name": STUDIO_NAME, "teamspace": teamspace}
    if user:
        kwargs["user"] = user

    print(f"Connecting to Studio '{STUDIO_NAME}' in teamspace '{teamspace}' (user: '{user}')...")
    studio = Studio(**kwargs)
    studio.start()
    return studio


def get_current_branch() -> str:
    import subprocess
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
        if branch == "HEAD":
            return "main"
        return branch
    except Exception:
        return "main"


def run_single(studio: Studio, max_timeout: int = MAX_TIMEOUT) -> dict:
    """Submit one training experiment and wait for it to finish."""
    job_name = f"waste-single-{int(time.time())}"
    branch = get_current_branch()
    print(f"Using git branch '{branch}' for remote job execution.")

    setup_cmd = (
        f"git clone --branch {branch} {GITHUB_REPO} /tmp/autoresearch-waste && "
        "cd /tmp/autoresearch-waste && "
        "pip install -q -e . && "
        "python train.py > run.log 2>&1 && "
        "grep '^val_accuracy:\\|^yield_mse:\\|^combined_score:\\|^peak_vram_mb:' run.log"
    )

    print(f"Submitting job: {job_name}")
    job = Job.run(
        command=setup_cmd,
        name=job_name,
        machine=GPU_MACHINE,
        studio=studio,
    )
    # Poll until done
    start = time.time()
    while job.status not in (Status.Completed, Status.Failed, Status.Stopped):
        elapsed = time.time() - start
        print(f"  [{elapsed:.0f}s] status: {job.status} ...", flush=True)
        if elapsed > max_timeout:
            print("  Timeout — stopping job.", flush=True)
            job.stop()
            return {"status": "timeout"}
        time.sleep(15)

    print(f"Job finished with status: {job.status}")
    return {"status": job.status.value if job.status else "unknown", "job_name": job_name}


def run_loop(studio: Studio, num_experiments: int = 100, time_budget: int = TIME_BUDGET):
    """
    Submit the full autonomous research loop as a single long-running job.
    The job clones the repo, creates a branch, and runs the autoresearch loop.
    """
    import datetime
    tag = datetime.datetime.now().strftime("%b%d").lower()
    branch = f"autoresearch/{tag}"
    job_name = f"waste-loop-{tag}"
    current_branch = get_current_branch()

    loop_cmd = f"""
set -e
git clone --branch {current_branch} {GITHUB_REPO} /tmp/autoresearch-waste
cd /tmp/autoresearch-waste
pip install -q -e .
git checkout -b {branch}

# Initialize results if missing
if [ ! -f results.tsv ]; then
  echo -e "commit\\taccuracy\\tyield_mse\\tcombined_score\\tmemory_gb\\tstatus\\tdescription" > results.tsv
fi

for i in $(seq 1 {num_experiments}); do
  echo ""
  echo "=== Experiment $i / {num_experiments} ==="

  # Commit current train.py
  git add train.py
  git commit -m "exp $i" || true

  # Run with timeout
  timeout {time_budget + 60} python train.py > run.log 2>&1
  EXIT_CODE=$?

  if [ $EXIT_CODE -ne 0 ]; then
    echo "Run failed or timed out (exit $EXIT_CODE)"
    echo -e "-\\t0.0\\t0.0\\t0.0\\t0.0\\tcrash\\texp $i" >> results.tsv
    git reset --hard HEAD~1 2>/dev/null || true
    continue
  fi

  # Parse metrics
  VAL_ACC=$(grep '^val_accuracy:' run.log | awk '{{print $2}}')
  YIELD_MSE=$(grep '^yield_mse:' run.log | awk '{{print $2}}')
  COMBINED=$(grep '^combined_score:' run.log | awk '{{print $2}}')
  VRAM=$(grep '^peak_vram_mb:' run.log | awk '{{print $2}}' | awk '{{printf "%.2f", $1/1024}}')

  VAL_ACC=${{VAL_ACC:-0.0}}
  YIELD_MSE=${{YIELD_MSE:-0.0}}
  COMBINED=${{COMBINED:-0.0}}
  VRAM=${{VRAM:-0.0}}

  echo "  val_accuracy=$VAL_ACC  yield_mse=$YIELD_MSE  combined=$COMBINED  vram_gb=$VRAM"

  COMMIT=$(git rev-parse --short HEAD)

  # Decide keep/revert — keep if combined > 0
  STATUS="keep"
  if python3 -c "exit(0 if float('$COMBINED') > 0 else 1)" 2>/dev/null; then
    STATUS="keep"
  else
    STATUS="discard"
    git reset --hard HEAD~1 2>/dev/null || true
  fi

  echo -e "$COMMIT\\t$VAL_ACC\\t$YIELD_MSE\\t$COMBINED\\t$VRAM\\t$STATUS\\texp $i" >> results.tsv
  echo "  -> $STATUS"
done

# Push results branch
git config user.email "autoresearch@lightning.ai"
git config user.name "Autoresearch Bot"
git add results.tsv run.log
git commit -m "autoresearch results [{num_experiments} experiments]" || true
echo "Loop complete. Results in results.tsv"
cat results.tsv
""".strip()

    print(f"Submitting overnight loop: {num_experiments} experiments on branch '{branch}'")
    print(f"Job name: {job_name}")

    job = Job.run(
        command=loop_cmd,
        name=job_name,
        machine=GPU_MACHINE,
        studio=studio,
    )

    print(f"\nJob submitted! Status: {job.status}")
    print(f"\nMonitor at: https://lightning.ai (Jobs tab)")
    print(f"Or poll with: python lightning_app.py --mode status --job {job_name}")

    return job


def main():
    parser = argparse.ArgumentParser(description="Run autoresearch-waste on Lightning AI")
    parser.add_argument("--mode", choices=["single", "loop", "status"], default="single",
                        help="single=one experiment, loop=overnight, status=check job")
    parser.add_argument("--experiments", type=int, default=100,
                        help="Number of experiments for loop mode")
    parser.add_argument("--job", type=str, default="",
                        help="Job name to check status of")
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET,
                        help=f"Time budget in seconds per training run (default: {TIME_BUDGET})")
    parser.add_argument("--max-timeout", type=int, default=7200,
                        help="Maximum polling timeout in seconds for single run (default: 7200)")
    args = parser.parse_args()

    studio = get_studio()

    if args.mode == "single":
        result = run_single(studio, max_timeout=args.max_timeout)
        print("\nResult:", result)

    elif args.mode == "loop":
        job = run_loop(studio, num_experiments=args.experiments, time_budget=args.time_budget)
        print(f"\nJob is running in background — close your terminal safely.")
        print(f"Results will be committed to: autoresearch/<today's tag> branch on GitHub")

    elif args.mode == "status":
        if not args.job:
            print("Provide --job <job_name> to check status")
        else:
            try:
                job = Job(name=args.job, teamspace=studio.teamspace)
                print(f"Job '{args.job}' status: {job.status}")
                if job.status in (Status.Completed, Status.Failed, Status.Stopped):
                    print("\n--- Job Logs ---")
                    try:
                        logs = job.logs
                        log_lines = logs.splitlines()
                        if len(log_lines) > 30:
                            print("... (truncated) ...")
                            print("\n".join(log_lines[-30:]))
                        else:
                            print(logs)
                    except Exception as le:
                        print(f"Could not retrieve logs: {le}")
                else:
                    print("Job is still running. Check again later or monitor in the dashboard.")
            except Exception as e:
                print(f"Error fetching status for job '{args.job}': {e}")
                print(f"Alternative - run CLI: lightning status {args.job}")
                print("Or check: https://lightning.ai (Jobs tab)")


if __name__ == "__main__":
    main()

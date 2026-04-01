import subprocess

import modal

APP_NAME = "autoresearch-waste"
GPU_TYPE = "t4"
TIME_BUDGET = 1800  # 30 minutes (generous for first-run dataset downloads; cached runs are ~5 min)

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "datasets",
        "Pillow",
        "numpy",
        "kaggle",
    )
    .add_local_file("train.py", remote_path="/root/train.py")
)

volume = modal.Volume.from_name("autoresearch-waste-cache", create_if_missing=True)

try:
    kaggle_secret = modal.Secret.from_name("kaggle-credentials")
except modal.exception.NotFoundError:
    kaggle_secret = modal.Secret.from_dict({})


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.cache/autoresearch-waste": volume},
    secrets=[kaggle_secret],
    timeout=TIME_BUDGET + 120,
)
def train():
    result = subprocess.run(
        ["python", "/root/train.py"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        print(f"TRAINING_FAILED: exit code {result.returncode}")


@app.local_entrypoint()
def main():
    train.remote()

# Autoresearch for Waste Classification

This is an autonomous research experiment for optimizing waste classification in Nigeria.

## Context

You are helping optimize an AI model that classifies waste into 3 categories:
- **E-waste** (class 0): batteries, phones, electronics, laptops
- **Recyclable** (class 1): plastic bottles, containers, wrappers, glass, metal, paper
- **Organic** (class 2): food waste, compost

The goal is to maximize classification accuracy AND yield prediction accuracy.

## Setup

1. **Agree on a run tag**: Propose a tag based on today's date (e.g. `mar31`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md` — repository context
   - `prepare.py` — fixed data loading, evaluation. DO NOT MODIFY.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that datasets can be loaded from HuggingFace. If not, the code should handle gracefully.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

## Experimentation

Each experiment runs on a single GPU with a **fixed time budget of 5 minutes**.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit
- Change model architecture (resnet18, resnet34, efficientnet_b0, vit, etc.)
- Modify hyperparameters (learning rate, batch size, weight decay)
- Change image size, data augmentation
- Add or remove components (yield prediction head, auxiliary losses)
- Try different pretrained backbones
- Modify the training loop

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages beyond what's in pyproject.toml.
- Modify the evaluation harness.

**The goal**: Maximize the combined score:
```
combined_score = classification_accuracy - 0.1 * yield_prediction_mse
```

Higher accuracy is always good. Lower yield MSE is always good.

**VRAM** is a soft constraint — some increase is acceptable for meaningful gains, but don't blow up.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity? Probably not worth it. Removing code and getting equal/better results? Great outcome.

## Output Format

When training finishes, it prints:

```
val_accuracy:     85.50
yield_mse:       0.0234
combined_score:  85.27
training_seconds: 298.5
peak_vram_mb:    4120.5
num_params_M:   11.2
```

Extract from log:
```bash
grep "^val_accuracy:\|^yield_mse:\|^combined_score:" run.log
```

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	accuracy	yield_mse	combined_score	memory_gb	status	description
```

Example:
```
a1b2c3d	85.50	0.0234	85.27	4.0	keep	baseline resnet18
b2c3d4e	87.20	0.0210	87.01	4.2	keep	switch to efficientnet_b0
c3d4e5f	86.80	0.0195	86.60	5.1	keep	add yield prediction head
d4e5f6g	0.00	0.0000	0.00	0.0	crash	vit vision transformer OOM
```

## Experiment Loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Tune `train.py` with an experimental idea
3. git commit
4. Run experiment: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_accuracy:\|^yield_mse:\|^combined_score:\|^peak_vram_mb:" run.log`
6. If grep output is empty, the run crashed. Read `tail -n 50 run.log` for errors. Fix or skip.
7. Record results in tsv (leave tsv untracked by git)
8. If combined_score improved, keep the commit
9. If combined_score is equal or worse, git reset back

**Timeout**: If run exceeds 10 minutes, kill and treat as failure.

**Crashes**: If OOM or bug, use judgment — easy fix? Retry. Fundamentally broken? Skip, log "crash", move on.

**NEVER STOP**: Once the loop begins, do NOT ask the human to continue. You are autonomous. If you run out of ideas, think harder. Try different architectures, combine previous near-misses, try radical changes. Run until interrupted.

## Datasets

The agent should utilize available waste classification datasets.

**HuggingFace (always available):**
- `omasteam/waste-garbage-management-dataset` — 10-class waste images (split: `train`)
- `huaweilin/waste-classification` — hierarchical waste labels with subclass detail (split: `cleaned`, label field: `subclass`)
- `NeoAivara/Waste_Classification_data` — 12-class waste images (split: `train`)

**Kaggle (requires credentials):**
- `asdasdasasdas/garbage-classification` — 6-class garbage images (cardboard, glass, metal, paper, plastic, trash)
- `isaacritharson/metal-glassgarbage-classification-data` — glass, metals, cardboard waste

Map all labels to 3 categories: e-waste (0), recyclable (1), organic (2).

## Notes

- The model classifies images, not text
- ImageNet-pretrained backbones work well for transfer learning
- Start with a baseline (resnet18), then iterate
- 5 minutes = ~12 experiments/hour, ~100 overnight

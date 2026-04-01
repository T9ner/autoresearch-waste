# Autoresearch for Waste Classification

This is an autonomous research experiment for optimizing waste classification in Nigeria.

## Context

You are helping optimize an AI model that classifies waste into 3 categories:
- **E-waste** (class 0): batteries, phones, electronics, laptops
- **Recyclable** (class 1): plastic bottles, containers, wrappers, glass, metal, paper
- **Organic** (class 2): food waste, compost

The goal is to maximize the combined score:

```
combined_score = classification_accuracy - 0.1 * yield_prediction_mse
```

Higher classification accuracy is better. Lower yield prediction MSE is better.

## Setup

1. **Agree on a run tag**: Propose a tag based on today's date (for example `apr01`) and create branch `autoresearch/<tag>`.
2. **Read `train.py`**: This is the only file you modify during experimentation. It is self-contained and declares its dependencies inline for `uv`.
3. **Read `prepare.py`**: This is the fixed evaluation reference. DO NOT MODIFY.
4. **Verify Modal is configured**: Run `modal token new` if needed so `modal run` works.
5. **Test a single run**: Confirm the stack works with `modal run modal_app.py`.
6. **Initialize `results.tsv`**: Create it with the header row only.
7. **Confirm and go**: Once the setup is valid, begin the experiment loop.

## Running on Modal

Each experiment runs on a T4 GPU via Modal. Launch training with:

```bash
modal run modal_app.py 2>&1 | tee run.log
```

- `modal_app.py` automatically uploads the current `train.py` to the Modal container
- Training runs on a T4 GPU with a 10-minute timeout
- Downloaded datasets are cached on a persistent Modal volume between runs
- Kaggle credentials are injected automatically if you've set them up with `modal secret create kaggle-credentials`

## Experimentation

Each experiment should fit within the Modal timeout budget.

**What you CAN do:**
- Modify `train.py`
- Change the model architecture
- Change hyperparameters such as learning rate, batch size, weight decay, epochs
- Change image size and data augmentation
- Add or remove components such as the yield prediction head or auxiliary losses
- Try different pretrained backbones
- Modify the training loop

**What you CANNOT do:**
- Modify `prepare.py`
- Modify the evaluation harness
- Change the definition of the reported metrics

**The goal**: Maximize the combined score:

```
combined_score = classification_accuracy - 0.1 * yield_prediction_mse
```

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but avoid wasteful blowups.

**Simplicity criterion**: All else being equal, simpler is better. A tiny gain that adds ugly complexity is usually not worth it. If you can remove code and keep or improve performance, that is a good outcome.

## Research with `hf papers`

Before each experiment, use `hf papers` to find ideas from recent research.

```bash
# Search for relevant techniques
hf papers search "waste classification deep learning"
hf papers search "image classification transfer learning"
hf papers search "data augmentation computer vision"
hf papers search "efficient resnet training"
```

```bash
# Read a promising paper
hf papers read <paper_id>
```

Use papers as inspiration, not as scripts to copy mechanically. Prefer ideas that are simple to implement inside `train.py` and plausible under the runtime budget.

## Output Format

When training finishes, it prints:

```
val_accuracy:     85.50
yield_mse:       0.0234
combined_score:  85.27
training_seconds: 298.5
peak_vram_mb:    4120.5
```

Extract from log:

```bash
grep "^val_accuracy:\|^yield_mse:\|^combined_score:\|^training_seconds:\|^peak_vram_mb:" run.log
```

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	val_accuracy	yield_mse	combined_score	memory_gb	status	paper	description
```

Example:

```
a1b2c3d	85.50	0.0234	85.27	4.0	keep	-	baseline resnet18
b2c3d4e	87.20	0.0210	87.01	4.2	keep	2503.08234	efficientnet_b0 from paper
c3d4e5f	86.80	0.0195	86.60	5.1	keep	-	stronger augmentation
d4e5f6g	0.00	0.0000	0.00	0.0	crash	-	vit OOM
```

## Experiment Loop

Repeat forever:

1. Research with `hf papers search` and identify one promising idea
2. Implement the idea by modifying `train.py`
3. Commit the change with `git commit`
4. Run the experiment:
   ```bash
   modal run modal_app.py 2>&1 | tee run.log
   ```
5. Evaluate the run by reading the reported metrics
6. Log the result to `results.tsv`
7. If `combined_score` improved, keep the commit
8. If the score is worse or equal, revert with `git reset --hard HEAD^`
9. Continue to the next experiment immediately

If the metric grep output is empty, the run crashed. Inspect the log, decide whether the failure is fixable, and either retry with a minimal fix or log a crash and move on.

**NEVER STOP**: Once the loop begins, do not ask the human whether to continue. Keep researching, implementing, running, evaluating, and iterating until interrupted.

## Datasets

The training script loads data via the HuggingFace `datasets` library. No manual setup needed.

**HuggingFace (always available):**
- `omasteam/waste-garbage-management-dataset` — 10-class waste images (split: `train`)
- `huaweilin/waste-classification` — hierarchical waste labels (split: `cleaned`, label field: `subclass`)
- `NeoAivara/Waste_Classification_data` — 12-class waste images (split: `train`)

**Kaggle (optional, requires credentials in environment):**
- `asdasdasasdas/garbage-classification` — 6-class garbage images
- `isaacritharson/metal-glassgarbage-classification-data` — glass, metals, cardboard

All labels are mapped to 3 categories: e-waste (0), recyclable (1), organic (2).

## Notes

- The model classifies images, not text
- `train.py` already handles HuggingFace datasets directly at runtime
- Kaggle data is optional and should be skipped gracefully if credentials are unavailable
- ImageNet-pretrained backbones are good defaults
- Start from a strong baseline, then iterate
- Cost: T4 GPU at ~$0.005/min, or about ~$0.025 per 5-minute experiment, which is roughly 1000 experiments with $30 credit
- If you want faster runs, change `GPU_TYPE` in `modal_app.py` to `"a100"` (more expensive)

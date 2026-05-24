# Autoresearch for Waste Classification

Autoresearch experiment for training and evaluating a computer-vision model that classifies waste images and estimates recoverable material yield. The project is adapted from the [autoresearch](https://github.com/karpathy/autoresearch) pattern for waste/recycling use cases, with a focus on practical waste streams relevant to Nigeria and similar contexts.

## What It Does

The model looks at an image of waste and predicts one of three categories:

- **E-waste** (`class 0`): batteries, phones, electronics, laptops, monitors, computers, TVs
- **Recyclable / dry waste** (`class 1`): plastic, bottles, wrappers, glass, metal, paper, cardboard, cans
- **Organic waste** (`class 2`): food waste, compostable or biodegradable waste

The training script also predicts a rough **yield percentage**: an estimate of how much useful/recoverable material may be available from that waste category.

## Latest Results

The latest completed cloud GPU loop was run on May 22 and its metrics are committed in [`results.tsv`](results.tsv).

Summary from **100 completed validation runs**:

| Metric | Min | Average | Max |
|---|---:|---:|---:|
| `accuracy` | `99.30` | `99.846` | `100.00` |
| `yield_mse` | `0.0091` | `0.01020` | `0.0115` |
| `combined_score` | `99.30` | `99.846` | `100.00` |
| `memory_gb` | `2.87` | `2.87` | `2.87` |

Additional notes:

- All **100 / 100** experiments completed successfully.
- **27 runs** reached `100.00` validation accuracy.
- The best observed `yield_mse` was `0.0091`.
- The runs used the same committed training setup, so these results mainly show **stability/repeatability** of the current model pipeline rather than a comparison between many different model architectures.
- The model used a pretrained computer-vision backbone and was evaluated on the validation split produced by `train.py`.

## Brief Layman Summary

In simple terms, we taught a computer to look at pictures of waste and sort them into useful categories: electronics, recyclable materials, or organic waste. We then ran the training/evaluation process 100 times on a cloud GPU to see how reliably the current setup performs.

The result was very strong: the model repeatedly scored around **99.8% validation accuracy**, with several runs reaching **100%** on the validation split. This means the current pipeline is working and produces consistent metrics for the waste-classification task.

## Project Structure

```text
train.py        — model definition, data loading, training loop, validation metrics
modal_app.py    — Modal GPU runner for cloud training
prepare.py      — reference/helper preparation code
program.md      — autoresearch instructions for an AI coding agent
results.tsv     — committed May 22 experiment metrics
run.log         — local/cloud run log output, ignored by Git
```

## Metrics

The training script reports:

- **`val_accuracy` / `accuracy`**: classification accuracy on the validation split. Higher is better.
- **`yield_mse`**: mean squared error for the yield prediction head. Lower is better.
- **`combined_score`**: overall score, calculated as:

```text
combined_score = accuracy - 0.1 * yield_mse
```

Because `yield_mse` is very small in the current runs, `combined_score` is nearly identical to `accuracy`.

## Can the Model Be Tested?

Yes, testing is possible, but there are two different meanings of “testing”:

### 1. Validation testing with the current script

This is already implemented.

Running `train.py` trains the model and then evaluates it on a held-out validation split. At the end, it prints metrics like:

```text
val_accuracy:     99.90
yield_mse:        0.0099
combined_score:   99.90
training_seconds: ...
peak_vram_mb:     ...
```

This is the type of testing used to produce `results.tsv`.

### 2. Testing one custom image manually

A standalone image-testing/inference command is **not implemented yet**.

The model training code saves a checkpoint to:

```text
~/.cache/autoresearch-waste/best_model.pt
```

However, the repository currently does not include a separate CLI such as:

```text
python predict.py path/to/image.jpg
```

So, to test an arbitrary new image, the next step would be to add a small `predict.py` script that:

1. loads the same model architecture,
2. loads `best_model.pt`,
3. preprocesses the image,
4. prints the predicted class and yield estimate.

## Quick Start: Local Validation Run

Install the core Python dependencies:

```bash
pip install torch torchvision datasets Pillow numpy
```

Then run:

```bash
python3 train.py
```

Notes:

- Local CPU testing is possible, but it may be slow.
- A GPU is recommended for realistic training time.
- If the HuggingFace dataset cannot be downloaded, the script falls back to synthetic placeholder data. That fallback is useful for checking that the code runs, but its metrics should not be treated as real model quality.

## Quick Start: Cloud GPU Run with Modal

Install and authenticate Modal:

```bash
pip install modal
modal token new
```

Run the cloud training job:

```bash
modal run modal_app.py
```

The Modal runner executes `train.py` on a T4 GPU and prints the same metrics to the job logs.

## Datasets

The training code currently loads data through HuggingFace `datasets`, primarily:

- `NeoAivara/Waste_Classification_data`

The labels are mapped into the three target categories: e-waste, recyclable/dry waste, and organic waste.

## Current Limitations

- `results.tsv` records validation metrics, not field deployment performance.
- The yield target is simulated from category-based assumptions inside `train.py`; it is not yet based on real measured recovery/yield labels.
- There is no dedicated `predict.py` CLI yet for testing arbitrary individual images.
- The May 22 results are repeated runs of the same training setup, so they show consistency more than autonomous model-discovery progress.

## License

MIT

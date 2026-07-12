# Particle track reconstruction with GNNs

A Graph Neural Network approach to hit-to-track association, trained and
evaluated on simulated straight-line and magnetically curved particle
trajectories.

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

Run the full pipeline (data generation, training/loading, testing, and
prediction export):

```bash
uv run main.py
```

By default, `main.py` loads pretrained model weights from `models/` rather
than retraining. To retrain from scratch, set `train = True` in the
hyperparameters section at the top of `main.py`:

```python
if __name__ == "__main__":
    train = True   # set to True to retrain instead of loading saved weights
    ...
```

## Outputs

All generated files are written to `results/`, including:

- Training curves (loss / F1-score per epoch)
- Example event visualizations (ground truth vs. predicted track assignments)
- Hit distribution histograms
- `predictions.csv` — per-hit predicted track assignments for the curved-track
  test set, submitted as required deliverable output
- additionally, `example.py` shows how to load the models.

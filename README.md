# Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity

This repository contains reference implementations and experiment scripts for the paper *Enabling Robust In-Context Memory and Rapid Task Adaptation in Transformers with Hebbian and Gradient-Based Plasticity*.  The code equips decoder-only Transformers with fast-weight components that are updated via neuromodulated Hebbian or gradient-based plasticity rules and evaluates them on the suite of tasks introduced by Duan et al.\ (2023).

## Repository Layout

```
.
├── requirements.txt              # Python dependencies
├── src/
│   ├── models/                    # Plastic Transformer and Conv-4 encoder
│   ├── tasks/                     # Task generators and datasets
│   └── experiments/               # CLI entry points for each benchmark
├── experiments/                   # JSON logs written by the runners
├── scripts/                       # Result aggregation and plotting utilities
├── figures/                       # Generated figures (after running plot_results.py)
└── README.md
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Tip:** The experiment entry points use package imports (`src.*`).  Run all commands from the repository root so that Python can resolve the package correctly.

## Running Experiments

Each experiment script accepts a consistent set of flags:

- `--rule {none, hebbian, gradient}` selects the plasticity rule.
- `--base-seed` sets the first random seed (default `123`).
- `--seeds` controls how many additional seeds to run (`base_seed + i` for `i in [0, seeds-1]`).
- `--output-path` writes a JSON log containing per-seed histories and aggregate statistics.
- `--device {auto, cpu, cuda, mps}` selects the hardware backend (`auto` prefers CUDA → MPS → CPU).

Example usages are shown below.

### Copying Task

```bash
python -m src.experiments.copying \
  --rule gradient \
  --seq-length 5 \
  --delay 20 \
  --seeds 3 \
  --output-path experiments/results/copy_delay20_rule-gradient.json
```

### Cue–Reward Association

```bash
python -m src.experiments.cue_reward \
  --rule hebbian \
  --num-pairs 8 \
  --cue-dim 20 \
  --seeds 3 \
  --output-path experiments/results/cue_rule-hebbian.json
```

### Few-Shot Regression

```bash
python -m src.experiments.few_shot_regression \
  --rule hebbian \
  --k-support 10 \
  --k-query 10 \
  --seeds 3 \
  --output-path experiments/results/regression_rule-hebbian.json
```

### Few-Shot Image Classification (CIFAR-FS / Omniglot)

```bash
python -m src.experiments.one_shot_classification \
  --rule hebbian \
  --dataset cifarfs \
  --ways 5 --shots 1 --queries 15 \
  --epochs 20 --episodes-per-epoch 200 \
  --seeds 3 \
  --output-path experiments/results/classification_cifarfs_rule-hebbian.json
```

Set `--dataset omniglot` to train on Omniglot.  Torchvision automatically downloads CIFAR-100 and Omniglot into `--data-root` (default `./data`).

## Working with Results

1. **Aggregate multiple runs**

   ```bash
   python scripts/aggregate_results.py
   ```

   This scans `experiments/results/*.json` and produces `experiments/results/summary.json` with per-task, per-rule aggregates.

2. **Regenerate figures**

   ```bash
   python scripts/plot_results.py
   ```

   The script reads from `experiments/results/` and writes publication-ready plots to `figures/`.

3. **Build auxiliary tables**

   - `python scripts/build_tables.py` collects metrics into CSV/Markdown tables.
   - `python scripts/compile_baseline_table.py` reproduces the cross-architecture comparison table.

All result JSON files follow the same schema:

```json
{
  "config": { "model": {...}, "task": {...}, "training": {...} },
  "runs": [
    {"seed": 3000, "history": [...], "final": {...}},
    ...
  ],
  "aggregate": {
    "loss_mean": 0.352,
    "loss_std": 0.021,
    ...
  }
}
```

## Reproducing the Paper Pipeline

1. Run each task for the rules (`none`, `hebbian`, `gradient`) with three seeds, saving outputs under `experiments/results/`.
2. Execute `python scripts/aggregate_results.py`.
3. Generate plots via `python scripts/plot_results.py`.
4. Update the manuscript or slides using the refreshed tables (`scripts/build_tables.py`) and figures (`figures/`).

The complete campaign (copying, cue–reward, regression, CIFAR-FS, and Omniglot with three seeds each) consumes roughly 25 GPU-hours on a single NVIDIA A100 (40 GB), including diagnostics.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`** — Ensure commands are launched from the repository root or set `PYTHONPATH=.`.
- **CUDA requested but not available** — Use `--device cpu` or install the appropriate CUDA toolkit/driver.
- **Dataset download stalls** — Torchvision downloads CIFAR-100 and Omniglot automatically; if mirrors are blocked, manually place the archives in `./data` and rerun.

For any other issues, inspect the JSON logs in `experiments/results/` (they capture per-epoch losses, neuromodulation statistics, and plastic weight norms for each seed).


from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("experiments/results")
FIGURES_DIR = Path("figures")


@dataclass
class ResultRecord:
    file: Path
    experiment: str
    rule: str
    dataset: Optional[str]
    aggregate: Dict[str, float]
    runs: List[Dict]


def determine_experiment(config: Dict) -> str:
    task_cfg = config.get("task", {}) or {}
    if "delay" in task_cfg and "seq_length" in task_cfg:
        return "copying"
    if "num_pairs" in task_cfg and "cue_dim" in task_cfg:
        return "cue_reward"
    if "k_support" in task_cfg and "k_query" in task_cfg:
        return "few_shot_regression"
    if "train" in task_cfg and isinstance(task_cfg.get("train"), dict):
        dataset = task_cfg["train"].get("dataset")
        return f"classification_{dataset}" if dataset else "classification"
    return "unknown"


def extract_dataset(config: Dict) -> Optional[str]:
    task_cfg = config.get("task", {}) or {}
    train_cfg = task_cfg.get("train", {}) or {}
    dataset = train_cfg.get("dataset")
    return dataset


def load_results() -> List[ResultRecord]:
    if not RESULTS_DIR.exists():
        raise SystemExit(f"Results directory '{RESULTS_DIR}' not found.")

    records: List[ResultRecord] = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("config", {}) or {}
        aggregate = data.get("aggregate", {}) or {}
        runs = data.get("runs", []) or []
        experiment = determine_experiment(config)
        dataset = extract_dataset(config)
        rule = (config.get("model", {}) or {}).get("rule", "unknown")
        records.append(
            ResultRecord(
                file=path,
                experiment=experiment,
                rule=rule,
                dataset=dataset,
                aggregate=aggregate,
                runs=runs,
            )
        )
    return records


def group_by_experiment(records: List[ResultRecord]) -> Dict[str, Dict[str, ResultRecord]]:
    grouped: Dict[str, Dict[str, ResultRecord]] = {}
    for record in records:
        grouped.setdefault(record.experiment, {})
        grouped[record.experiment][record.rule] = record
    return grouped


def prepare_performance_data(
    grouped: Dict[str, Dict[str, ResultRecord]],
) -> Tuple[List[Dict], List[Dict]]:
    acc_specs = [
        {
            "experiment": "copying",
            "title": "Copying (delay 20)",
            "mean_key": "recall_accuracy_mean",
            "std_key": "recall_accuracy_std",
            "ylabel": "Recall ↑",
        },
        {
            "experiment": "classification_cifarfs",
            "title": "CIFAR-FS 5-way/1-shot",
            "mean_key": "accuracy_mean",
            "std_key": "accuracy_std",
            "ylabel": "Accuracy ↑",
        },
        {
            "experiment": "classification_omniglot",
            "title": "Omniglot 5-way/1-shot",
            "mean_key": "accuracy_mean",
            "std_key": "accuracy_std",
            "ylabel": "Accuracy ↑",
        },
    ]

    loss_specs = [
        {
            "experiment": "cue_reward",
            "title": "Cue-Reward Association",
            "mean_key": "query_loss_mean",
            "std_key": "query_loss_std",
            "ylabel": "Query Loss ↓",
        },
        {
            "experiment": "few_shot_regression",
            "title": "Few-Shot Regression (10-shot)",
            "mean_key": "query_loss_mean",
            "std_key": "query_loss_std",
            "ylabel": "Query MSE ↓",
        },
    ]

    def fill_spec(specs: List[Dict]) -> List[Dict]:
        filled: List[Dict] = []
        for spec in specs:
            experiment = spec["experiment"]
            if experiment not in grouped:
                continue
            filled.append(spec)
        return filled

    return fill_spec(acc_specs), fill_spec(loss_specs)


def plot_performance(grouped: Dict[str, Dict[str, ResultRecord]]) -> None:
    acc_specs, loss_specs = prepare_performance_data(grouped)
    rules = ["gradient", "hebbian", "none"]
    rule_labels = {"gradient": "Gradient", "hebbian": "Hebbian", "none": "None"}
    colors = {"gradient": "#1f77b4", "hebbian": "#ff7f0e", "none": "#2ca02c"}

    # Accuracy-oriented figure
    if acc_specs:
        fig, axes = plt.subplots(1, len(acc_specs), figsize=(4 * len(acc_specs), 3), sharey=False)
        if len(acc_specs) == 1:
            axes = [axes]
        for ax, spec in zip(axes, acc_specs):
            experiment = spec["experiment"]
            records = grouped.get(experiment, {})
            values = []
            errors = []
            labels = []
            bar_colors = []
            for rule in rules:
                record = records.get(rule)
                if not record:
                    continue
                labels.append(rule_labels[rule])
                values.append(record.aggregate.get(spec["mean_key"], np.nan))
                errors.append(record.aggregate.get(spec["std_key"], 0.0))
                bar_colors.append(colors.get(rule, "#555555"))
            positions = np.arange(len(values))
            ax.bar(positions, values, yerr=errors, color=bar_colors, capsize=4)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=20)
            ax.set_title(spec["title"])
            ax.set_ylabel(spec["ylabel"])
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / "performance_accuracy.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "performance_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Loss-oriented figure
    if loss_specs:
        fig, axes = plt.subplots(1, len(loss_specs), figsize=(4 * len(loss_specs), 3), sharey=False)
        if len(loss_specs) == 1:
            axes = [axes]
        for ax, spec in zip(axes, loss_specs):
            experiment = spec["experiment"]
            records = grouped.get(experiment, {})
            values = []
            errors = []
            labels = []
            bar_colors = []
            for rule in rules:
                record = records.get(rule)
                if not record:
                    continue
                labels.append(rule_labels[rule])
                values.append(record.aggregate.get(spec["mean_key"], np.nan))
                errors.append(record.aggregate.get(spec["std_key"], 0.0))
                bar_colors.append(colors.get(rule, "#555555"))
            positions = np.arange(len(values))
            ax.bar(positions, values, yerr=errors, color=bar_colors, capsize=4)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=20)
            ax.set_title(spec["title"])
            ax.set_ylabel(spec["ylabel"])
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        FIGURES_DIR.mkdir(exist_ok=True)
        fig.savefig(FIGURES_DIR / "performance_losses.pdf", bbox_inches="tight")
        fig.savefig(FIGURES_DIR / "performance_losses.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def aggregate_epoch_series(record: ResultRecord, metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    epoch_values: Dict[int, List[float]] = {}
    for run in record.runs:
        for entry in run.get("history", []):
            epoch = entry.get("epoch")
            if epoch is None:
                continue
            value = entry.get(metric)
            if value is None:
                continue
            epoch_values.setdefault(epoch, []).append(value)
    if not epoch_values:
        return np.array([]), np.array([]), np.array([])
    epochs = sorted(epoch_values.keys())
    means = np.array([np.mean(epoch_values[e]) for e in epochs])
    stds = np.array([np.std(epoch_values[e]) for e in epochs])
    return np.array(epochs), means, stds


def plot_mechanistic_traces(grouped: Dict[str, Dict[str, ResultRecord]]) -> None:
    targets = [
        ("copying", "Copying (delay 20)"),
        ("classification_cifarfs", "CIFAR-FS 5-way/1-shot"),
    ]
    metrics = [
        ("eta_mean", "Neuromodulation $\\eta(t)$"),
        ("plastic_norm_mean", "Plastic Weight Frobenius Norm"),
    ]
    rules = ["gradient", "hebbian", "none"]
    rule_labels = {"gradient": "Gradient", "hebbian": "Hebbian", "none": "None"}
    colors = {"gradient": "#1f77b4", "hebbian": "#ff7f0e", "none": "#2ca02c"}

    num_cols = len(targets)
    num_rows = len(metrics)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows), sharex=False)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for col, (experiment, title) in enumerate(targets):
        records = grouped.get(experiment, {})
        if not records:
            continue
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for rule in rules:
                record = records.get(rule)
                if not record:
                    continue
                epochs, means, stds = aggregate_epoch_series(record, metric)
                if epochs.size == 0:
                    continue
                ax.plot(epochs, means, label=rule_labels[rule], color=colors.get(rule, "#555555"), marker="o")
                ax.fill_between(
                    epochs,
                    means - stds,
                    means + stds,
                    color=colors.get(rule, "#555555"),
                    alpha=0.2,
                )
            ax.set_title(title if row == 0 else "")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle="--", alpha=0.3)
            if row == 0 and col == num_cols - 1:
                ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / "mechanistic_traces.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "mechanistic_traces.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records = load_results()
    grouped = group_by_experiment(records)
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_performance(grouped)
    plot_mechanistic_traces(grouped)
    print(f"Wrote figures to '{FIGURES_DIR}'")


if __name__ == "__main__":
    main()

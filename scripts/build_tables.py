from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def format_mean_std(mean: float, std: float, precision: int = 3) -> str:
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def load_summary(summary_path: Path) -> Dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_copying_table(summary: Dict) -> str:
    rows: List[str] = []
    experiment_key = "copying"
    if experiment_key not in summary:
        return "% Copying summary not available\n"
    header = "\\begin{tabular}{lcc}\n\\toprule\nRule & Loss & Recall Acc. \\\\\n\\midrule\n"
    footer = "\\bottomrule\n\\end{tabular}\n"
    for rule, info in summary[experiment_key].items():
        agg = info.get("aggregate", {})
        loss = format_mean_std(agg.get("loss_mean", 0.0), agg.get("loss_std", 0.0))
        recall = format_mean_std(agg.get("recall_accuracy_mean", 0.0), agg.get("recall_accuracy_std", 0.0))
        rows.append(f"{rule.title()} & {loss} & {recall} \\\\")
    body = "\n".join(rows) + "\n"
    return header + body + footer


def build_cue_table(summary: Dict) -> str:
    rows: List[str] = []
    experiment_key = "cue_reward"
    if experiment_key not in summary:
        return "% Cue-reward summary not available\n"
    header = "\\begin{tabular}{lcc}\n\\toprule\nRule & Validation Loss & Query Loss \\\\\n\\midrule\n"
    footer = "\\bottomrule\n\\end{tabular}\n"
    for rule, info in summary[experiment_key].items():
        agg = info.get("aggregate", {})
        loss = format_mean_std(agg.get("loss_mean", 0.0), agg.get("loss_std", 0.0))
        query = format_mean_std(agg.get("query_loss_mean", 0.0), agg.get("query_loss_std", 0.0))
        rows.append(f"{rule.title()} & {loss} & {query} \\\\")
    body = "\n".join(rows) + "\n"
    return header + body + footer


def build_regression_table(summary: Dict) -> str:
    rows: List[str] = []
    experiment_key = "few_shot_regression"
    if experiment_key not in summary:
        return "% Regression summary not available\n"
    header = "\\begin{tabular}{lcc}\n\\toprule\nRule & Val. MSE & Query MSE \\\\\n\\midrule\n"
    footer = "\\bottomrule\n\\end{tabular}\n"
    for rule, info in summary[experiment_key].items():
        agg = info.get("aggregate", {})
        val_mse = format_mean_std(agg.get("loss_mean", 0.0), agg.get("loss_std", 0.0))
        query_mse = format_mean_std(agg.get("query_loss_mean", 0.0), agg.get("query_loss_std", 0.0))
        rows.append(f"{rule.title()} & {val_mse} & {query_mse} \\\\")
    body = "\n".join(rows) + "\n"
    return header + body + footer


def build_classification_table(summary: Dict, dataset: str) -> str:
    experiment_key = f"classification_{dataset}"
    if experiment_key not in summary:
        return f"% Classification summary not available for {dataset}\n"
    header = "\\begin{tabular}{lc}\n\\toprule\nRule & Accuracy \\\\\n\\midrule\n"
    footer = "\\bottomrule\n\\end{tabular}\n"
    rows: List[str] = []
    for rule, info in summary[experiment_key].items():
        agg = info.get("aggregate", {})
        acc = format_mean_std(agg.get("accuracy_mean", 0.0), agg.get("accuracy_std", 0.0))
        rows.append(f"{rule.title()} & {acc} \\\\")
    body = "\n".join(rows) + "\n"
    return header + body + footer


def write_tables(summary: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "copying.tex": build_copying_table(summary),
        "cue_reward.tex": build_cue_table(summary),
        "regression.tex": build_regression_table(summary),
        "classification_cifarfs.tex": build_classification_table(summary, "cifarfs"),
        "classification_omniglot.tex": build_classification_table(summary, "omniglot"),
    }
    for filename, content in tables.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
        print(f"Wrote {output_dir / filename}")


def main() -> None:
    results_dir = Path("experiments/results")
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Summary file '{summary_path}' not found. Run aggregate_results.py first.")
    summary = load_summary(summary_path)
    write_tables(summary, Path("experiments/tables"))


if __name__ == "__main__":
    main()

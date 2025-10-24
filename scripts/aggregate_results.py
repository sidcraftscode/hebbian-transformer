from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ResultRecord:
    file: Path
    experiment: str
    rule: str
    dataset: Optional[str]
    aggregate: Dict[str, float]
    task_config: Dict
    training_config: Dict


def determine_experiment(config: Dict) -> str:
    task = config.get("task", {})
    if "delay" in task and "seq_length" in task:
        return "copying"
    if "num_pairs" in task and "cue_dim" in task:
        return "cue_reward"
    if "train" in task and isinstance(task["train"], dict):
        dataset = task["train"].get("dataset", "")
        return f"classification_{dataset}" if dataset else "classification"
    if "k_support" in task and "k_query" in task:
        return "few_shot_regression"
    return "unknown"


def extract_dataset(experiment: str, config: Dict) -> Optional[str]:
    if experiment.startswith("classification"):
        task = config.get("task", {})
        if isinstance(task, dict):
            train_cfg = task.get("train", {})
            if isinstance(train_cfg, dict):
                return train_cfg.get("dataset")
    return None


def load_results(results_dir: Path) -> List[ResultRecord]:
    records: List[ResultRecord] = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("config", {})
        aggregate = data.get("aggregate", {})
        task_cfg = config.get("task", {})
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        experiment = determine_experiment(config)
        dataset = extract_dataset(experiment, config)
        records.append(
            ResultRecord(
                file=path,
                experiment=experiment,
                rule=model_cfg.get("rule", "unknown"),
                dataset=dataset,
                aggregate=aggregate,
                task_config=task_cfg,
                training_config=training_cfg,
            )
        )
    return records


def summarise(records: List[ResultRecord]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for record in records:
        key = record.experiment
        summary.setdefault(key, {})
        summary[key][record.rule] = {
            "dataset": record.dataset,
            "aggregate": record.aggregate,
            "task_config": record.task_config,
            "training_config": record.training_config,
            "source_file": record.file.name,
        }
    return summary


def main() -> None:
    results_dir = Path("experiments/results")
    if not results_dir.exists():
        raise SystemExit(f"Results directory '{results_dir}' not found.")
    records = load_results(results_dir)
    summary = summarise(records)
    output_path = results_dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary for {len(records)} runs to {output_path}")


if __name__ == "__main__":
    main()

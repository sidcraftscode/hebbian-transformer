import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.plastic_transformer import PlasticTransformerModel
from src.tasks.copying import CopyingTaskConfig, CopyingTaskDataset


def resolve_device(preferred: str) -> torch.device:
    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available.")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def summarise(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_metrics(finals: List[Dict[str, float]]) -> Dict[str, float]:
    if not finals:
        return {}
    keys = finals[0].keys()
    aggregate: Dict[str, float] = {"num_runs": len(finals)}
    for key in keys:
        vals = [run[key] for run in finals if isinstance(run.get(key), (int, float))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        aggregate[f"{key}_mean"] = float(arr.mean())
        aggregate[f"{key}_std"] = float(arr.std(ddof=0))
    return aggregate


@dataclass
class TrainingConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_norm: float = 5.0
    log_interval: int = 50


def train_epoch(
    model: PlasticTransformerModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    clip_norm: float,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    eta_values: List[float] = []
    plastic_values: List[float] = []
    progress = tqdm(dataloader, desc="train", leave=False)
    for inputs, targets in progress:
        inputs = inputs.squeeze(0).long().to(device)
        targets = targets.squeeze(0).long().to(device)
        state = model.init_state(device)
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for t in range(inputs.shape[0]):
            x_vec = F.one_hot(inputs[t], num_classes=model.output_dim).float().to(device)
            outputs = model.forward_step(x_vec, state)
            logits = outputs["logits"].unsqueeze(0)
            step_target = targets[t].unsqueeze(0)
            step_loss = loss_fn(logits, step_target)
            loss = loss + step_loss
            eta_values.append(float(outputs["eta"].item()))
            plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()
        total_steps += inputs.shape[0]
        if total_steps > 0:
            progress.set_postfix(loss=loss.item() / inputs.shape[0])
    diag = {
        "eta_mean": summarise(eta_values)[0],
        "eta_std": summarise(eta_values)[1],
        "plastic_norm_mean": summarise(plastic_values)[0],
        "plastic_norm_std": summarise(plastic_values)[1],
    }
    return total_loss / max(total_steps, 1), diag


def evaluate(
    model: PlasticTransformerModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    correct_recall = 0
    total_recall = 0
    eta_values: List[float] = []
    plastic_values: List[float] = []
    context = torch.enable_grad if model.rule == "gradient" else torch.no_grad
    with context():
        for inputs, targets in dataloader:
            inputs = inputs.squeeze(0).long().to(device)
            targets = targets.squeeze(0).long().to(device)
            state = model.init_state(device)
            trial_loss = 0.0
            for t in range(inputs.shape[0]):
                x_vec = F.one_hot(inputs[t], num_classes=model.output_dim).float().to(device)
                outputs = model.forward_step(x_vec, state)
                logits = outputs["logits"]
                step_target = targets[t].unsqueeze(0)
                trial_loss += loss_fn(logits.unsqueeze(0), step_target).item()
                eta_values.append(float(outputs["eta"].item()))
                plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
                if t >= inputs.shape[0] // 2:
                    pred = logits.argmax().item()
                    if pred == targets[t].item():
                        correct_recall += 1
                    total_recall += 1
            losses.append(trial_loss / inputs.shape[0])
    avg_loss = sum(losses) / max(len(losses), 1)
    recall_acc = correct_recall / max(total_recall, 1)
    eta_mean, eta_std = summarise(eta_values)
    plastic_mean, plastic_std = summarise(plastic_values)
    return {
        "loss": avg_loss,
        "recall_accuracy": recall_acc,
        "eta_mean": eta_mean,
        "eta_std": eta_std,
        "plastic_norm_mean": plastic_mean,
        "plastic_norm_std": plastic_std,
    }


def execute_single_run(args: argparse.Namespace, device: torch.device, seed: int) -> Dict[str, List[Dict[str, float]]]:
    set_seed(seed)
    copying_cfg = CopyingTaskConfig(
        seq_length=args.seq_length,
        delay=args.delay,
        vocab_size=args.vocab_size,
        dataset_size=args.dataset_size,
    )
    train_dataset = CopyingTaskDataset(copying_cfg)
    val_dataset = CopyingTaskDataset(copying_cfg)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = PlasticTransformerModel(
        input_dim=args.vocab_size,
        output_dim=args.vocab_size,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        aux_dim=args.aux_dim,
        rule=args.rule,
        eta0=args.eta0,
        max_norm=args.max_norm,
    ).to(device)

    train_cfg = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_norm=args.clip_norm,
        log_interval=args.log_interval,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    for epoch in range(train_cfg.epochs):
        train_loss, train_diag = train_epoch(model, train_loader, device, loss_fn, train_cfg.clip_norm, optimizer)
        metrics = evaluate(model, val_loader, device, loss_fn)
        metrics["train_loss"] = train_loss
        metrics["train_eta_mean"] = train_diag["eta_mean"]
        metrics["train_eta_std"] = train_diag["eta_std"]
        metrics["train_plastic_norm_mean"] = train_diag["plastic_norm_mean"]
        metrics["train_plastic_norm_std"] = train_diag["plastic_norm_std"]
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(
            f"[seed={seed}] Epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"val_loss={metrics['loss']:.4f}, recall_acc={metrics['recall_accuracy']:.4f}"
        )
    return {
        "history": history,
        "final": history[-1],
        "config": asdict(copying_cfg),
        "training": asdict(train_cfg),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, float]:
    device = resolve_device(args.device)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    run_summaries = []
    final_metrics: List[Dict[str, float]] = []
    task_config = None
    training_config = None

    for seed in seeds:
        summary = execute_single_run(args, device, seed)
        run_summaries.append({"seed": seed, "history": summary["history"], "final": summary["final"]})
        final_metrics.append(summary["final"])
        if task_config is None:
            task_config = summary["config"]
        if training_config is None:
            training_config = summary["training"]

    aggregate = aggregate_metrics(final_metrics)
    last_final = final_metrics[-1] if final_metrics else {}

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        payload = {
            "config": {
                "model": {
                    "rule": args.rule,
                    "model_dim": args.model_dim,
                    "num_heads": args.num_heads,
                    "num_layers": args.num_layers,
                    "ffn_dim": args.ffn_dim,
                },
                "task": task_config,
                "training": training_config,
            },
            "runs": run_summaries,
            "aggregate": aggregate,
        }
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return last_final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copying task experiment runner")
    parser.add_argument("--rule", choices=["none", "hebbian", "gradient"], default="hebbian")
    parser.add_argument("--seq-length", type=int, default=5)
    parser.add_argument("--delay", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=10)
    parser.add_argument("--dataset-size", type=int, default=500)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-dim", type=int, default=4)
    parser.add_argument("--eta0", type=float, default=0.2)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

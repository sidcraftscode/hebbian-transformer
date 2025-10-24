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
from tqdm import tqdm

from src.models.conv_encoder import Conv4Encoder
from src.models.plastic_transformer import PlasticTransformerModel
from src.tasks.few_shot_classification import (
    FewShotClassificationConfig,
    FewShotClassificationTask,
)


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
    aggregate: Dict[str, float] = {"num_runs": len(finals)}
    keys = finals[0].keys()
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
    epochs: int = 20
    episodes_per_epoch: int = 200
    val_episodes: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-4
    clip_norm: float = 5.0


def build_support_vector(embedding: torch.Tensor, label_one_hot: torch.Tensor) -> torch.Tensor:
    return torch.cat([embedding, label_one_hot, torch.zeros(1, device=embedding.device)])


def build_query_vector(embedding: torch.Tensor, ways: int) -> torch.Tensor:
    return torch.cat(
        [
            embedding,
            torch.zeros(ways, device=embedding.device),
            torch.ones(1, device=embedding.device),
        ]
    )


def train_epoch(
    encoder: nn.Module,
    transformer: PlasticTransformerModel,
    task: FewShotClassificationTask,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: TrainingConfig,
    ways: int,
) -> Tuple[float, Dict[str, float]]:
    encoder.train()
    transformer.train()
    total_loss = 0.0
    eta_values: List[float] = []
    plastic_values: List[float] = []
    for _ in tqdm(range(config.episodes_per_epoch), desc="train episodes", leave=False):
        support_images, support_labels, query_images, query_labels, _ = task.sample_episode()
        support_images = support_images.to(device)
        query_images = query_images.to(device)
        support_labels = support_labels.to(device)
        query_labels = query_labels.to(device)

        support_embeddings = encoder(support_images)
        query_embeddings = encoder(query_images)
        state = transformer.init_state(device)
        optimizer.zero_grad()
        episode_loss = torch.tensor(0.0, device=device)
        for idx in range(support_embeddings.shape[0]):
            embedding = support_embeddings[idx]
            label = support_labels[idx]
            one_hot = F.one_hot(label, num_classes=ways).float()
            step_vec = build_support_vector(embedding, one_hot)
            outputs = transformer.forward_step(step_vec, state)
            eta_values.append(float(outputs["eta"].item()))
            plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
        for idx in range(query_embeddings.shape[0]):
            embedding = query_embeddings[idx]
            label = query_labels[idx]
            step_vec = build_query_vector(embedding, ways)
            outputs = transformer.forward_step(step_vec, state)
            logits = outputs["logits"].unsqueeze(0)
            episode_loss = episode_loss + loss_fn(logits, label.unsqueeze(0))
            eta_values.append(float(outputs["eta"].item()))
            plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
        episode_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(transformer.parameters()), config.clip_norm)
        optimizer.step()
        total_loss += episode_loss.item() / max(query_embeddings.shape[0], 1)
    diag = {
        "eta_mean": summarise(eta_values)[0],
        "eta_std": summarise(eta_values)[1],
        "plastic_norm_mean": summarise(plastic_values)[0],
        "plastic_norm_std": summarise(plastic_values)[1],
    }
    return total_loss / max(config.episodes_per_epoch, 1), diag


def evaluate(
    encoder: nn.Module,
    transformer: PlasticTransformerModel,
    task: FewShotClassificationTask,
    device: torch.device,
    loss_fn: nn.Module,
    ways: int,
    num_episodes: int,
) -> Dict[str, float]:
    encoder.eval()
    transformer.eval()
    losses: List[float] = []
    accuracies: List[float] = []
    eta_values: List[float] = []
    plastic_values: List[float] = []
    context = torch.enable_grad if transformer.rule == "gradient" else torch.no_grad
    with context():
        for _ in tqdm(range(num_episodes), desc="eval episodes", leave=False):
            support_images, support_labels, query_images, query_labels, _ = task.sample_episode()
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

            support_embeddings = encoder(support_images)
            query_embeddings = encoder(query_images)
            state = transformer.init_state(device)
            loss = 0.0
            correct = 0
            total = 0

            for idx in range(support_embeddings.shape[0]):
                embedding = support_embeddings[idx]
                label = support_labels[idx]
                one_hot = F.one_hot(label, num_classes=ways).float()
                step_vec = build_support_vector(embedding, one_hot)
                outputs = transformer.forward_step(step_vec, state)
                eta_values.append(float(outputs["eta"].item()))
                plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))

            for idx in range(query_embeddings.shape[0]):
                embedding = query_embeddings[idx]
                label = query_labels[idx]
                step_vec = build_query_vector(embedding, ways)
                outputs = transformer.forward_step(step_vec, state)
                logits = outputs["logits"]
                loss += loss_fn(logits.unsqueeze(0), label.unsqueeze(0)).item()
                pred = logits.argmax().item()
                correct += int(pred == label.item())
                total += 1
                eta_values.append(float(outputs["eta"].item()))
                plastic_values.append(float(outputs["diagnostics"]["plastic_norm"].item()))
            losses.append(loss / max(total, 1))
            accuracies.append(correct / max(total, 1))
    eta_mean, eta_std = summarise(eta_values)
    plastic_mean, plastic_std = summarise(plastic_values)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": sum(accuracies) / max(len(accuracies), 1),
        "eta_mean": eta_mean,
        "eta_std": eta_std,
        "plastic_norm_mean": plastic_mean,
        "plastic_norm_std": plastic_std,
    }


def execute_single_run(args: argparse.Namespace, device: torch.device, seed: int) -> Dict[str, List[Dict[str, float]]]:
    set_seed(seed)
    train_task = FewShotClassificationTask(
        FewShotClassificationConfig(
            root=args.data_root,
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="train",
            dataset=args.dataset,
        )
    )
    val_task = FewShotClassificationTask(
        FewShotClassificationConfig(
            root=args.data_root,
            ways=args.ways,
            shots=args.shots,
            queries=args.queries,
            split="test",
            dataset=args.dataset,
        )
    )

    encoder = Conv4Encoder(
        in_channels=train_task.input_channels,
        output_dim=args.embedding_dim,
        input_resolution=train_task.input_resolution,
    ).to(device)
    transformer = PlasticTransformerModel(
        input_dim=args.embedding_dim + args.ways + 1,
        output_dim=args.ways,
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
        episodes_per_epoch=args.episodes_per_epoch,
        val_episodes=args.val_episodes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_norm=args.clip_norm,
    )
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(transformer.parameters()),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    for epoch in range(train_cfg.epochs):
        train_loss, train_diag = train_epoch(
            encoder,
            transformer,
            train_task,
            device,
            optimizer,
            loss_fn,
            train_cfg,
            args.ways,
        )
        metrics = evaluate(
            encoder,
            transformer,
            val_task,
            device,
            loss_fn,
            args.ways,
            train_cfg.val_episodes,
        )
        metrics["train_loss"] = train_loss
        metrics["train_eta_mean"] = train_diag["eta_mean"]
        metrics["train_eta_std"] = train_diag["eta_std"]
        metrics["train_plastic_norm_mean"] = train_diag["plastic_norm_mean"]
        metrics["train_plastic_norm_std"] = train_diag["plastic_norm_std"]
        metrics["epoch"] = epoch + 1
        history.append(metrics)
        print(
            f"[seed={seed}] Epoch {epoch+1}: train_loss={train_loss:.4f}, "
            f"val_loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}"
        )
    return {
        "history": history,
        "final": history[-1],
        "task": {
            "train": asdict(train_task.config),
            "val": asdict(val_task.config),
        },
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
            task_config = summary["task"]
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
                    "embedding_dim": args.embedding_dim,
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
    parser = argparse.ArgumentParser(description="Few-shot classification experiment")
    parser.add_argument("--rule", choices=["none", "hebbian", "gradient"], default="gradient")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--dataset", choices=["cifarfs", "omniglot"], default="cifarfs")
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux-dim", type=int, default=4)
    parser.add_argument("--eta0", type=float, default=0.2)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes-per-epoch", type=int, default=200)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)

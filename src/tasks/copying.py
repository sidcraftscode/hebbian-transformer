from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CopyingTaskConfig:
    seq_length: int = 5
    delay: int = 5
    vocab_size: int = 8
    dataset_size: int = 2000
    device: str = "cpu"


class CopyingTaskDataset(Dataset):
    """
    Generates sequences for the copying task. Each trial consists of:

    - seq_length random tokens sampled from {1, ..., vocab_size - 2}
    - delay tokens of the blank symbol (0)
    - a delimiter token (vocab_size - 1) signalling recall
    - seq_length blank tokens where the target is the original sequence
    """

    def __init__(self, config: CopyingTaskConfig) -> None:
        self.config = config
        self.total_length = config.seq_length * 2 + config.delay + 1

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        seq = torch.randint(
            low=1,
            high=cfg.vocab_size - 1,
            size=(cfg.seq_length,),
            dtype=torch.long,
        )
        blank = torch.zeros(cfg.delay, dtype=torch.long)
        delimiter = torch.full((1,), cfg.vocab_size - 1, dtype=torch.long)
        recall_inputs = torch.zeros(cfg.seq_length, dtype=torch.long)

        inputs = torch.cat([seq, blank, delimiter, recall_inputs], dim=0)

        # Targets: during initial presentation and delay predict blank token,
        # after delimiter predict the stored sequence.
        blank_targets = torch.zeros(cfg.seq_length + cfg.delay + 1, dtype=torch.long)
        targets = torch.cat([blank_targets, seq], dim=0)
        return inputs, targets


def generate_copying_batch(
    dataset: CopyingTaskDataset,
    batch_indices: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs_batch = []
    targets_batch = []
    for idx in batch_indices:
        inputs, targets = dataset[idx]
        inputs_batch.append(inputs)
        targets_batch.append(targets)
    inputs_tensor = torch.stack(inputs_batch).to(device)
    targets_tensor = torch.stack(targets_batch).to(device)
    return inputs_tensor, targets_tensor


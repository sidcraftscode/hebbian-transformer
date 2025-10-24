from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class CueRewardConfig:
    num_pairs: int = 5
    cue_dim: int = 10
    dataset_size: int = 2000
    reward_range: Tuple[float, float] = (0.0, 1.0)

    @property
    def input_dim(self) -> int:
        # cue one-hot + reward channel + query flag
        return self.cue_dim + 2

    @property
    def sequence_length(self) -> int:
        return self.num_pairs * 2


class CueRewardDataset(Dataset):
    """
    Sequential cue-reward association task. Each trial shows num_pairs cue/reward
    presentations, followed by queries for each cue. The model must output the
    stored reward when queried.
    """

    def __init__(self, config: CueRewardConfig) -> None:
        self.config = config

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        cue_indices = torch.randperm(cfg.cue_dim)[: cfg.num_pairs]
        rewards = torch.empty(cfg.num_pairs).uniform_(*cfg.reward_range)

        total_steps = cfg.sequence_length
        inputs = torch.zeros(total_steps, cfg.input_dim)
        targets = torch.zeros(total_steps)

        for i, cue_idx in enumerate(cue_indices):
            # Presentation step
            step = i
            cue_one_hot = torch.zeros(cfg.cue_dim)
            cue_one_hot[cue_idx] = 1.0
            inputs[step, : cfg.cue_dim] = cue_one_hot
            inputs[step, cfg.cue_dim] = rewards[i]  # reward channel
            inputs[step, cfg.cue_dim + 1] = 0.0  # query flag
            targets[step] = 0.0

            # Query step
            step = cfg.num_pairs + i
            inputs[step, : cfg.cue_dim] = cue_one_hot
            # reward channel left at 0.0
            inputs[step, cfg.cue_dim + 1] = 1.0  # query flag
            targets[step] = rewards[i]

        return inputs, targets

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class FewShotRegressionConfig:
    input_dim: int = 3
    k_support: int = 10
    k_query: int = 10
    dataset_size: int = 2000
    weight_scale: float = 1.0
    bias_scale: float = 1.0

    @property
    def sequence_length(self) -> int:
        return self.k_support + self.k_query

    @property
    def model_input_dim(self) -> int:
        # x vector + observed y + query flag
        return self.input_dim + 2


class FewShotRegressionDataset(Dataset):
    """
    Generates few-shot regression tasks where the underlying mapping is a random
    linear function (with bias). The learner observes k_support pairs and must
    predict outputs for k_query unseen inputs.
    """

    def __init__(self, config: FewShotRegressionConfig) -> None:
        self.config = config

    def __len__(self) -> int:
        return self.config.dataset_size

    def sample_function(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        weight = torch.randn(cfg.input_dim) * cfg.weight_scale
        bias = torch.randn(1) * cfg.bias_scale
        return weight, bias

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        weight, bias = self.sample_function()

        support_x = torch.randn(cfg.k_support, cfg.input_dim)
        support_y = support_x @ weight + bias

        query_x = torch.randn(cfg.k_query, cfg.input_dim)
        query_y = query_x @ weight + bias

        total_steps = cfg.sequence_length
        inputs = torch.zeros(total_steps, cfg.model_input_dim)
        targets = torch.zeros(total_steps)

        for i in range(cfg.k_support):
            x_vec = support_x[i]
            y_val = support_y[i]
            inputs[i, : cfg.input_dim] = x_vec
            inputs[i, cfg.input_dim] = y_val
            inputs[i, cfg.input_dim + 1] = 0.0  # query flag
            targets[i] = 0.0

        for j in range(cfg.k_query):
            idx_step = cfg.k_support + j
            x_vec = query_x[j]
            y_val = query_y[j]
            inputs[idx_step, : cfg.input_dim] = x_vec
            inputs[idx_step, cfg.input_dim] = 0.0
            inputs[idx_step, cfg.input_dim + 1] = 1.0
            targets[idx_step] = y_val

        return inputs, targets


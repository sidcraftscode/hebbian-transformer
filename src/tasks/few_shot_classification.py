import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, Omniglot


@dataclass
class FewShotClassificationConfig:
    root: str = "./data"
    ways: int = 5
    shots: int = 1
    queries: int = 15
    split: str = "train"  # "train" or "test"
    dataset: str = "cifarfs"


class FewShotClassificationTask:
    """
    Samples few-shot classification episodes from configurable vision datasets.
    """

    def __init__(self, config: FewShotClassificationConfig) -> None:
        self.config = config
        dataset_name = config.dataset.lower()
        self.input_resolution: int
        if dataset_name == "cifarfs":
            train = config.split == "train"
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
                ]
            )
            self.dataset: Dataset = CIFAR100(
                root=config.root,
                train=train,
                download=True,
                transform=transform,
            )
            self.input_channels = 3
            self.input_resolution = 32
        elif dataset_name == "omniglot":
            background = config.split == "train"
            transform = T.Compose(
                [
                    T.Resize(84),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5]),
                ]
            )
            self.dataset = Omniglot(
                root=config.root,
                background=background,
                download=True,
                transform=transform,
            )
            self.input_channels = 1
            self.input_resolution = 84
        else:
            raise ValueError(f"Unsupported dataset '{config.dataset}'.")

        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, (_, target) in enumerate(self.dataset):
            self.class_to_indices.setdefault(target, []).append(idx)
        self.available_classes = list(self.class_to_indices.keys())

    def sample_episode(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        cfg = self.config
        selected_classes = random.sample(self.available_classes, cfg.ways)
        support_images: List[torch.Tensor] = []
        support_labels: List[int] = []
        query_images: List[torch.Tensor] = []
        query_labels: List[int] = []
        local_classes: List[int] = []

        for local_id, class_id in enumerate(selected_classes):
            indices = random.sample(self.class_to_indices[class_id], cfg.shots + cfg.queries)
            for idx in indices[: cfg.shots]:
                image, _ = self.dataset[idx]
                support_images.append(image)
                support_labels.append(local_id)
            for idx in indices[cfg.shots :]:
                image, _ = self.dataset[idx]
                query_images.append(image)
                query_labels.append(local_id)
            local_classes.append(class_id)

        support_tensor = torch.stack(support_images)  # (ways * shots, C, H, W)
        query_tensor = torch.stack(query_images)  # (ways * queries, C, H, W)
        support_targets = torch.tensor(support_labels, dtype=torch.long)
        query_targets = torch.tensor(query_labels, dtype=torch.long)
        return support_tensor, support_targets, query_tensor, query_targets, local_classes

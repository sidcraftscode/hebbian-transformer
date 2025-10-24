from __future__ import annotations

import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    summary = load_summary(Path("experiments/results/summary.json"))
    copying = summary["copying"]
    cue = summary["cue_reward"]
    classification = summary["classification_cifarfs"]
    regression = summary["few_shot_regression"]
    print("Copying gradient eta:", copying["gradient"]["aggregate"]["eta_mean_mean"])
    print("Copying hebbian eta:", copying["hebbian"]["aggregate"]["eta_mean_mean"])
    print("Cue gradient loss:", cue["gradient"]["aggregate"]["loss_mean"])
    print("Cue none loss:", cue["none"]["aggregate"]["loss_mean"])
    print("Reg Hebbian Query MSE:", regression["hebbian"]["aggregate"]["query_loss_mean"])
    print("Reg none Query MSE:", regression["none"]["aggregate"]["query_loss_mean"])
    print("CIFAR Hebbian acc:", classification["hebbian"]["aggregate"]["accuracy_mean"])


if __name__ == "__main__":
    main()

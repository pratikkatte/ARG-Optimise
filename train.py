"""YAML-driven training entry point.

Usage: ``python train.py --config config.yaml``
"""

import argparse

from training.evaluation import evaluate_generator
from training.config import (
    DEFAULT_LOG_Z_LR,
    DEFAULT_MU_PER_BP,
    DEFAULT_NE,
    MODEL_VERSION,
    TrainConfig,
)
from training.loop import seed_everything, train, train_epoch

__all__ = [
    "DEFAULT_LOG_Z_LR", "DEFAULT_MU_PER_BP", "DEFAULT_NE", "MODEL_VERSION",
    "TrainConfig", "evaluate_generator", "seed_everything", "train", "train_epoch",
]


def main():
    parser = argparse.ArgumentParser(description="Train the ARG GFlowNet from YAML configuration.")
    parser.add_argument("--config", required=True, help="Path to a training YAML file")
    args = parser.parse_args()
    train(TrainConfig.load(args.config))


if __name__ == "__main__":
    main()

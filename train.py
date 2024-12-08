import argparse

import wandb

from binary_diffusion_tabular.trainer import FixedSizeTableBinaryDiffusionTrainer
from binary_diffusion_tabular.utils import get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()
    config = get_config(args.config)

    comment = config["comment"]
    logger = wandb.init(project="binary_diffusion_tabular", name=comment, config=config)

    trainer = FixedSizeTableBinaryDiffusionTrainer.from_config(config, logger=logger)

    if config["fine_tune_from"]:
        trainer.load_checkpoint(config["fine_tune_from"])

    trainer.train()

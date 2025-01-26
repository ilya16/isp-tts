""" A minimal training script. """

import argparse
import os
from shutil import copyfile

from tts.experiments import Trainer
from tts.experiments.modules import ExperimentModules

if __name__ == "__main__":
    parser = argparse.ArgumentParser("training the model")
    parser.add_argument("--config-root", "-r", type=str, default="recipes")
    parser.add_argument("--config-name", "-n", type=str, default="acoustic/config.yaml")

    args = parser.parse_args()

    exp_modules = ExperimentModules(
        config=args.config_name,
        config_root=args.config_root
    )
    modules = exp_modules.init_modules()

    trainer = Trainer(
        **modules,
        config=exp_modules.config
    )

    # print(modules["model"])

    copyfile(os.path.join(args.config_root, args.config_name), os.path.join(trainer.config.output_dir, "config.yaml"))

    trainer.train()

""" A centralized initialization of Experiment Modules. """
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, fields

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from tts.data import DATASETS, COLLATORS
from tts.models import MODELS, EVALUATORS
from tts.modules.constructor import ModuleConfig
from tts.utils.config import disable_nodes, Config
from .trainer_config import TrainerConfig


@dataclass
class ExperimentConfig(Config):
    dataset: DictConfig
    collator: DictConfig
    model: DictConfig | ModuleConfig
    criterion: DictConfig | ModuleConfig | None
    trainer: DictConfig | TrainerConfig
    evaluator: DictConfig | None = None


ExperimentConfigFields = list(map(lambda f: f.name, fields(ExperimentConfig)))


def resolve_config_hierarchy(config: DictConfig, config_root: str | None = None):
    if "base" in config and config.base is not None:
        base_configs_names = [config.base] if isinstance(config.base, str) else config.base

        base_configs = []
        for base_config_name in base_configs_names:
            config_root = config_root or "./"
            base_config_path = os.path.join(config_root, base_config_name)
            base_config = OmegaConf.load(base_config_path)
            base_config = resolve_config_hierarchy(base_config, config_root=config_root)
            base_configs.append(base_config)

        del config["base"]

        config = OmegaConf.merge(*base_configs, config)

    return config


class ExperimentModules:
    def __init__(self, config: ExperimentConfig | str, config_root: str | None = None):
        if isinstance(config, str):
            config = os.path.join(config_root, config) if config_root is not None else config
            config = OmegaConf.load(config)
            if not isinstance(config, DictConfig):
                config = OmegaConf.load(config)

        config = resolve_config_hierarchy(config, config_root=config_root)

        assert all([key in config for key in ExperimentConfigFields[:3]]), \
            f"ExperimentConfig is missing one of the keys: {ExperimentConfigFields[:3]}"

        disable_nodes(config)
        OmegaConf.resolve(config)

        self.config = ExperimentConfig(
            dataset=config.dataset,
            collator=config.collator,
            model=config.model,
            criterion=config.criterion,
            trainer=config.trainer,
            evaluator=config.get("evaluator", None)
        )

        self.train_dataset = None
        self.eval_dataset = None
        self.collator = None
        self.model = None
        self.criterion = None
        self.evaluator = None

    def init_modules(self):
        self.init_datasets()
        self.init_collator()
        self.init_model()
        self.init_criterion()
        self.init_evaluator()

        return {
            "model": self.model,
            "criterion": self.criterion,
            "train_dataset": self.train_dataset,
            "eval_dataset": self.eval_dataset,
            "collator": self.collator,
            "evaluator": self.evaluator
        }

    def init_datasets(self):
        cfg = self.config.dataset
        self.train_dataset = build_dataset(cfg, split="train")
        self.eval_dataset = build_dataset(cfg, split="eval")

        return self.train_dataset, self.eval_dataset

    def init_collator(self):
        dataset = self.train_dataset or self.eval_dataset
        assert dataset is not None

        cfg = self.config.collator
        self.collator = build_collator(cfg)

        return self.collator

    def init_model(self, inject_data: bool = True):
        cfg = self.config.model

        dataset = None
        if inject_data:
            dataset = self.train_dataset or self.eval_dataset
            assert dataset is not None

        self.model = build_model(cfg, dataset=dataset)

        return self.model

    def init_criterion(self):
        cfg = self.config.criterion
        self.criterion = self.model.get_criterion(cfg) if cfg is not None else None

        return self.criterion

    def init_evaluator(self):
        assert self.model is not None

        cfg = self.config.evaluator
        if cfg is not None:
            self.evaluator = build_evaluator(cfg, model=self.model)

        return self.evaluator


def build_dataset(config, split: str = "train"):
    if config._name_ in DATASETS:
        config = copy.deepcopy(config)

        for key, value in config.get(f"_{split}_", {}).items():
            config[key] = value

        dataset_cls = DATASETS[config._name_]
        config = {key: value for key, value in config.items() if key[0] != "_"}
        dataset = dataset_cls(**config)
        return dataset
    else:
        raise ValueError(
            f"Invalid dataset type: {config._name_}. Supported types: {list(DATASETS.keys())}"
        )


def build_collator(config):
    if config._name_ in COLLATORS:
        config = copy.deepcopy(config)

        collator_cls = COLLATORS[config._name_]
        config = {key: value for key, value in config.items() if key[0] != "_"}
        collator = collator_cls(**config)
        return collator
    else:
        raise ValueError(
            f"Invalid data collator type: {config._name_}. Supported types: {list(COLLATORS.keys())}"
        )


def build_model(config, *, dataset: Dataset | None = None, **kwargs):
    if config._name_ in MODELS:
        model_cls = MODELS[config._name_]
        if dataset is not None:
            model_cls.inject_data_config(config, dataset)
        model = model_cls.init(config, **kwargs)
        model_cls.cleanup_config(config)
        return model
    else:
        raise ValueError(
            f"Invalid model type: {config._name_}. Supported types: {list(MODELS.keys())}"
        )


def build_evaluator(config, **kwargs):
    if config is not None and config._name_ in EVALUATORS:
        evaluator_cls = EVALUATORS[config._name_]
        config = {key: value for key, value in config.items() if key[0] != "_"}
        config.update(**kwargs)
        return evaluator_cls(**config)
    else:
        return None

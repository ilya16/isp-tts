from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def load_array(file_path: Path):
    if file_path.suffix == ".npy":
        mel = torch.from_numpy(np.load(str(file_path)))
    elif file_path.suffix in [".tensor"]:
        mel = torch.load(file_path, map_location="cpu", weights_only=True)
    else:
        raise RuntimeError

    return mel


def dynamic_range_compression(x, C: float = 1, clip_val: float = 1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C: float = 1):
    return torch.exp(x) / C


def remove_outliers(values):
    p25 = torch.quantile(values, 0.25)
    p75 = torch.quantile(values, 0.75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    return values[torch.logical_and(values > lower, values < upper)]


@dataclass
class FeatureStats:
    min: float = np.finfo(np.float32).max
    max: float = np.finfo(np.float32).min
    mean: float = 0.0
    std: float = 1.0

    def to_dict(self):
        return {
            "min": float(self.min),
            "max": float(self.max),
            "mean": float(self.mean),
            "std": float(self.std)
        }


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = 0

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.n_samples_seen_ = X.shape[0]
        return self

    def partial_fit(self, X):
        if self.mean_ is None:
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.n_samples_seen_ = X.shape[0]
        else:
            new_mean = np.mean(X, axis=0)
            new_variance = np.var(X, axis=0)
            new_count = X.shape[0]

            old_mean = self.mean_
            old_variance = self.scale_ ** 2
            old_count = self.n_samples_seen_

            self.mean_ = (old_mean * old_count + new_mean * new_count) / (old_count + new_count)
            self.scale_ = np.sqrt(
                ((old_count * (old_variance + old_mean ** 2)) + (new_count * (new_variance + new_mean ** 2)))
                / (old_count + new_count) - self.mean_ ** 2
            )
            self.n_samples_seen_ += new_count

        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

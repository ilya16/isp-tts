""" A general purpose Config with IO. That's it. """

import json
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum

from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass
class Config:
    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            elif isinstance(v, (DictConfig, ListConfig)):
                d[k] = OmegaConf.to_container(v)
            elif isinstance(v, Config):
                d[k] = v.to_dict()
        return d

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json_string(cls, json_string):
        return cls(**json.loads(json_string))

    def __contains__(self, item):
        return item in self.to_dict()


def disable_nodes(config, parent=None):
    if isinstance(config, DictConfig):
        eject = config.pop("_disable_", False)

        if eject and parent is not None:
            key = config._key()
            parent.pop(key)
        else:
            for value in list(config.values()):
                disable_nodes(value, config)


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


def asdict(data):
    if is_dataclass(data):
        return {k: asdict(v) for k, v in vars(data).items()}
    return data

"""Odin VAE: encoder-decoder over sets of inventor-name surfaces."""

from .config_classes import (
    ConfigForAugmentation,
    ConfigForDataLoader,
    ConfigForDataModule,
    ConfigForDatasetSplit,
    ConfigForHarness,
    ConfigForModel,
    ConfigForRoot,
    ConfigForWebDataset,
)
from .configid import configid, configid_of_checkpoint, verify_configid
from .model import OdinModel

__all__ = [
    "ConfigForAugmentation",
    "ConfigForDataLoader",
    "ConfigForDataModule",
    "ConfigForDatasetSplit",
    "ConfigForHarness",
    "ConfigForModel",
    "ConfigForRoot",
    "ConfigForWebDataset",
    "OdinModel",
    "configid",
    "configid_of_checkpoint",
    "verify_configid",
]

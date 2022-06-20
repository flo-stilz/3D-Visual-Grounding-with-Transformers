# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset#, ScannetDatasetConfig
from model_util_scannet import ScannetDatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig]
}


def build_dataset(name):
    dataset_config = DATASET_FUNCTIONS[name][1]()
    
    return dataset_config
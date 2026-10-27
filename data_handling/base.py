from typing import Tuple

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

"""A batch is a 3-tuple of imgs, labels, and metadata dict"""
BatchType = Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]

class BaseDataModuleClass(LightningDataModule):
    def __init__(
        self, 
        args: DictConfig, 
 
    ) -> None:
        super().__init__()
        self.args = args
        self.create_datasets()

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.args.batch_size,
            shuffle=self.args.shuffle,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

    @property
    def dataset_name(self):
        raise NotImplementedError

    def create_datasets(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

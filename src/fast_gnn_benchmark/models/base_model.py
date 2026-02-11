from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import lightning as L
import torch
from torch_geometric.data import Data

from fast_gnn_benchmark.metrics.base_metrics import MetricsCollection
from fast_gnn_benchmark.schemas.model import BaseModelParameters

T = TypeVar("T", bound=BaseModelParameters)


class BaseGNN(L.LightningModule, Generic[T], ABC):
    def __init__(self, model_parameters: T):
        super().__init__()
        self.model_parameters = model_parameters
        self.model = self.load_model()
        self.loss = self.model_parameters.loss.get()
        self.train_metrics = self.load_metrics(prefix="train/")
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        self.save_hyperparameters(logger=False)

    def load_metrics(self, prefix: str) -> MetricsCollection:
        metrics = {metric.display_name: metric.get() for metric in self.model_parameters.metrics}

        return MetricsCollection(metrics, prefix=prefix)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.model_parameters.optimizer.get(self.model.parameters())

    @abstractmethod
    def load_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        pass

    def on_train_epoch_start(self):
        self.train_metrics.reset()

    def on_validation_epoch_start(self):
        self.val_metrics.reset()

    def on_test_epoch_start(self):
        self.test_metrics.reset()

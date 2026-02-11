import torch
from torch_geometric.data import Data

from fast_gnn_benchmark.models.backbones import load_backbone
from fast_gnn_benchmark.models.base_model import BaseGNN
from fast_gnn_benchmark.schemas.model import NodeClassificationModelParameters


class NodeClassifier(torch.nn.Module):
    def __init__(self, model_parameters: NodeClassificationModelParameters):
        super().__init__()
        self.model_parameters = model_parameters

        self.backbone = load_backbone(model_parameters.architecture_parameters)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x, edge_index)
        return x


class NodeClassificationModel(BaseGNN[NodeClassificationModelParameters]):
    def __init__(self, model_parameters: NodeClassificationModelParameters):
        super().__init__(model_parameters)

    def load_model(self) -> torch.nn.Module:
        return NodeClassifier(self.model_parameters)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        y_pred = self.model(batch.x, batch.edge_index)
        y_true: torch.Tensor = batch.y  # type: ignore

        loss_all = self.loss(y_pred, y_true)
        mask = batch.compute_mask.float()

        mask_sum = mask.sum().clamp(min=1)
        loss = (loss_all * mask).sum() / mask_sum

        batch_metrics = self.train_metrics(y_pred, y_true, batch.compute_mask)
        self.log_dict(
            {"train/loss": loss, **batch_metrics}, on_step=True, on_epoch=True, batch_size=mask_sum.int(), prog_bar=True
        )

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        y_pred = self.model(batch.x, batch.edge_index)
        y_true: torch.Tensor = batch.y  # type: ignore

        loss_all = self.loss(y_pred, y_true)
        mask = batch.compute_mask.float()

        mask_sum = mask.sum().clamp(min=1)
        loss = (loss_all * mask).sum() / mask_sum

        batch_metrics = self.val_metrics(y_pred, y_true, batch.compute_mask)

        self.log_dict({"val/loss": loss, **batch_metrics}, on_epoch=True, batch_size=mask_sum.int(), prog_bar=True)

        return loss

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        y_pred = self.model(batch.x, batch.edge_index)
        y_true: torch.Tensor = batch.y  # type: ignore

        loss_all = self.loss(y_pred, y_true)
        mask = batch.compute_mask.float()

        mask_sum = mask.sum().clamp(min=1)
        loss = (loss_all * mask).sum() / mask_sum

        batch_metrics = self.test_metrics(y_pred, y_true, batch.compute_mask)

        self.log_dict({"test/loss": loss, **batch_metrics}, on_epoch=True, batch_size=mask_sum.int(), prog_bar=True)

        return loss

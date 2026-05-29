import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from fast_gnn_benchmark.models.backbones import load_backbone
from fast_gnn_benchmark.models.base_model import BaseGNN
from fast_gnn_benchmark.schemas.model import SEALModelParameters


class SealGNN(torch.nn.Module):
    def __init__(self, params: SEALModelParameters):
        super().__init__()
        self.params = params
        self.backbone = load_backbone(params.architecture_parameters)

        output_dim = params.architecture_parameters.output_dim
        layers: list[torch.nn.Module] = []
        for _ in range(params.num_classifier_layers - 1):
            layers += [torch.nn.Linear(output_dim, output_dim), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(output_dim, 1))
        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, batch: Batch) -> torch.Tensor:
        z = batch.z.clamp(max=self.params.max_z - 1)
        z_onehot = F.one_hot(z, num_classes=self.params.max_z).float()

        if batch.x is not None:
            x = torch.cat([batch.x, z_onehot], dim=-1)
        else:
            x = z_onehot

        x = self.backbone(x, batch.edge_index)

        if self.params.pooling_type == "sum":
            x = global_add_pool(x, batch.batch)
        else:
            x = global_mean_pool(x, batch.batch)

        return self.classifier(x).squeeze(-1)


class SealModel(BaseGNN[SEALModelParameters]):
    def __init__(self, model_parameters: SEALModelParameters):
        super().__init__(model_parameters)

    def load_model(self) -> torch.nn.Module:
        return SealGNN(self.model_parameters)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred = self.model(batch)
        loss = self.loss(pred, batch.y)
        batch_metrics = self.train_metrics(pred, batch.y)
        self.log_dict(
            {"train/loss": loss, **batch_metrics},
            on_step=True,
            on_epoch=True,
            batch_size=batch.y.shape[0],
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred = self.model(batch)
        loss = self.loss(pred, batch.y)
        self.val_metrics(pred, batch.y)  # accumulate only — logged via on_validation_epoch_end
        self.log("val/loss", loss, on_epoch=True, batch_size=batch.y.shape[0], prog_bar=False)
        return loss

    def on_validation_epoch_end(self) -> None:
        # Use compute() on the accumulated scores so hit@k is the *global*
        # metric over all accumulated pos/neg, not an average of per-batch values.
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=False)

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred = self.model(batch)
        loss = self.loss(pred, batch.y)
        self.test_metrics(pred, batch.y)  # accumulate only — logged via on_test_epoch_end
        self.log("test/loss", loss, on_epoch=True, batch_size=batch.y.shape[0], prog_bar=False)
        return loss

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, prog_bar=False)

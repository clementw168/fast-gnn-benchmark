import torch
from torch_geometric.data import Data

from fast_gnn_benchmark.models.backbones import load_backbone
from fast_gnn_benchmark.models.base_model import BaseGNN
from fast_gnn_benchmark.schemas.model import (
    ArchitectureParametersChoices,
    LinkPredictionModelParameters,
    LinkPredictorParameters,
    LinkPredictorType,
)


class CosineSimilarityClassifier(torch.nn.Module):
    def forward(
        self, embedding_1: torch.Tensor, embedding_2: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        return (embedding_1[edge_label_index[0]] * embedding_2[edge_label_index[1]]).sum(dim=-1)


class LinkPredictorBase(torch.nn.Module):
    def __init__(
        self, architecture_parameters: ArchitectureParametersChoices, link_predictor_parameters: LinkPredictorParameters
    ):
        super().__init__()
        self.architecture_parameters = architecture_parameters
        self.link_predictor_parameters = link_predictor_parameters

        self.backbone = load_backbone(architecture_parameters)
        self.classifier = self.load_classifier()

    def load_classifier(self) -> torch.nn.Module:
        match self.link_predictor_parameters.link_predictor_type:
            case LinkPredictorType.COSINE_SIMILARITY:
                return CosineSimilarityClassifier()
            case _:
                raise ValueError(f"Invalid classifier type: {self.link_predictor_parameters.link_predictor_type}")

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edges = data.edge_index

        x = self.backbone(x, edges)
        return self.classifier(x, x, edges)


class LinkPredictionModel(BaseGNN[LinkPredictionModelParameters]):
    def __init__(self, model_parameters: LinkPredictionModelParameters):
        super().__init__(model_parameters)

    def load_model(self) -> torch.nn.Module:
        return LinkPredictorBase(
            self.model_parameters.architecture_parameters, self.model_parameters.link_predictor_parameters
        )

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred = self.model(batch)
        loss = self.loss(pred, batch.edge_label)
        batch_metrics = self.train_metrics(pred, batch.edge_label)
        self.log_dict(
            {"train/loss": loss, **batch_metrics},
            on_step=True,
            on_epoch=True,
            batch_size=batch.edge_label.shape[0],
            prog_bar=True,
        )  # type: ignore

        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred = self.model(batch)
        loss = self.loss(pred, batch.edge_label)
        batch_metrics = self.val_metrics(pred, batch.edge_label)
        self.log_dict(
            {"val/loss": loss, **batch_metrics}, on_epoch=True, batch_size=batch.edge_label.shape[0], prog_bar=True
        )  # type: ignore

        return loss

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        pred = self.model(batch)
        loss = self.loss(pred, batch.edge_label)
        batch_metrics = self.test_metrics(pred, batch.edge_label)
        self.log_dict(
            {"test/loss": loss, **batch_metrics}, on_epoch=True, batch_size=batch.edge_label.shape[0], prog_bar=True
        )  # type: ignore

        return loss

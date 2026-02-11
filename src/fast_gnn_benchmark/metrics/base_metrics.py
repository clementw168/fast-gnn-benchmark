import copy
from abc import ABC, abstractmethod

import torch
import torchmetrics


class BinaryDistribution(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("prediction_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        with torch.no_grad():
            self.prediction_class += (pred > 0).sum()
            self.total_samples += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.prediction_class.float() / self.total_samples.float()  # type: ignore


# -------------------- Compilation-friendly Metrics for Masked data  --------------------


class OptimizedMetric(torch.nn.Module, ABC):
    @abstractmethod
    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute(self) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.update(*args, **kwargs)


class MetricsCollection(torch.nn.Module):
    def __init__(self, metrics: dict[str, OptimizedMetric | torchmetrics.Metric], prefix: str):
        super().__init__()
        self.metrics = torch.nn.ModuleDict(metrics)
        self.prefix = prefix

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}
        for name, metric in self.metrics.items():
            results[f"{self.prefix}{name}"] = metric(*args, **kwargs)

        return results

    def compute(self) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}
        for name, metric in self.metrics.items():
            results[f"{self.prefix}{name}"] = metric.compute()  # type: ignore
        return results

    def reset(self) -> None:
        for metric in self.metrics.values():
            if hasattr(metric, "reset"):
                metric.reset()  # type: ignore

    def clone(self, prefix: str) -> "MetricsCollection":
        new_metrics: dict[str, OptimizedMetric | torchmetrics.Metric] = {
            name: copy.deepcopy(metric)
            for name, metric in self.metrics.items()  # type: ignore
        }
        return MetricsCollection(new_metrics, prefix=prefix)

    def add_metrics(self, metrics: dict[str, OptimizedMetric | torchmetrics.Metric]) -> None:
        self.metrics.update(metrics)


class OptimizedMultiClassAccuracy(OptimizedMetric):
    def __init__(self):
        super().__init__()
        self.register_buffer("correct_predictions", torch.tensor(0.0))
        self.register_buffer("total_samples", torch.tensor(0.0))

        self.reset()

    @staticmethod
    def get_accuracy(correct_predictions: torch.Tensor, total_samples: torch.Tensor) -> torch.Tensor:
        return correct_predictions / total_samples.clamp(min=1)

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = pred.argmax(dim=1)
            batch_correct_predictions = ((pred == target) * mask).sum()
            batch_total_samples = mask.sum().clamp(min=1)

            self.correct_predictions += batch_correct_predictions
            self.total_samples += batch_total_samples

            return self.get_accuracy(batch_correct_predictions, batch_total_samples)

    def compute(self) -> torch.Tensor:
        return self.get_accuracy(self.correct_predictions, self.total_samples)

    def reset(self) -> None:
        self.correct_predictions.zero_()  # type: ignore
        self.total_samples.zero_()  # type: ignore


class OptimizedStatScores(OptimizedMetric):
    def __init__(self):
        super().__init__()
        self.register_buffer("tp", torch.tensor(0.0))
        self.register_buffer("fp", torch.tensor(0.0))
        self.register_buffer("tn", torch.tensor(0.0))
        self.register_buffer("fn", torch.tensor(0.0))
        self.reset()

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pred = pred.argmax(dim=1)
            batch_tp = ((pred == target) * mask).sum()
            batch_fp = ((pred != target) * mask).sum()
            batch_tn = ((pred == target) * mask).sum()
            batch_fn = ((pred != target) * mask).sum()

            self.tp += batch_tp
            self.fp += batch_fp
            self.tn += batch_tn
            self.fn += batch_fn

        return batch_tp, batch_fp, batch_tn, batch_fn

    def reset(self) -> None:
        self.tp.zero_()  # type: ignore
        self.fp.zero_()  # type: ignore
        self.tn.zero_()  # type: ignore
        self.fn.zero_()  # type: ignore


class OptimizedF1Score(OptimizedStatScores):
    @staticmethod
    def get_f1_score(batch_tp: torch.Tensor, batch_fp: torch.Tensor, batch_fn: torch.Tensor) -> torch.Tensor:
        precision = batch_tp / (batch_tp + batch_fp)
        recall = batch_tp / (batch_tp + batch_fn)
        return 2 * (precision * recall) / (precision + recall)

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_tp, batch_fp, _, batch_fn = super().update(pred, target, mask)
        return self.get_f1_score(batch_tp, batch_fp, batch_fn)

    def compute(self) -> torch.Tensor:
        return self.get_f1_score(self.tp, self.fp, self.fn)  # type: ignore


class OptimizedPrecision(OptimizedStatScores):
    @staticmethod
    def get_precision(batch_tp: torch.Tensor, batch_fp: torch.Tensor) -> torch.Tensor:
        return batch_tp / (batch_tp + batch_fp)

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_tp, batch_fp, _, _ = super().update(pred, target, mask)
        return self.get_precision(batch_tp, batch_fp)

    def compute(self) -> torch.Tensor:
        return self.get_precision(self.tp, self.fp)


class OptimizedRecall(OptimizedStatScores):
    @staticmethod
    def get_recall(batch_tp: torch.Tensor, batch_fn: torch.Tensor) -> torch.Tensor:
        return batch_tp / (batch_tp + batch_fn)

    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_tp, _, _, batch_fn = super().update(pred, target, mask)
        return self.get_recall(batch_tp, batch_fn)

    def compute(self) -> torch.Tensor:
        return self.get_recall(self.tp, self.fn)

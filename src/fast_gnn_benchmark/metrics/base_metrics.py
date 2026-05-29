import copy
from abc import ABC, abstractmethod

import torch
import torchmetrics


class BinaryDistribution(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("prediction_class", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:  # noqa: ARG002
        with torch.no_grad():
            self.prediction_class += (pred > 0).sum()
            self.total_samples += pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.prediction_class.float() / self.total_samples.float()  # type: ignore


# -------------------- Compilation-friendly Metrics for Masked data  --------------------


class OptimizedMetric(torch.nn.Module, ABC):
    @abstractmethod
    def update(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
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

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
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


# -------------------- Link prediction metrics --------------------


class HitRate(OptimizedMetric):
    def __init__(self, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k

        # persistent=False so these accumulating buffers are not saved in checkpoints
        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        pred: logits/scores for each sample
        target: tensor containing 0.0 or 1.0 labels (same first-dim length as pred)
        """
        with torch.no_grad():
            # target could be float/bool/int; interpret >0.5 as positive
            pos_mask = target > 0.5  # noqa: PLR2004
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask]
            neg_batch = pred[neg_mask]

            # Append to buffers
            if pos_batch.numel() > 0:
                self.pos_scores = torch.cat([self.pos_scores, pos_batch.detach()])
            if neg_batch.numel() > 0:
                self.neg_scores = torch.cat([self.neg_scores, neg_batch.detach()])

            # Return current hit rate computed on this batch
            return self._hits_at_k(pos_batch, neg_batch, self.k)

    @staticmethod
    def _hits_at_k(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor, k: int) -> torch.Tensor:
        """
        Vectorized version of your eval_hits, returning a scalar tensor.
        """
        # If no positive samples, define as 0 to avoid NaNs (you can choose another convention)
        if y_pred_pos.numel() == 0:
            return torch.tensor(0.0, device=y_pred_neg.device if y_pred_neg.is_cuda else y_pred_pos.device)

        # If not enough negatives, hits@k is 1.0 per your reference function
        if y_pred_neg.numel() < k:
            return torch.tensor(1.0, device=y_pred_pos.device)

        kth_score_in_neg = torch.topk(y_pred_neg, k).values[-1]
        return (y_pred_pos > kth_score_in_neg).float().mean()

    def compute(self) -> torch.Tensor:
        with torch.no_grad():
            return self._hits_at_k(self.pos_scores, self.neg_scores, self.k)

    def reset(self) -> None:
        # "Empty" the buffers while keeping device/dtype consistent
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))


class MRR(OptimizedMetric):
    """
    Accumulates positive and negative scores across updates, then computes
    mean reciprocal rank using the same logic as eval_mrr.
    """

    def __init__(self) -> None:
        super().__init__()
        # persistent=False so these accumulating buffers are not saved in checkpoints
        self.register_buffer("pos_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer("neg_scores", torch.empty(0, dtype=torch.float32), persistent=False)
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        pred: scores/logits for each sample (1D tensor)
        target: tensor containing 0.0 or 1.0 labels (same length as pred)
        """
        with torch.no_grad():
            pos_mask = target > 0.5  # noqa: PLR2004
            neg_mask = ~pos_mask

            pos_batch = pred[pos_mask].detach()
            neg_batch = pred[neg_mask].detach()

            if pos_batch.numel() > 0:
                self.pos_scores = torch.cat([self.pos_scores, pos_batch])
            if neg_batch.numel() > 0:
                self.neg_scores = torch.cat([self.neg_scores, neg_batch])

            # Return batch MRR (computed on this batch)
            return self._mrr(pos_batch, neg_batch)

    @staticmethod
    def _mrr(y_pred_pos: torch.Tensor, y_pred_neg: torch.Tensor) -> torch.Tensor:
        """
        Port of eval_mrr, but returns a scalar tensor.
        Assumes y_pred_pos are the scores for positive edges and y_pred_neg for negatives.
        """
        # Match "no positives" edge case handling similar to HitRate (avoid NaNs)
        if y_pred_pos.numel() == 0:
            device = y_pred_neg.device if y_pred_neg.is_cuda else y_pred_pos.device
            return torch.tensor(0.0, device=device)

        # eval_mrr assumes y_pred_neg is 2D (num_pos, num_neg) so it can rank per positive.
        # With the same buffering scheme as HitRate (a flat list of negatives),
        # we treat *all* accumulated negatives as the comparison set for each positive.
        y_pred_pos_2d = y_pred_pos.view(-1, 1)  # [P, 1]
        y_pred_neg_2d = y_pred_neg.view(1, -1)  # [1, N] -> broadcast to [P, N]

        print(y_pred_neg_2d.shape, y_pred_pos_2d.shape)

        optimistic_rank = (y_pred_neg_2d > y_pred_pos_2d).sum(dim=1)
        pessimistic_rank = (y_pred_neg_2d >= y_pred_pos_2d).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1.0
        mrr_list = 1.0 / ranking_list.to(torch.float32)

        return mrr_list.mean()

    def compute(self) -> torch.Tensor:
        with torch.no_grad():
            return self._mrr(self.pos_scores, self.neg_scores)

    def reset(self) -> None:
        self.pos_scores = self.pos_scores.new_empty((0,))
        self.neg_scores = self.neg_scores.new_empty((0,))


class BinaryAccuracy(OptimizedMetric):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("correct_predictions", torch.tensor(0.0))
        self.register_buffer("total_samples", torch.tensor(0.0))
        self.reset()

    @staticmethod
    def get_accuracy(correct_predictions: torch.Tensor, total_samples: torch.Tensor) -> torch.Tensor:
        return correct_predictions / total_samples.clamp(min=1)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        with torch.no_grad():
            # pred is a raw logit (model uses bce_with_logits_loss), so the
            # decision boundary is 0, not 0.5.
            correct_predictions = (pred > 0) == target
            self.correct_predictions += correct_predictions.sum()
            self.total_samples += pred.shape[0]

            return self.get_accuracy(self.correct_predictions, self.total_samples)  # type: ignore

    def compute(self) -> torch.Tensor:
        return self.get_accuracy(self.correct_predictions, self.total_samples)

    def reset(self) -> None:
        self.correct_predictions.zero_()  # type: ignore
        self.total_samples.zero_()  # type: ignore

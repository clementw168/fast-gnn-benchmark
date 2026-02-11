import lightning as L
import torch

from fast_gnn_benchmark.data.dataloaders import OptimizedRandomNodeLoader


def average_per_params(list_of_grads: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: sum(grad[key] for grad in list_of_grads) / len(list_of_grads) for key in list_of_grads[0].keys()}  # type: ignore


def norm_per_params(grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.linalg.vector_norm(value, ord=2) for key, value in grads.items()}


def norm_over_params(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(value.pow(2).sum() for value in grads.values()))  # type: ignore


def difference_per_params(grads1: dict[str, torch.Tensor], grads2: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.abs(grads1[key] - grads2[key]) for key in grads1.keys()}


def dot_product_per_params(grads1: dict[str, torch.Tensor], grads2: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(torch.dot(grads1[key].flatten(), grads2[key].flatten()) for key in grads1.keys())  # type: ignore


def get_batch_grads(
    loader: OptimizedRandomNodeLoader,
    wrapper_model: L.LightningModule,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]], float]:
    batch_grads = []
    losses = []
    samples = 0
    correct_samples = 0
    for batch in loader:
        y_pred: torch.Tensor = wrapper_model.model(batch.x, batch.edge_index)  # type: ignore
        y_true: torch.Tensor = batch.y  # type: ignore

        samples += y_true.shape[0]
        correct_samples += (y_pred.argmax(dim=1) == y_true).sum().cpu().item()

        loss_all: torch.Tensor = wrapper_model.loss(y_pred, y_true)  # type: ignore
        mask = batch.compute_mask.float()

        mask_sum = mask.sum().clamp(min=1)
        loss = (loss_all * mask).sum() / mask_sum

        optimizer.zero_grad()

        loss.backward()

        batch_grad = {
            name: (p.grad.detach().clone() if p.grad is not None else None)
            for name, p in wrapper_model.model.named_parameters()  # type: ignore
        }

        batch_grads.append(batch_grad)
        losses.append(loss.detach().clone())
    return sum(losses) / len(losses), batch_grads, correct_samples / max(1, samples)  # type: ignore


def get_loss_terms(
    full_grads: list[dict[str, torch.Tensor]], parts_grads: list[dict[str, torch.Tensor]], learning_rate: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_average = average_per_params(parts_grads)
    full_grad = average_per_params(full_grads)
    gd_term = learning_rate * norm_over_params(full_grad) ** 2 / 4

    dot_term = learning_rate * dot_product_per_params(full_grad, batch_average) / 2

    bias_term = learning_rate * norm_over_params(difference_per_params(full_grad, batch_average)) ** 2 / 4

    diff = [
        norm_over_params(difference_per_params(batch_average, parts_grads[i])) ** 2 for i in range(len(parts_grads))
    ]
    variance_term: torch.Tensor = sum(diff) / len(diff) * learning_rate / 4  # type: ignore

    full_regularization_term: torch.Tensor = (
        sum(norm_over_params(part_grad) ** 2 for part_grad in parts_grads) / len(parts_grads) * learning_rate / 4
    )  # type: ignore

    return gd_term, dot_term, bias_term, variance_term, full_regularization_term


class ImplicitGradientCorrectionCallback(L.Callback):
    def __init__(self, metrics_num_parts: int | None = None):
        super().__init__()
        self.dataloader = None
        self.metrics_num_parts = metrics_num_parts

    def set_dataloader(self, dataloader: OptimizedRandomNodeLoader) -> None:
        if not isinstance(dataloader, OptimizedRandomNodeLoader):
            raise ValueError("Dataloader must be an instance of OptimizedRandomNodeLoader")
        self.dataloader = dataloader
        self.dataloader_num_parts = dataloader.num_parts
        if self.metrics_num_parts is None:
            self.metrics_num_parts = self.dataloader_num_parts

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.dataloader is None or self.metrics_num_parts is None:
            raise ValueError("Dataloader not set or metrics_num_parts not set")

        was_training = pl_module.training
        pl_module.eval()  # Remove dropout and other training specific operations

        optimizer = trainer.optimizers[0]
        learning_rate = optimizer.param_groups[0]["lr"]

        self.dataloader.num_parts = 1
        full_loss, full_grads, full_accuracy = get_batch_grads(self.dataloader, pl_module, optimizer)

        self.dataloader.num_parts = self.metrics_num_parts
        parts_loss, parts_grads, parts_accuracy = get_batch_grads(self.dataloader, pl_module, optimizer)

        self.dataloader.num_parts = self.dataloader_num_parts

        gd_term, dot_term, bias_term, variance_term, full_regularization_term = get_loss_terms(
            full_grads, parts_grads, learning_rate
        )

        metrics = {
            "igc/full_accuracy": full_accuracy,
            "igc/part_accuracy": parts_accuracy,
            "igc/full_loss": full_loss.cpu().item(),
            "igc/parts_loss": parts_loss.cpu().item(),
            "igc/gd_term": gd_term.cpu().item(),
            "igc/dot_term": dot_term.cpu().item(),
            "igc/bias_term": bias_term.cpu().item(),
            "igc/variance_term": variance_term.cpu().item(),
            "igc/full_regularization_term": full_regularization_term.cpu().item(),
            "epoch": int(trainer.current_epoch),
        }

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics)

        optimizer.zero_grad()
        if was_training:
            pl_module.train()

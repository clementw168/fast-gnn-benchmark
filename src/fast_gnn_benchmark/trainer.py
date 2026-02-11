import random
from typing import Any

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized

from fast_gnn_benchmark.models.link_prediction import LinkPredictionModel
from fast_gnn_benchmark.models.node_classification import NodeClassificationModel
from fast_gnn_benchmark.schemas.data_models import DataLoaderTypeChoices
from fast_gnn_benchmark.schemas.model import (
    LinkPredictionModelParameters,
    NodeClassificationModelParameters,
    TrainerParameters,
)


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def override_nested_dict(nested_dict: dict[str, Any], override_dict: dict[str, Any]) -> dict[str, Any]:
    """Override a nested dictionary with another nested dictionary

    Args:
        nested_dict: The nested dictionary to override
        override_dict: The nested dictionary to override with

    Returns:
        The overridden nested dictionary

    Example:
        >>> nested_dict = {"a": {"b": {"c": 1, "d": 2}}}
        >>> override_dict = {"a": {"b": {"c": 3}}}
        >>> override_nested_dict(nested_dict, override_dict)
        {"a": {"b": {"c": 3, "d": 2}}}
    """
    for key, value in override_dict.items():
        if key in nested_dict:
            if isinstance(override_dict[key], dict):
                nested_dict[key] = override_nested_dict(nested_dict[key], value)
            else:
                nested_dict[key] = value

        else:
            nested_dict[key] = value

    return nested_dict


def get_global_config() -> dict[str, Any]:
    with open("global_config.yml", "r") as file:
        return yaml.safe_load(file)


def get_trainer_parameters_from_config(
    config_file: str, override_dict: dict | None = None, import_global_config: bool = True
) -> TrainerParameters:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if import_global_config:
        config = override_nested_dict(config, get_global_config())

    if override_dict is not None:
        config = override_nested_dict(config, override_dict)

    return TrainerParameters.model_validate(config)


def _configure_and_compile_model(
    model: NodeClassificationModel | LinkPredictionModel,
    use_compiled_torch: bool,
    full_graph: bool,
) -> NodeClassificationModel | LinkPredictionModel:
    """Configure PyTorch Dynamo and compile the model if CUDA is available."""
    if use_compiled_torch:
        if torch.cuda.is_available():
            # Configure dynamo to allow dynamic parameter shapes to reduce recompilations
            # This helps with models that have varying input sizes or dynamic architectures
            torch._dynamo.config.force_parameter_static_shapes = False
            model = torch.compile(model, fullgraph=full_graph)  # type: ignore
        else:
            print("No CUDA available, skipping compilation")
    return model  # type: ignore


def get_model(trainer_parameters: TrainerParameters) -> NodeClassificationModel | LinkPredictionModel:
    match trainer_parameters.model_parameters.task_type:
        case "node_classification":
            assert isinstance(trainer_parameters.model_parameters, NodeClassificationModelParameters)
            model = NodeClassificationModel(trainer_parameters.model_parameters)
        case "link_prediction":
            assert isinstance(trainer_parameters.model_parameters, LinkPredictionModelParameters)
            model = LinkPredictionModel(trainer_parameters.model_parameters)

        case _:
            raise ValueError(f"Invalid task type: {trainer_parameters.model_parameters.task_type}")

    return _configure_and_compile_model(
        model,
        trainer_parameters.compilation_parameters.use_compiled_torch,
        trainer_parameters.compilation_parameters.full_graph,
    )


def load_model_from_checkpoint(checkpoint_path: str) -> NodeClassificationModel | LinkPredictionModel:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model_parameters = checkpoint["hyper_parameters"]["model_parameters"]
    match model_parameters.task_type:
        case "node_classification":
            return NodeClassificationModel.load_from_checkpoint(checkpoint_path, weights_only=False)
        case "link_prediction":
            return LinkPredictionModel.load_from_checkpoint(checkpoint_path, weights_only=False)

        case _:
            raise ValueError(f"Invalid task type: {model_parameters.task_type}")


def get_model_to_test(
    callbacks: list[L.Callback],
    last_model: NodeClassificationModel | LinkPredictionModel,
    trainer_parameters: TrainerParameters,
) -> NodeClassificationModel | LinkPredictionModel:
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            best_model_path = callback.best_model_path
            print("Loaded best model from checkpoint")

            best_model = load_model_from_checkpoint(best_model_path)

            return _configure_and_compile_model(
                best_model,
                trainer_parameters.compilation_parameters.use_compiled_torch,
                trainer_parameters.compilation_parameters.full_graph,
            )

    print("No ModelCheckpoint callback found, testing on last model")
    return last_model


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
        print("Tensor Core optimization enabled (medium precision)")

        return "cuda"

    return "cpu"


def get_wandb_logger(trainer_parameters: TrainerParameters) -> WandbLogger | None:
    if trainer_parameters.wandb_logger_parameters is None:
        return None

    if trainer_parameters.wandb_logger_parameters.reinit:
        trainer_parameters.wandb_logger_parameters.name = None
        trainer_parameters.wandb_logger_parameters.id = None

    wandb_logger = trainer_parameters.wandb_logger_parameters.get()

    # Upload the full config to wandb
    wandb_logger.experiment.config.update(trainer_parameters.model_dump())

    # Add data to summary for easier visualization
    if hasattr(trainer_parameters.data_parameters.train_data_loader_parameters, "num_parts"):
        wandb_logger.experiment.summary["train/num_parts"] = (
            trainer_parameters.data_parameters.train_data_loader_parameters.num_parts  # type: ignore
        )
    if hasattr(trainer_parameters.data_parameters.val_data_loader_parameters, "num_parts"):
        wandb_logger.experiment.summary["val/num_parts"] = (
            trainer_parameters.data_parameters.val_data_loader_parameters.num_parts  # type: ignore
        )
    if hasattr(trainer_parameters.data_parameters.test_data_loader_parameters, "num_parts"):
        wandb_logger.experiment.summary["test/num_parts"] = (
            trainer_parameters.data_parameters.test_data_loader_parameters.num_parts  # type: ignore
        )

    return wandb_logger


def get_callbacks(trainer_parameters: TrainerParameters) -> list[L.Callback]:
    return [callback.get() for callback in trainer_parameters.callbacks]


def check_test_batch(
    model: NodeClassificationModel | LinkPredictionModel,
    test_loader: DataLoaderTypeChoices,
    device: str,
) -> None:
    model = _maybe_unwrap_optimized(model)  # type: ignore

    with torch.no_grad():
        model.to(device)
        model.eval()
        for batch in test_loader:
            batch.to(device)
            model.test_step(batch, 0)  # type: ignore
            break

    model.train()
    print("Test batch passed")


def do_run(trainer_parameters: TrainerParameters) -> list[dict[str, float]]:
    wandb_logger = get_wandb_logger(trainer_parameters)

    train_loader, val_loader, test_loader = trainer_parameters.data_parameters.get()

    device = get_device()
    model = get_model(trainer_parameters)

    callbacks = get_callbacks(trainer_parameters)

    check_test_batch(model, test_loader, device)

    trainer = L.Trainer(
        **trainer_parameters.trainer_config,
        accelerator=device,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model = get_model_to_test(callbacks, model, trainer_parameters)
    test_metrics = trainer.test(model=model, dataloaders=test_loader)

    if wandb_logger is not None:
        wandb_logger.experiment.finish()

    return test_metrics  # type: ignore

import wandb

from fast_gnn_benchmark.models.node_classification import NodeClassificationModel
from fast_gnn_benchmark.schemas.data_models import DataLoaderTypeChoices, DataParameters
from fast_gnn_benchmark.schemas.dataset_models import SplitType


def get_run_id_from_name(run_name: str, entity: str, project: str) -> str:
    api = wandb.Api()
    runs_candidates = api.runs(f"{entity}/{project}", filters={"display_name": {"$eq": run_name}})

    if len(runs_candidates) == 0:
        raise ValueError(f"No run found for name {run_name}")

    # Return the most recent run id for a given run name
    id = sorted(runs_candidates, key=lambda r: r.created_at, reverse=True)[0].id

    return id


def load_model_from_wandb(run_id: str, project: str) -> NodeClassificationModel:
    weight_path = f"{project}/{run_id}/checkpoints/best.ckpt"
    model = NodeClassificationModel.load_from_checkpoint(weight_path, weights_only=False)
    return model


def get_test_data_loader(data_parameters: dict) -> DataLoaderTypeChoices:
    if data_parameters.get("data_loader_parameters") is not None:
        data_parameters["train_data_loader_parameters"] = data_parameters["data_loader_parameters"]
        data_parameters["val_data_loader_parameters"] = data_parameters["data_loader_parameters"]
        data_parameters["test_data_loader_parameters"] = data_parameters["data_loader_parameters"]

    data_parameters_model = DataParameters(**data_parameters)

    dataset = data_parameters_model.get_dataset()

    test_data_loader = data_parameters_model.get_data_loader(
        dataset, SplitType.TEST, data_parameters_model.test_data_loader_parameters
    )

    return test_data_loader

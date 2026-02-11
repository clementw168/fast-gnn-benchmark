from pprint import pprint
from typing import Any

import numpy as np

from fast_gnn_benchmark.trainer import do_run, fix_seed, get_trainer_parameters_from_config


def main(file_path: str, override_dict: dict[str, Any] = {}) -> None:
    trainer_parameters = get_trainer_parameters_from_config(file_path, override_dict)

    pprint(trainer_parameters.model_dump())
    print()

    if trainer_parameters.seed is not None:
        fix_seed(trainer_parameters.seed)
    else:
        print("No seed provided")

    test_metrics = []

    for run in range(trainer_parameters.n_runs):
        print(f"Run {run + 1} of {trainer_parameters.n_runs}")

        test_metrics.append(do_run(trainer_parameters))

    if not test_metrics:
        raise ValueError("No run was done")

    for data_loader_idx in range(len(test_metrics[0])):
        data_loader_metrics = {}
        for metric in test_metrics[0][data_loader_idx]:
            metric_values = [test_metrics[run][data_loader_idx][metric] for run in range(len(test_metrics))]

            data_loader_metrics[metric] = metric_values

        print(f"Results for data loader {data_loader_idx}:")
        for metric, values in data_loader_metrics.items():
            mean, std = np.mean(values), np.std(values)
            print(f"{metric} : {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    import argparse

    from fast_gnn_benchmark.utils import recursive_defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, required=False, default="configs/ogbn_products/sage.yml")
    parser.add_argument("--epochs", type=int, required=False, default=None)
    parser.add_argument("--lr", type=float, required=False, default=None)
    parser.add_argument("--train_parts", type=int, required=False, default=None)
    parser.add_argument("--val_test_parts", type=int, required=False, default=None)
    parser.add_argument("--drop_edge_ratio", type=float, required=False, default=None)
    parser.add_argument("--seed", type=int, required=False, default=None)
    parser.add_argument("--tag", type=str, required=False, default=None)
    args = parser.parse_args()

    override_dict = recursive_defaultdict()
    if args.train_parts is not None:
        override_dict["data_parameters"]["train_data_loader_parameters"]["num_parts"] = args.train_parts

    if args.drop_edge_ratio is not None:
        override_dict["data_parameters"]["train_data_loader_parameters"]["drop_edge_ratio"] = args.drop_edge_ratio

    if args.val_test_parts is not None:
        override_dict["data_parameters"]["val_data_loader_parameters"]["num_parts"] = args.val_test_parts
        override_dict["data_parameters"]["test_data_loader_parameters"]["num_parts"] = args.val_test_parts

    if args.lr is not None:
        override_dict["model_parameters"]["optimizer"]["parameters"]["lr"] = args.lr

    if args.seed is not None:
        override_dict["seed"] = args.seed

    if args.epochs is not None:
        override_dict["trainer_config"]["max_epochs"] = args.epochs

    if args.tag is not None:
        override_dict["wandb_logger_parameters"]["tags"] = [args.tag]

    file_path = args.config_file
    main(file_path, override_dict)

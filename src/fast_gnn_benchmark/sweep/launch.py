import argparse
from pprint import pprint
from typing import Any

import yaml

import wandb


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        type=str,
        required=True,
        help="Path to the sweep config file. For example: configs/sweep/example.yml",
    )
    parser.add_argument("--project", "-p", type=str, default="gnn_experiments")
    parser.add_argument("--entity", "-e", type=str, default="clement_wang")
    return parser.parse_args()


def get_sweep_config(sweep_config_file: str) -> dict[str, Any]:
    with open(sweep_config_file, "r") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    args = arg_parser()
    sweep_config_file = args.config_file
    project = args.project
    entity = args.entity

    sweep_config = get_sweep_config(sweep_config_file)
    pprint(sweep_config)
    sweep_id = wandb.sweep(
        sweep_config,
        project=project,
        entity=entity,
    )

    print()

    print("Run the following command to launch the sweep:")
    print(f"uv run wandb agent {entity}/{project}/{sweep_id}")

    print()

    print("To stop the sweep, run the following command:")
    print(f"uv run wandb sweep --stop {entity}/{project}/{sweep_id}")

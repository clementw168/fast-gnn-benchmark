from pprint import pprint

import wandb
from fast_gnn_benchmark.trainer import do_run, get_trainer_parameters_from_config

if __name__ == "__main__":
    duplicate_data_loader_parameters = True
    with wandb.init() as run:
        config = run.config

        base_config_file = config["base_config_file"]

        if duplicate_data_loader_parameters and "data_loader_parameters" in config:
            config["train_data_loader_parameters"] = config["data_loader_parameters"]
            config["val_data_loader_parameters"] = config["data_loader_parameters"]
            config["test_data_loader_parameters"] = config["data_loader_parameters"]

        base_config = get_trainer_parameters_from_config(base_config_file, config)  # type: ignore

        pprint(base_config.model_dump())

        do_run(base_config)

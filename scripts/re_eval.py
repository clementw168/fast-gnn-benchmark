import argparse

import lightning as L

from fast_gnn_benchmark.trainer import get_device, get_global_config
from fast_gnn_benchmark.wandb_utils import get_run_id_from_name, get_test_data_loader, load_model_from_wandb

if __name__ == "__main__":
    global_config = get_global_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--entity", type=str, default=global_config["wandb_logger_parameters"]["entity"])
    parser.add_argument("--project", type=str, default=global_config["wandb_logger_parameters"]["project"])
    args = parser.parse_args()

    # data_parameters = {
    #     "dataset_type": "ogbn-products",
    #     "add_self_loops_and_remove_duplicate_edges": False,
    #     "remove_duplicate_edges": False,
    #     "data_loader_parameters": {
    #         "data_loader_type": "optimized_random_node_loader",
    #         "num_parts": 1,
    #         "pin_memory": False,
    #         "on_device": True,
    #     },
    # }

    # data_parameters = {
    #     "dataset_type": "ogbn-products",
    #     "add_self_loops_and_remove_duplicate_edges": False,
    #     "remove_duplicate_edges": False,
    #     "data_loader_parameters": {
    #         "data_loader_type": "ppr_node_loader",
    #         "num_parts": 1,
    #         "node_budget": 1000000000,
    #         "pin_memory": False,
    #         "on_device": True,
    #     },
    # }

    data_parameters = {
        "dataset_type": "ogbn-papers100M-on-ram",
        "add_self_loops_and_remove_duplicate_edges": False,
        "remove_duplicate_edges": False,
        "data_loader_parameters": {
            "data_loader_type": "neighbor_loader",
            "batch_size": 1024,
            "num_neighbors": [20, 20, 10, 5, 5],
            "num_workers": 64,
            "pin_memory": False,
            "persistent_workers": False,
            "on_device": False,
        },
    }

    run_id = get_run_id_from_name(args.run_name, args.entity, args.project)

    model = load_model_from_wandb(run_id, args.project)
    model = model.eval()
    print(model)

    test_data_loader = get_test_data_loader(data_parameters)

    trainer = L.Trainer(
        accelerator=get_device(),
    )
    trainer.test(model=model, dataloaders=test_data_loader)

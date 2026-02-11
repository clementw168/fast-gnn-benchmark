import argparse
import gc
import pathlib

import lightning as L
import pandas as pd

from fast_gnn_benchmark.trainer import get_device
from fast_gnn_benchmark.wandb_utils import get_run_id_from_name, get_test_data_loader, load_model_from_wandb


def append_result_row(row: dict, results_file: str):
    file_exists = pathlib.Path(results_file).exists()
    pd.DataFrame([row]).to_csv(
        results_file,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_done_keys(results_file: str):
    # We use (num_parts, node_budget) as a deterministic integer key to avoid float quirks
    if not pathlib.Path(results_file).exists():
        return set()
    df_prev = pd.read_csv(results_file)
    if df_prev.empty:
        return set()
    return set(zip(df_prev["num_parts"].astype(int), df_prev["node_budget"].astype(int)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="rare-thunder-1874")
    parser.add_argument("--entity", type=str, default="clement_wang")
    parser.add_argument("--project", type=str, default="gnn_experiments")
    parser.add_argument("--results_file", type=str, default="ppr_loader_params_test.csv")
    args = parser.parse_args()

    run_id = get_run_id_from_name(args.run_name, args.entity, args.project)

    model = load_model_from_wandb(run_id, args.project)
    model = model.eval()

    test_nodes = 2213091

    num_parts_list = [2, 3, 4, 7, 10, 15, 20, 30, 40, 50]
    multiplier_list = [1, 1.2, 1.5, 1.8, 2, 2.5, 3, 3.5, 4]

    data_parameters = {
        "dataset_type": "ogbn-products",
        "add_self_loops_and_remove_duplicate_edges": False,
        "remove_duplicate_edges": False,
        "data_loader_parameters": {
            "data_loader_type": "ppr_node_loader",
            "num_parts": 1,
            "node_budget": 1000000000,
            "pin_memory": False,
            "on_device": True,
        },
    }

    done_keys = load_done_keys(args.results_file)

    for num_parts in num_parts_list:
        for multiplier in multiplier_list:
            test_nodes_per_part = test_nodes // num_parts
            node_budget = int(test_nodes_per_part * multiplier)

            if multiplier > num_parts:
                continue

            key = (num_parts, node_budget)
            if key in done_keys:
                print(f"Skipping {key} because it has already been run")
                continue

            print(f"Running for num_parts={num_parts} and multiplier={multiplier}")
            data_parameters["data_loader_parameters"]["num_parts"] = num_parts
            data_parameters["data_loader_parameters"]["node_budget"] = node_budget
            test_data_loader = get_test_data_loader(data_parameters)
            trainer = L.Trainer(
                accelerator=get_device(),
            )
            performance = trainer.test(model=model, dataloaders=test_data_loader)

            row = {
                "num_parts": num_parts,
                "node_budget": node_budget,
                "multiplier": multiplier,
                "accuracy": performance[0].get("test/accuracy"),
                "loss": performance[0].get("test/loss"),
            }
            append_result_row(row, args.results_file)

            del test_data_loader
            del trainer
            del performance
            gc.collect()

    print(f"Done. Results saved to {args.results_file}")

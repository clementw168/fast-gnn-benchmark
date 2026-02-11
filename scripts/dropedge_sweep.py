import subprocess

if __name__ == "__main__":
    datasets = ["ogbn_arxiv", "ogbn_products", "pokec"]
    drop_edge_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for dataset_idx, dataset in enumerate(datasets):
        for drop_edge_ratio_idx, drop_edge_ratio in enumerate(drop_edge_ratios):
            print()
            print(f"Running for dataset={dataset} and drop_edge_ratio={drop_edge_ratio}")
            print(
                f"({dataset_idx * len(drop_edge_ratios) + drop_edge_ratio_idx + 1}/{len(datasets) * len(drop_edge_ratios)})"
            )
            subprocess.run(
                [
                    "uv",
                    "run",
                    "src/fast_gnn_benchmark/main.py",
                    "--config_file",
                    f"configs/{dataset}/sage_dropedge.yml",
                    "--drop_edge_ratio",
                    str(drop_edge_ratio),
                    "--seed",
                    str(42),
                    "--tag",
                    f"dropedge_{dataset}_sweep",
                ],
                check=False,
            )
            print(
                f"Done for dataset={dataset} and drop_edge_ratio={drop_edge_ratio} ({dataset_idx * len(drop_edge_ratios) + drop_edge_ratio_idx + 1}/{len(datasets) * len(drop_edge_ratios)})"
            )
            print("________________________________________________________")
            print()

import subprocess

datasets = ["ogbn_arxiv", "ogbn_products", "pokec"]


if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 101112]
    for seed_idx, seed in enumerate(seeds):
        for dataset_idx, dataset in enumerate(datasets):
            print()
            print(f"Running for dataset={dataset} and seed={seed}")
            print(f"({seed_idx * len(datasets) + dataset_idx + 1}/{len(seeds) * len(datasets)})")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "src/fast_gnn_benchmark/main.py",
                    "--config_file",
                    f"configs/{dataset}/sage_dropedge.yml",
                    "--seed",
                    str(seed),
                    "--tag",
                    f"dropedge_benchmark_{dataset}",
                ],
                check=False,
            )
            print(
                f"Done for dataset={dataset} and seed={seed} ({dataset_idx * len(seeds) + seed_idx + 1}/{len(datasets) * len(seeds)})"
            )
            print("________________________________________________________")
            print()
